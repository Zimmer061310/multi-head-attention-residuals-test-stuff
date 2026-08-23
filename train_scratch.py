"""
Train Qwen3 / Qwen3-AttnRes from scratch on FineWeb-Edu.

Usage:
    # Baseline (no AttnRes)
    torchrun --nproc_per_node=8 train_scratch.py --mode baseline

    # Block AttnRes
    torchrun --nproc_per_node=8 train_scratch.py --mode block

    # Full AttnRes
    torchrun --nproc_per_node=8 train_scratch.py --mode full
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Attention-Residuals"))

from modeling_qwen3_attnres import Qwen3AttnResConfig, Qwen3AttnResForCausalLM, enable_compile as enable_attnres_compile
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="baseline", choices=["baseline", "moe", "block", "block_v", "block_mh", "full", "full_mh", "full_hw", "hyper_connection", "full_split", "full_split_shared", "full_split_shared_v", "full_v", "depth_attn", "full_additive", "delta", "delta_sublayer", "delta_centered", "delta_centered_block", "delta_centered_reset", "delta_avg_block", "delta_replace", "delta_replace_block", "delta_block", "delta_block_v", "delta_v", "first_layer", "pre_gated"],
                   help="baseline, block, block_v, full, full_mh, full_hw, full_split, full_split_shared, full_v, delta, delta_block, delta_block_v, delta_v, first_layer, pre_gated")
    p.add_argument("--hyper_n", type=int, default=4,
                   help="Expansion rate (number of parallel streams) for hyper_connection mode")
    p.add_argument("--moe_experts", type=int, default=8,
                   help="moe mode: number of experts")
    p.add_argument("--moe_topk", type=int, default=2,
                   help="moe mode: experts per token")
    p.add_argument("--moe_ff", type=int, default=768,
                   help="moe mode: per-expert intermediate size (topk*moe_ff ~ dense ff for iso-activated-FLOPs)")
    p.add_argument("--attnres_heads", type=int, default=8,
                   help="Routing heads for full_mh, and for full_hw's MLP/final routing (hidden_size must be divisible)")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)
    p.add_argument("--num_blocks", type=int, default=4,
                   help="Number of AttnRes blocks (for block mode)")
    p.add_argument("--gate_type", default="bias",
                   choices=["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"],
                   help="Gate type for mixing AttnRes output with residual stream")
    p.add_argument("--null_source", action="store_true",
                   help="Add null source for identity init")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="default")
    p.add_argument("--dataset_revision", default=None,
                   help="Immutable Hugging Face dataset commit for reproducibility")
    p.add_argument("--data_files", default=None,
                   help="glob of local parquet shards (overrides --dataset), "
                        "e.g. /mnt/localssd/data/anneal_pt_v3/*.parquet")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--expected_global_batch", type=int, default=None,
                   help="Fail if batch_size * grad_accum * world_size differs")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=500,
                   help="Run validation every N steps (0 to disable)")
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Number of batches for validation")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--wandb_project", default="residual")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_required", action="store_true",
                   help="Fail instead of training untracked if W&B initialization fails")
    p.add_argument("--run_name", default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer_revision", default=None,
                   help="Immutable tokenizer/model commit for reproducibility")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile on AttnRes kernels for faster training")
    p.add_argument("--fused", action="store_true",
                   help="Fused Triton MHAR routing kernels (full_mh only): shared source "
                        "buffer + single fwd/bwd kernel per routing call")
    p.add_argument("--compile_model", action="store_true",
                   help="torch.compile the entire model (fuses attention+MLP+routing)")
    p.add_argument("--fsdp", action="store_true",
                   help="Use FSDP full-shard (ZeRO-3) instead of DDP — required for 7B+")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Activation checkpointing on each decoder layer (cuts memory for 7B+)")
    p.add_argument("--resume_from", default=None,
                   help="Checkpoint directory containing training_state.pt")
    p.add_argument("--keep_last", type=int, default=2,
                   help="Number of resumable step checkpoints to retain")
    return p.parse_args()


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def training_identity(args, world_size):
    """Fields that must remain fixed for a scientifically exact resume."""

    return {
        "source_commit": git_commit(),
        "mode": args.mode,
        "attnres_heads": args.attnres_heads,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "intermediate_size": args.intermediate_size,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "data_files": args.data_files,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "per_gpu_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "world_size": world_size,
        "global_batch_size": args.batch_size * args.grad_accum * world_size,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "warmup": args.warmup,
        "max_norm": args.max_norm,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-8,
        "weight_decay": 0.1,
        "precision": "bf16",
        "seed": args.seed,
        "fused": args.fused,
        "compile": args.compile,
        "compile_model": args.compile_model,
        "fsdp": args.fsdp,
        "grad_ckpt": args.grad_ckpt,
    }


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_training_state(checkpoint_dir):
    path = Path(checkpoint_dir).resolve() / "training_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing resumable training state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "global_step", "chunks_consumed", "optimizer", "scheduler",
        "cpu_rng_state", "cuda_rng_state_all", "run_identity",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"training state is missing fields: {sorted(missing)}")
    return state


def validate_model_config(model, args):
    expected = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "num_key_value_heads": args.num_kv_heads,
        "intermediate_size": args.intermediate_size,
        "head_dim": args.hidden_size // args.num_heads,
        "max_position_embeddings": args.seq_len * 2,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if args.mode not in ("baseline", "moe"):
        expected.update({
            "attnres_num_heads": args.attnres_heads,
            "attnres_mode": args.mode,
        })
    observed = {key: getattr(model.config, key, None) for key in expected}
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"checkpoint/model configuration mismatch: {mismatches}")
    return observed


def token_stream(dataset_name, config_name, tokenizer, seq_len, rank, world_size, seed,
                 data_files=None, dataset_revision=None, skip_chunks=0):
    import time
    from datasets import load_dataset

    def make_ds(n_skip):
        if data_files:
            ds = load_dataset("parquet", data_files=data_files, split="train",
                              streaming=True)
        else:
            ds = load_dataset(dataset_name, name=config_name, split="train",
                              revision=dataset_revision, streaming=True)
        ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)
        return ds.skip(rank + n_skip)

    # Transient HF CDN errors (e.g. 408 mid-stream) bypass datasets' built-in
    # retries; reconnect and fast-forward past the samples already consumed.
    n_consumed = 0
    n_chunks = 0
    buf = []
    while True:
        try:
            for sample in make_ds(n_consumed):
                n_consumed += 1
                text = sample.get("text") or sample.get("content") or sample.get("wikitext") or ""
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(tokenizer.eos_token_id)
                buf.extend(ids)
                while len(buf) >= seq_len + 1:
                    chunk = buf[:seq_len + 1]
                    buf = buf[world_size * seq_len:]
                    n_chunks += 1
                    if n_chunks > skip_chunks:
                        yield torch.tensor(chunk, dtype=torch.long)
            return
        except Exception as e:
            print(f"[rank {rank}] data stream error after {n_consumed} samples: "
                  f"{e!r} — reconnecting in 30s", flush=True)
            time.sleep(30)


def build_model(args, device):
    """Build model from scratch based on mode."""
    if args.resume_from:
        if args.fsdp:
            raise NotImplementedError("--resume_from is currently supported for DDP runs")
        if args.mode == "baseline":
            model_class = Qwen3ForCausalLM
        elif args.mode == "moe":
            from transformers import Qwen3MoeForCausalLM
            model_class = Qwen3MoeForCausalLM
        else:
            model_class = Qwen3AttnResForCausalLM
        model = model_class.from_pretrained(
            args.resume_from, dtype=torch.bfloat16).to(device=device)
        validate_model_config(model, args)
        return model

    common = dict(
        vocab_size=151936,  # Qwen3 tokenizer vocab
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_len * 2,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=args.hidden_size // args.num_heads,
    )

    if args.mode == "baseline":
        config = Qwen3Config(**common)
        model = Qwen3ForCausalLM(config)
    elif args.mode == "moe":
        # Stock Qwen3-MoE, attention identical to baseline; iso-activated-FLOPs
        # MLP (topk * moe_ff == dense intermediate_size). Aux load-balancing
        # loss is train-only; evaluate() reports pure CE for moe.
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        config = Qwen3MoeConfig(
            num_experts=args.moe_experts,
            num_experts_per_tok=args.moe_topk,
            moe_intermediate_size=args.moe_ff,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            norm_topk_prob=True,
            output_router_logits=True,
            router_aux_loss_coef=0.001,
            **common,
        )
        model = Qwen3MoeForCausalLM(config)
    else:
        config = Qwen3AttnResConfig(
            attnres_num_blocks=args.num_blocks,

            attnres_mode=args.mode,
            attnres_gate_type=args.gate_type,
            attnres_use_null_source=args.null_source,
            attnres_num_heads=args.attnres_heads,
            attnres_hyper_n=args.hyper_n,
            **common,
        )
        model = Qwen3AttnResForCausalLM(config)

    validate_model_config(model, args)

    if getattr(args, "fsdp", False):
        # Keep params fp32; FSDP MixedPrecision does bf16 compute and keeps an
        # fp32 sharded master for the optimizer (stable + memory-fits when sharded).
        model = model.to(device=device)
    else:
        model = model.to(dtype=torch.bfloat16, device=device)
    return model


def unwrap_model(model):
    value = model.module if hasattr(model, "module") else model
    return value._orig_mod if hasattr(value, "_orig_mod") else value


def prune_step_checkpoints(out_dir, keep_last):
    if keep_last < 1:
        raise ValueError("--keep_last must be at least 1")
    out_dir = Path(out_dir).resolve()
    checkpoints = []
    for candidate in out_dir.iterdir():
        match = re.fullmatch(r"step-(\d+)", candidate.name)
        if match and candidate.is_dir():
            checkpoints.append((int(match.group(1)), candidate))
    for _, candidate in sorted(checkpoints)[:-keep_last]:
        shutil.rmtree(candidate)


def save_training_checkpoint(
    *, model, tokenizer, optimizer, scheduler, global_step, chunks_consumed,
    run_identity, out_dir, keep_last, wandb_run_id, elapsed_training_seconds,
    final=False,
):
    """Atomically save weights plus optimizer/scheduler/RNG state."""

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "final" if final else f"step-{global_step}"
    destination = out_dir / name
    temporary = out_dir / f".{name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {destination}")
    temporary.mkdir()

    unwrap_model(model).save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    state = {
        "format_version": 1,
        "created_at": utc_now(),
        "global_step": global_step,
        "chunks_consumed": chunks_consumed,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "run_identity": run_identity,
        "wandb_run_id": wandb_run_id,
        "elapsed_training_seconds": elapsed_training_seconds,
    }
    state_path = temporary / "training_state.pt"
    state_temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, state_temporary)
    os.replace(state_temporary, state_path)
    atomic_write_json(temporary / "training_manifest.json", {
        "format_version": 1,
        "created_at": state["created_at"],
        "global_step": global_step,
        "chunks_consumed": chunks_consumed,
        "elapsed_training_seconds": elapsed_training_seconds,
        "run_identity": run_identity,
        "wandb_run_id": wandb_run_id,
    })
    os.replace(temporary, destination)
    if not final:
        prune_step_checkpoints(out_dir, keep_last)
    return destination


def main():
    args = parse_args()

    if args.keep_last < 1:
        raise ValueError("--keep_last must be at least 1")

    if args.run_name is None:
        args.run_name = f"scratch-{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"
    if args.out_dir is None:
        args.out_dir = f"./output/scratch-{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"

    # ── distributed ──
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    run_identity = training_identity(args, world_size)
    if (args.expected_global_batch is not None
            and run_identity["global_batch_size"] != args.expected_global_batch):
        raise ValueError(
            f"expected global batch {args.expected_global_batch}, got "
            f"{run_identity['global_batch_size']}")
    resume_state = load_training_state(args.resume_from) if args.resume_from else None
    if resume_state is not None and resume_state["run_identity"] != run_identity:
        raise RuntimeError(
            "resume checkpoint identity does not match this invocation:\n"
            f"saved={json.dumps(resume_state['run_identity'], sort_keys=True)}\n"
            f"current={json.dumps(run_identity, sort_keys=True)}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = out_dir / "training_run_manifest.json"
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest["run_identity"] != run_identity:
            raise RuntimeError("output directory belongs to a different training run")
        if resume_state is None:
            raise RuntimeError(
                "output directory already contains a run manifest; pass --resume_from "
                "with a saved checkpoint or choose a new --out_dir")
    else:
        manifest = {
            "format_version": 1,
            "created_at": utc_now(),
            "run_identity": run_identity,
            "command": sys.argv,
        }
        if is_main:
            atomic_write_json(run_manifest_path, manifest)
    dist.barrier()

    # ── W&B ──
    use_wandb = False
    wandb_run_id = resume_state.get("wandb_run_id") if resume_state else None
    if is_main:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                job_type="pretraining",
                tags=["mhar", "full-mh", "h4", "1b", "fineweb-edu"],
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                name=args.run_name,
                config={**vars(args), **run_identity},
            )
            wandb_run_id = wandb.run.id
            use_wandb = True
        except Exception as e:
            if args.wandb_required:
                raise
            print(f"W&B init failed ({e}), continuing without logging")
    if is_main:
        manifest["wandb"] = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "run_name": args.run_name,
            "run_id": wandb_run_id,
            "required": args.wandb_required,
        }
        atomic_write_json(run_manifest_path, manifest)

    # ── model ──
    if is_main:
        print(f"Building {args.mode} model from scratch...")

    model = build_model(args, device)

    if args.compile and args.mode != "baseline":
        enable_attnres_compile()
        if is_main:
            print("torch.compile enabled for AttnRes kernels")

    if args.fused:
        if args.mode != "full_mh":
            raise ValueError("--fused currently supports --mode full_mh only")
        from modeling_qwen3_attnres import enable_fused_mhar
        enable_fused_mhar(True)
        if is_main:
            print("fused Triton MHAR routing kernels enabled")

    if args.grad_ckpt and not args.fsdp:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if is_main:
            print("activation checkpointing enabled on decoder layers")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.1f}M params | mode={args.mode} | d={args.hidden_size} L={args.num_layers}")
        if args.mode != "baseline":
            n_attnres = sum(p.numel() for n, p in model.named_parameters() if "res_" in n)
            print(f"AttnRes params: {n_attnres/1e3:.1f}K")

    # torch.compile the full model before DDP wrapping.
    # Gives ~2.5-2.9x throughput improvement for all modes.
    if args.compile_model:
        model = torch.compile(model)
        if is_main:
            print("torch.compile enabled for full model")

    if args.fsdp:
        import functools
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        # Wrap each transformer decoder layer (works for baseline Qwen3 and AttnRes).
        layer_cls = type(model.model.layers[0])
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
        mp = MixedPrecision(param_dtype=torch.bfloat16,
                            reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        model = FSDP(model, auto_wrap_policy=wrap_policy, mixed_precision=mp,
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     device_id=local_rank, sync_module_states=True,
                     use_orig_params=True)
        if args.grad_ckpt:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda m: isinstance(m, layer_cls))
            if is_main:
                print("activation checkpointing enabled on decoder layers")
        if is_main:
            print(f"FSDP FULL_SHARD enabled, wrapping {layer_cls.__name__}")
    else:
        if args.mode == "moe":
            # Plain default DDP: with topk*batch tokens every expert is hit
            # each micro-batch (all params used), and both find_unused and
            # static_graph deadlock against grad-accumulation's repeated
            # backwards here.
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    lr_min_ratio = args.lr_min / args.lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio),
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    # ── data ──
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision)
    resume_chunks = int(resume_state["chunks_consumed"]) if resume_state else 0
    stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                          args.seq_len, rank, world_size, args.seed,
                          data_files=args.data_files,
                          dataset_revision=args.dataset_revision,
                          skip_chunks=resume_chunks)

    # ── validation data (use a different seed to avoid overlap) ──
    val_stream = None
    if args.eval_every > 0:
        val_stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                                  args.seq_len, rank, world_size, args.seed + 9999,
                                  data_files=args.data_files,
                                  dataset_revision=args.dataset_revision)

    @torch.no_grad()
    def evaluate(val_iter, n_steps):
        model.eval()
        total_loss = 0.0
        count = 0
        for _ in range(n_steps):
            batch_chunks = []
            for _ in range(args.batch_size):
                try:
                    chunk = next(val_iter)
                    batch_chunks.append(chunk[:-1])
                except StopIteration:
                    break
            if not batch_chunks:
                break
            input_ids = torch.stack(batch_chunks).to(device)
            labels = input_ids
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            if args.mode == "moe":
                # out.loss includes the router aux term; report pure CE so val
                # is comparable to the dense baseline.
                import torch.nn.functional as F
                lg = out.logits[:, :-1].float()
                ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                     labels[:, 1:].reshape(-1))
                total_loss += ce.item()
            else:
                total_loss += out.loss.item()
            count += 1
        model.train()
        avg = torch.tensor(total_loss / max(count, 1), device=device)
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        return avg.item()

    # ── training ──
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    val_iter = iter(val_stream) if val_stream is not None else None

    global_step = int(resume_state["global_step"]) if resume_state else 0
    accum_step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0
    chunks_consumed = resume_chunks
    elapsed_before_resume = (
        float(resume_state.get("elapsed_training_seconds", 0.0)) if resume_state else 0.0)
    run_started = time.time() - elapsed_before_resume
    if resume_state is not None:
        torch.set_rng_state(resume_state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        if is_main:
            print(
                f"Resumed exact state at step {global_step}; replay-skipping "
                f"{resume_chunks} packed chunks", flush=True)

    batch_buf = []
    for chunk in stream:
        if global_step >= args.steps:
            break

        chunks_consumed += 1
        batch_buf.append(chunk[:-1])
        if len(batch_buf) < args.batch_size:
            continue

        input_ids = torch.stack(batch_buf).to(device)  # [batch_size, seq_len]
        labels = input_ids
        batch_buf = []

        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss / args.grad_accum
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step={global_step} accum={accum_step}: {loss.item()}")
        loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        tokens_seen += args.seq_len * args.batch_size

        if accum_step < args.grad_accum:
            continue

        if args.fsdp:
            grad_norm = model.clip_grad_norm_(args.max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        accum_step = 0

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_main:
                elapsed = time.time() - t0
                tok_sec = tokens_seen * world_size / elapsed
                avg_loss = loss_t.item()
                lr_now = scheduler.get_last_lr()[0]
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {global_step:6d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | grad_norm {grad_norm:.3f} | "
                      f"{tok_sec/1e3:.1f}k tok/s | {mem_gb:.1f}GB")

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/grad_norm": grad_norm,
                        "train/tok_per_s": tok_sec,
                        "train/peak_cuda_allocated_gib": (
                            torch.cuda.max_memory_allocated() / 2**30),
                        "train/chunks_consumed": chunks_consumed,
                        "train/tokens_consumed": (
                            global_step * run_identity["global_batch_size"] * args.seq_len),
                        "train/elapsed_hours": (time.time() - run_started) / 3600,
                    }, step=global_step)

                tokens_seen = 0
                t0 = time.time()
        accum_loss = 0.0

        if not args.fsdp and is_main and global_step % args.save_every == 0:
            ckpt_dir = save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                chunks_consumed=chunks_consumed,
                run_identity=run_identity,
                out_dir=out_dir,
                keep_last=args.keep_last,
                wandb_run_id=wandb_run_id,
                elapsed_training_seconds=time.time() - run_started,
            )
            print(f"Saved checkpoint → {ckpt_dir}")
            if use_wandb:
                import wandb
                wandb.run.summary["latest_checkpoint"] = str(ckpt_dir)
                wandb.run.summary["latest_checkpoint_step"] = global_step

        if args.eval_every > 0 and global_step % args.eval_every == 0 and val_iter is not None:
            val_loss = evaluate(val_iter, args.eval_steps)
            if is_main:
                import math as _math
                val_ppl = _math.exp(val_loss) if val_loss < 20 else float('inf')
                print(f"step {global_step:6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    # ── final validation ──
    if args.eval_every > 0 and val_iter is not None:
        val_loss = evaluate(val_iter, args.eval_steps)
        if is_main:
            import math as _math
            val_ppl = _math.exp(val_loss) if val_loss < 20 else float('inf')
            print(f"FINAL   | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
            if use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    if is_main:
        final_dir = save_training_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            chunks_consumed=chunks_consumed,
            run_identity=run_identity,
            out_dir=out_dir,
            keep_last=args.keep_last,
            wandb_run_id=wandb_run_id,
            elapsed_training_seconds=time.time() - run_started,
            final=True,
        )
        print(f"Training done. Final model → {final_dir}")
        if use_wandb:
            import wandb
            wandb.run.summary["final_checkpoint"] = str(final_dir)
            wandb.run.summary["completed_steps"] = global_step
            wandb.run.summary["total_tokens"] = (
                global_step * run_identity["global_batch_size"] * args.seq_len)
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
