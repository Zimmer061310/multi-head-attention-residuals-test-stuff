"""
Continual pretraining (WSD-style resume-and-decay) from a pre-decay base checkpoint.

Loads a pretrained base (default: Marin 8B @ revision `phoenix`, a constant-LR
WSD-S plateau), then DECAYS the LR starting from the base's terminal plateau LR
down to a small floor over an "anneal" token budget, on a local parquet corpus.

Because no PyTorch-loadable Adam moments ship with these bases, AdamW is COLD-STARTED;
a short LR re-warmup (0 -> plateau LR) rebuilds the 2nd moment before the decay,
avoiding the loss spike you'd get cold-starting straight at a high plateau LR.

Usage (single 8xH100 node, FSDP full-shard):
    torchrun --nproc_per_node=8 --module src.training.train_cpt --fsdp \
        --init_from marin-community/marin-8b-base --init_revision phoenix \
        --data_files '/mnt/localssd/data/anneal_pt_v3/*.parquet' \
        --seq_len 4096 --lr 1.7e-3 --lr_min 1.7e-5 --rewarmup 500 --decay_shape one_minus_sqrt \
        --steps 9500 --batch_size 4 --grad_accum 16
"""

import argparse
import glob as _glob
import math
import os
import sys
import time
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    # base model to continue from
    p.add_argument("--init_from", default="marin-community/marin-8b-base",
                   help="HF repo id of the pre-decay base checkpoint")
    p.add_argument("--init_revision", default="phoenix",
                   help="git tag/branch/revision of the pre-decay checkpoint")
    # data
    p.add_argument("--data_files", required=True,
                   help="glob of local parquet shards, e.g. /mnt/localssd/data/anneal_pt_v3/*.parquet")
    p.add_argument("--seq_len", type=int, default=4096)
    # WSD resume-decay schedule
    p.add_argument("--lr", type=float, default=1.7e-3,
                   help="PEAK LR = the base model's terminal plateau LR to start decaying from")
    p.add_argument("--lr_min", type=float, default=1.7e-5,
                   help="floor LR at end of decay")
    p.add_argument("--attnres_lr", type=float, default=None,
                   help="separate peak LR for the AttnRes adaptation params (_res_/null_source); "
                        "follows the same warmup/decay shape (floor scales by lr_min/lr). "
                        "None = use --lr for everything")
    p.add_argument("--rewarmup", type=int, default=500,
                   help="short LR re-warmup steps (0 -> lr) to rebuild Adam 2nd moment on cold start")
    p.add_argument("--decay_shape", default="one_minus_sqrt",
                   choices=["one_minus_sqrt", "linear"],
                   help="cooldown shape; 1-sqrt slightly beats linear (Hagele et al.)")
    # budget / batch
    p.add_argument("--steps", type=int, default=9500)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU micro-batch")
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.05,
                   help="match Marin (0.05); applied only to 2D weights, NOT embeddings/norms/bias")
    p.add_argument("--z_loss", type=float, default=1e-4,
                   help="z-loss coefficient on final logits (match Marin's cooldown); 0 to disable")
    p.add_argument("--ema", type=float, default=0.999,
                   help="weight-EMA decay; checkpoints save the EMA weights (Marin ships EMA). 0 to disable")
    # logging / ckpt
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--wandb_project", default="residual")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--seed", type=int, default=0)
    # distributed / memory
    p.add_argument("--fsdp", action="store_true", help="FSDP full-shard (required for 7B+)")
    p.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing per decoder layer")
    # AttnRes conversion: continue-train the transformer INTO a multi-head delta
    # attention-residual model (transformer -> MHAR). Empty = plain CPT.
    p.add_argument("--attnres_mode", default="",
                   help="'' = plain CPT; 'delta_mh' = convert to multi-head delta attention "
                        "residual (identity init); 'full_mh' = convert to MHAR with pretrain-"
                        "style init (random routing, no null source, no identity trick)")
    p.add_argument("--attnres_num_heads", type=int, default=32,
                   help="per-head depth-routing subspaces for delta_mh (hidden_size must divide this)")
    p.add_argument("--attnres_num_blocks", type=int, default=8,
                   help="delta blocks; layers_per_block = num_layers / this")
    p.add_argument("--attnres_gate_type", default="learnable_alpha",
                   help="'learnable_alpha' = zero-init gate (alpha=0) -> EXACT identity at step 0")
    p.add_argument("--no_compile", action="store_true",
                   help="disable torch.compile of the AttnRes routing kernels (fallback to eager)")
    p.add_argument("--fused", action="store_true",
                   help="fused Triton routing kernels (delta_mh): one fwd/bwd kernel per "
                        "routing call; composes with FSDP and --grad_ckpt")
    return p.parse_args()


def lr_factor(step, rewarmup, total, lr_min_ratio, shape):
    """Decay-only schedule that PEAKS at the plateau LR (factor 1.0) after a short
    re-warmup, then cools to lr_min_ratio."""
    if step < rewarmup:
        return step / max(1, rewarmup)
    progress = (step - rewarmup) / max(1, total - rewarmup)
    progress = min(1.0, max(0.0, progress))
    if shape == "one_minus_sqrt":
        s = 1.0 - math.sqrt(progress)
    else:  # linear
        s = 1.0 - progress
    return lr_min_ratio + (1.0 - lr_min_ratio) * s


def token_stream(data_files, tokenizer, seq_len, rank, world_size, seed):
    """Stream local parquet shards, tokenize raw text with the BASE model's tokenizer,
    and pack into seq_len+1 windows. Sharding/recovery mirrors train_scratch.py."""
    from datasets import load_dataset

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id or 0

    def make_ds(n_skip):
        ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)
        return ds.skip(rank + n_skip)

    n_consumed = 0
    buf = []
    while True:
        try:
            for sample in make_ds(n_consumed):
                n_consumed += 1
                text = sample.get("text") or sample.get("content") or sample.get("wikitext") or ""
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(eos)
                buf.extend(ids)
                while len(buf) >= seq_len + 1:
                    chunk = buf[:seq_len + 1]
                    buf = buf[world_size * seq_len:]
                    yield torch.tensor(chunk, dtype=torch.long)
            return
        except Exception as e:
            print(f"[rank {rank}] data stream error after {n_consumed} samples: "
                  f"{e!r} — reconnecting in 30s", flush=True)
            time.sleep(30)


def build_model(args, device):
    dtype = torch.float32 if args.fsdp else torch.bfloat16
    if args.attnres_mode:
        # Convert a pretrained (Llama) transformer INTO a multi-head delta
        # attention-residual model. Load base weights strict=False so the new
        # routing params keep their identity init (query=0, null=0, gate alpha=0)
        # -> step 0 is bit-exact the base model; the gate opens during training.
        from src.attention_residuals import modeling_llama_attnres as _M
        from src.attention_residuals.modeling_llama_attnres import (
            LlamaAttnResConfig,
            LlamaAttnResForCausalLM,
        )
        if not args.no_compile:
            _M.enable_compile()   # torch.compile(dynamic=True) the AttnRes routing kernels
        if args.fused:
            _M.enable_fused_mhar(True)   # fused Triton path takes precedence over compile
            if int(os.environ.get("RANK", 0)) == 0:
                print("fused Triton routing kernels enabled (delta_mh/block_mh)")
        base = AutoModelForCausalLM.from_pretrained(
            args.init_from, revision=args.init_revision, torch_dtype=dtype)
        base_cfg = base.config.to_dict()
        base_cfg.pop("model_type", None)
        base_cfg.pop("architectures", None)
        cfg = LlamaAttnResConfig(**base_cfg)
        cfg.attnres_mode = args.attnres_mode
        cfg.attnres_num_heads = args.attnres_num_heads
        cfg.attnres_num_blocks = args.attnres_num_blocks
        cfg.attnres_gate_type = args.attnres_gate_type
        # Null source is part of the delta identity-init recipe; full_mh keeps the
        # pretrain configuration (no null source, routing params at random init).
        cfg.attnres_use_null_source = (args.attnres_mode == "delta_mh")
        model = LlamaAttnResForCausalLM(cfg)
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected, f"unexpected base keys: {unexpected[:8]}"
        bad = [k for k in missing if not any(t in k for t in
               ("res_proj", "res_norm", "null_source", "res_alpha"))]
        assert not bad, f"base keys missing from converted model: {bad[:8]}"
        model = model.to(dtype=dtype)
        del base
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.init_from, revision=args.init_revision, torch_dtype=dtype)
    if args.fsdp:
        model = model.to(device=device)          # keep fp32 for FSDP MixedPrecision master
    else:
        model = model.to(dtype=torch.bfloat16, device=device)
    return model


def fsdp_save(model, tokenizer, out_dir, is_main):
    """Gather a FULL (unsharded) state dict on rank0 and write a real HF checkpoint."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        sd = model.state_dict()                  # collective: all ranks call, rank0 gets full tensors
    if is_main:
        sd = {k: (v.bfloat16() if v.is_floating_point() else v) for k, v in sd.items()}
        hf = model.module                        # FSDP-wrapped HF model
        os.makedirs(out_dir, exist_ok=True)
        hf.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        print(f"Saved checkpoint -> {out_dir}", flush=True)
    dist.barrier()


def save_ckpt(model, ema, tokenizer, out_dir, is_main, fsdp):
    """Save a checkpoint. If an EMA shadow is given, save the EMA weights
    (swap EMA into the live params, gather+save, swap back)."""
    if not fsdp:
        if is_main:
            hf = model.module if hasattr(model, "module") else model
            if ema is not None:
                backup = [p.detach().clone() for p in hf.parameters()]
                with torch.no_grad():
                    for p, e in zip(hf.parameters(), ema):
                        p.data.copy_(e)
                hf.save_pretrained(out_dir)
                with torch.no_grad():
                    for p, b in zip(hf.parameters(), backup):
                        p.data.copy_(b)
                del backup
            else:
                hf.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"Saved checkpoint -> {out_dir}", flush=True)
        return
    if ema is None:
        fsdp_save(model, tokenizer, out_dir, is_main)
        return
    backup = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p, e in zip(model.parameters(), ema):
            p.data.copy_(e)
    fsdp_save(model, tokenizer, out_dir, is_main)
    with torch.no_grad():
        for p, b in zip(model.parameters(), backup):
            p.data.copy_(b)
    del backup


def main():
    args = parse_args()
    base_tag = f"{args.init_from.split('/')[-1]}-{args.init_revision}"
    if args.run_name is None:
        args.run_name = f"cpt-{base_tag}-{args.steps//1000}k"
    if args.out_dir is None:
        args.out_dir = f"./output/{args.run_name}"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    use_wandb = False
    if is_main:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.run_name, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging", flush=True)

    if is_main:
        print(f"Loading base {args.init_from}@{args.init_revision} ...", flush=True)
    model = build_model(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.init_from, revision=args.init_revision)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.1f}M params | base={base_tag} | "
              f"plateau_lr={args.lr} -> lr_min={args.lr_min} | rewarmup={args.rewarmup} "
              f"| decay={args.decay_shape} | seq={args.seq_len}", flush=True)

    if args.fsdp:
        import functools
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
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
                print("activation checkpointing enabled", flush=True)
        if is_main:
            print(f"FSDP FULL_SHARD enabled, wrapping {layer_cls.__name__}", flush=True)
    else:
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            if is_main:
                print("activation checkpointing enabled (DDP)", flush=True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                    gradient_as_bucket_view=True)   # bucket shares .grad memory (~16GB saved)

    # weight EMA shadow (sharded, same order as model.parameters()); saved at ckpt time
    ema = None
    if args.ema > 0:
        ema = [p.detach().clone() for p in model.parameters()]
        if is_main:
            print(f"weight EMA enabled (decay={args.ema}); checkpoints save EMA weights", flush=True)

    # Two param groups: weight decay on 2D weights only; exclude embeddings,
    # norms (RMSNorm/LayerNorm) and biases — matches Marin/Levanter's WD mask.
    decay, no_decay, attnres = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        # FSDP FULL_SHARD flattens params to 1D shards, so ndim is unreliable here —
        # split by NAME: exclude embeddings, norms and biases from weight decay.
        # The AttnRes adaptation params (routing query/norm/gate-alpha and null
        # source) get their own group: they start at identity (0) and must learn
        # WITHOUT decay pulling them back to 0, else the gate never opens; they
        # can also take a separate (much higher) LR via --attnres_lr.
        if "_res_" in nl or "null_source" in nl:
            attnres.append(p)
        elif "embed" in nl or "norm" in nl or "bias" in nl:
            no_decay.append(p)
        else:
            decay.append(p)
    attnres_lr = args.attnres_lr if args.attnres_lr is not None else args.lr
    if is_main:
        print(f"AdamW groups: decay={len(decay)} (wd={args.weight_decay}, lr={args.lr}), "
              f"no_decay={len(no_decay)} (embeddings/norms/bias, wd=0, lr={args.lr}), "
              f"attnres={len(attnres)} (wd=0, lr={attnres_lr})", flush=True)
    groups = [{"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
              {"params": no_decay, "weight_decay": 0.0, "lr": args.lr}]
    if attnres:
        groups.append({"params": attnres, "weight_decay": 0.0, "lr": attnres_lr})
    optimizer = AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    lr_min_ratio = args.lr_min / args.lr
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: lr_factor(
        s, args.rewarmup, args.steps, lr_min_ratio, args.decay_shape))

    stream = token_stream(args.data_files, tokenizer, args.seq_len, rank, world_size, args.seed)
    val_stream = None
    if args.eval_every > 0:
        val_stream = token_stream(args.data_files, tokenizer, args.seq_len,
                                  rank, world_size, args.seed + 9999)

    @torch.no_grad()
    def evaluate(val_iter, n_steps):
        model.eval()
        total_loss, count = 0.0, 0
        for _ in range(n_steps):
            chunks = []
            for _ in range(args.batch_size):
                try:
                    chunks.append(next(val_iter)[:-1])
                except StopIteration:
                    break
            if not chunks:
                break
            input_ids = torch.stack(chunks).to(device)
            out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            total_loss += out.loss.item()
            count += 1
        model.train()
        avg = torch.tensor(total_loss / max(count, 1), device=device)
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        return avg.item()

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()
    val_iter = iter(val_stream) if val_stream is not None else None

    global_step, accum_step, accum_loss = 0, 0, 0.0
    t0 = time.time()
    tokens_seen = 0
    total_tokens = 0
    batch_buf = []

    for chunk in stream:
        if global_step >= args.steps:
            break
        batch_buf.append(chunk[:-1])
        if len(batch_buf) < args.batch_size:
            continue
        input_ids = torch.stack(batch_buf).to(device)
        batch_buf = []

        # DDP: skip the gradient all-reduce on all but the last micro-batch of
        # each accumulation group (else DDP all-reduces grad_accum times/step —
        # catastrophic on a slow-collective node). FSDP handles this internally.
        is_last_micro = (accum_step + 1 == args.grad_accum)
        sync_ctx = (model.no_sync() if (not args.fsdp and not is_last_micro
                                        and hasattr(model, "no_sync"))
                    else contextlib.nullcontext())
        with sync_ctx:
            out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = out.loss
            if args.z_loss > 0:
                z = torch.logsumexp(out.logits.float(), dim=-1)   # [b, seq]
                loss = loss + args.z_loss * (z * z).mean()
            loss = loss / args.grad_accum
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
        optimizer.zero_grad()
        if ema is not None:
            with torch.no_grad():
                for e, p in zip(ema, model.parameters()):
                    e.mul_(args.ema).add_(p.detach(), alpha=1.0 - args.ema)
        global_step += 1
        accum_step = 0

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            tok_t = torch.tensor(float(tokens_seen), device=device)
            dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
            total_tokens += int(tok_t.item())
            if is_main:
                elapsed = time.time() - t0
                tok_sec = tok_t.item() / elapsed
                last_lrs = scheduler.get_last_lr()
                lr_now = last_lrs[0]
                lr_res = last_lrs[2] if len(last_lrs) > 2 else None
                lr_str = f"lr {lr_now:.3e}" + (f" | lr_res {lr_res:.3e}" if lr_res is not None else "")
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {global_step:6d} | loss {loss_t.item():.4f} | "
                      f"{lr_str} | grad_norm {float(grad_norm):.3f} | "
                      f"{tok_sec/1e3:.1f}k tok/s | {total_tokens/1e9:.2f}B tok | {mem_gb:.1f}GB",
                      flush=True)
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": loss_t.item(), "train/lr": lr_now,
                               **({"train/lr_attnres": lr_res} if lr_res is not None else {}),
                               "train/grad_norm": float(grad_norm),
                               "train/tok_per_s": tok_sec,
                               "train/total_tokens_B": total_tokens / 1e9}, step=global_step)
            tokens_seen = 0
            t0 = time.time()
        accum_loss = 0.0

        if global_step % args.save_every == 0:
            ckpt = os.path.join(args.out_dir, f"step-{global_step}")
            save_ckpt(model, ema, tokenizer, ckpt, is_main, args.fsdp)

        if args.eval_every > 0 and global_step % args.eval_every == 0 and val_iter is not None:
            val_loss = evaluate(val_iter, args.eval_steps)
            if is_main:
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                print(f"step {global_step:6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}",
                      flush=True)
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    if args.eval_every > 0 and val_iter is not None:
        val_loss = evaluate(val_iter, args.eval_steps)
        if is_main:
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"FINAL | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    final_dir = os.path.join(args.out_dir, "final")
    save_ckpt(model, ema, tokenizer, final_dir, is_main, args.fsdp)
    if is_main:
        print(f"Training done. Final model -> {final_dir}", flush=True)
        if use_wandb:
            import wandb
            wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
