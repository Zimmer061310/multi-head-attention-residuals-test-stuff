#!/usr/bin/env python3
"""Fit and validate the corrected Experiment 2 boundary contribution model."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Attention-Residuals"))

from experiment1_partition_compatibility import (  # noqa: E402
    atomic_write_json,
    git_commit,
    load_jsonl,
    sha256_file,
    utc_now,
)
from mhar_partition import (  # noqa: E402
    generate_adjacent_merge_partitions,
    merged_boundaries,
    mixed_partition_id,
)


N_BOUNDARIES = 15
EXPECTED_K4 = 495
DEFAULT_SEED = 20260826
DEFAULT_WANDB_PROJECT = "MHAR Stuff"


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, using zero-based ranks internally."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    left, right = rankdata(actual), rankdata(predicted)
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    denominator = float(np.square(actual - actual.mean()).sum())
    return {
        "r2": 1.0 - float(np.square(residual).sum()) / denominator,
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "mae": float(np.abs(residual).mean()),
        "spearman": spearman(actual, predicted),
    }


def contrast_matrix(size: int = N_BOUNDARIES) -> np.ndarray:
    matrix = np.zeros((size, size - 1), dtype=float)
    matrix[:-1] = np.eye(size - 1)
    matrix[-1] = -1.0
    return matrix


def fit_additive(indicators: np.ndarray, target: np.ndarray) -> dict:
    frequencies = indicators.mean(axis=0)
    centered = indicators - frequencies
    contrast = contrast_matrix(indicators.shape[1])
    design = np.column_stack([np.ones(len(target)), centered @ contrast])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    beta = contrast @ coefficients[1:]
    return {
        "mu": float(coefficients[0]),
        "beta": beta,
        "frequencies": frequencies,
    }


def predict_additive(model: dict, indicators: np.ndarray) -> np.ndarray:
    return model["mu"] + (indicators - model["frequencies"]) @ model["beta"]


def make_folds(indices: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    shuffled = np.random.default_rng(seed).permutation(indices)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled, count)]


def additive_cross_validation(indicators, target, *, folds=5, seed=DEFAULT_SEED):
    all_indices = np.arange(len(target))
    split = make_folds(all_indices, folds, seed)
    predicted = np.empty_like(target)
    baseline = np.empty_like(target)
    fold_ids = np.empty(len(target), dtype=int)
    for fold_id, test in enumerate(split):
        train = np.setdiff1d(all_indices, test)
        model = fit_additive(indicators[train], target[train])
        predicted[test] = predict_additive(model, indicators[test])
        baseline[test] = target[train].mean()
        fold_ids[test] = fold_id
    return {
        "predicted": predicted,
        "baseline_predicted": baseline,
        "fold_ids": fold_ids,
        "metrics": regression_metrics(target, predicted),
        "baseline_metrics": regression_metrics(target, baseline),
        "fold_count": folds,
        "seed": seed,
    }


def compatible_pairs() -> list[tuple[int, int]]:
    return [
        pair for pair in itertools.combinations(range(N_BOUNDARIES), 2)
        if pair[1] != pair[0] + 1
    ]


def ridge_features(indicators: np.ndarray, pairs) -> np.ndarray:
    pair_values = np.column_stack([
        indicators[:, left] * indicators[:, right] for left, right in pairs
    ])
    return np.column_stack([indicators, pair_values])


def ridge_predict(train_x, train_y, test_x, alpha):
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized_train = (train_x - means) / scales
    standardized_test = (test_x - means) / scales
    target_mean = train_y.mean()
    gram = standardized_train.T @ standardized_train
    weights = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0]),
        standardized_train.T @ (train_y - target_mean),
    )
    return target_mean + standardized_test @ weights


def ridge_nested_cross_validation(
    indicators, target, *, folds=5, inner_folds=4, seed=DEFAULT_SEED,
):
    pairs = compatible_pairs()
    features = ridge_features(indicators, pairs)
    alphas = np.logspace(-4, 4, 17)
    all_indices = np.arange(len(target))
    outer_split = make_folds(all_indices, folds, seed)
    predicted = np.empty_like(target)
    fold_ids = np.empty(len(target), dtype=int)
    selected_alphas = []
    for fold_id, test in enumerate(outer_split):
        train = np.setdiff1d(all_indices, test)
        inner_split = make_folds(train, inner_folds, seed + 1000 + fold_id)
        losses = []
        for alpha in alphas:
            squared_errors = []
            for validation in inner_split:
                fit = np.setdiff1d(train, validation)
                inner_prediction = ridge_predict(
                    features[fit], target[fit], features[validation], alpha)
                squared_errors.extend(np.square(target[validation] - inner_prediction))
            losses.append(float(np.mean(squared_errors)))
        alpha = float(alphas[int(np.argmin(losses))])
        selected_alphas.append(alpha)
        predicted[test] = ridge_predict(features[train], target[train], features[test], alpha)
        fold_ids[test] = fold_id
    augmented = np.column_stack([np.ones(len(target)), features])
    return {
        "predicted": predicted,
        "fold_ids": fold_ids,
        "metrics": regression_metrics(target, predicted),
        "fold_count": folds,
        "inner_fold_count": inner_folds,
        "seed": seed,
        "selected_alphas": selected_alphas,
        "compatible_pair_count": len(pairs),
        "nominal_design_columns": augmented.shape[1],
        "design_rank": int(np.linalg.matrix_rank(augmented)),
    }


def load_k4_design(path: Path):
    rows = load_jsonl(path)
    by_id = {row["partition_id"]: row for row in rows}
    if len(by_id) != len(rows) or "native_h16" not in by_id:
        raise ValueError("results must contain unique rows and native_h16")
    mixed = [row for row in rows if row["partition_id"] != "native_h16"]
    expected = {
        mixed_partition_id(partition)
        for partition in generate_adjacent_merge_partitions(16, 4)
    }
    if len(mixed) != EXPECTED_K4 or {row["partition_id"] for row in mixed} != expected:
        raise ValueError("results are not the complete 495-partition k=4 design")
    mixed.sort(key=lambda row: row["partition_id"])
    indicators = np.zeros((len(mixed), N_BOUNDARIES), dtype=float)
    for row_index, row in enumerate(mixed):
        boundaries = tuple(row["merged_boundaries"])
        if len(boundaries) != 4:
            raise ValueError("every k=4 row must contain four boundaries")
        indicators[row_index, list(boundaries)] = 1.0
    native_nll = float(by_id["native_h16"]["nll"])
    target = np.asarray([float(row["nll"]) - native_nll for row in mixed])
    return mixed, indicators, target, native_nll


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frozen_json(path: Path, payload: dict) -> dict:
    comparable = {key: value for key, value in payload.items() if key != "created_at"}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key != "created_at"} != comparable:
            raise RuntimeError(f"frozen manifest differs: {path}")
        return existing
    value = {"created_at": utc_now(), **comparable}
    atomic_write_json(path, value)
    return value


def boundary_effect_rows(mixed, indicators, target, beta):
    order = {index: rank for rank, index in enumerate(np.argsort(beta), 1)}
    measured_order = np.argsort(target)
    result = []
    for boundary in range(N_BOUNDARIES):
        selected = indicators[:, boundary].astype(bool)
        row = {
            "boundary": boundary,
            "pair": f"({boundary},{boundary + 1})",
            "beta": float(beta[boundary]),
            "relative_rank": order[boundary],
            "all_frequency": float(selected.mean()),
            "top10_frequency": float(selected[measured_order[:10]].mean()),
            "top20_frequency": float(selected[measured_order[:20]].mean()),
            "top50_frequency": float(selected[measured_order[:50]].mean()),
            "mean_delta_when_merged": float(target[selected].mean()),
            "mean_delta_when_not_merged": float(target[~selected].mean()),
        }
        row["descriptive_not_merged_minus_merged"] = (
            row["mean_delta_when_not_merged"] - row["mean_delta_when_merged"])
        result.append(row)
    return sorted(result, key=lambda row: row["beta"])


def candidate_rankings(beta: np.ndarray, num_merges: int) -> list[dict]:
    rows = []
    for partition in generate_adjacent_merge_partitions(16, num_merges):
        boundaries = list(merged_boundaries(partition))
        rows.append({
            "partition_id": mixed_partition_id(partition),
            "merged_boundaries": boundaries,
            "predicted_score": float(beta[boundaries].sum()),
        })
    rows.sort(key=lambda row: (row["predicted_score"], row["partition_id"]))
    for rank, row in enumerate(rows, 1):
        row["predicted_rank"] = rank
    return rows


def selection_manifest(
    rankings, *, num_merges, source_hash, score_hash, seed, uniform_size,
):
    roles = defaultdict(list)
    for row in rankings[:10]:
        roles[row["partition_id"]].append("target_top")
    middle_start = (len(rankings) - 5) // 2
    for row in rankings[middle_start:middle_start + 5]:
        roles[row["partition_id"]].append("target_middle")
    for row in rankings[-5:]:
        roles[row["partition_id"]].append("target_bottom")
    generator = random.Random(seed + num_merges * 1009)
    for index in generator.sample(range(len(rankings)), uniform_size):
        roles[rankings[index]["partition_id"]].append("uniform_transfer")
    by_id = {row["partition_id"]: row for row in rankings}
    candidates = []
    for identifier in sorted(roles):
        row = by_id[identifier]
        candidates.append({
            "partition_id": identifier,
            "roles": sorted(roles[identifier]),
            "predicted_rank": row["predicted_rank"],
            "predicted_score": row["predicted_score"],
            "merged_boundaries": row["merged_boundaries"],
        })
    return {
        "format_version": 1,
        "experiment": 2,
        "stage": "boundary-score-transfer",
        "num_merges": num_merges,
        "candidate_space_size": len(rankings),
        "source_discovery_results_sha256": source_hash,
        "boundary_effects_sha256": score_hash,
        "source_commit": git_commit(),
        "selection_seed": seed,
        "uniform_sample_size": uniform_size,
        "target_rule": "predicted top 10, centered middle 5, bottom 5",
        "uniform_rule": "uniform without replacement from full candidate space",
        "candidates": candidates,
    }


def write_fit_figures(output_dir, effects, target, additive_prediction, ridge_prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
    with plt.rc_context(style):
        ordered = list(reversed(effects))
        fig, axis = plt.subplots(figsize=(6.75, 4.6))
        values = [row["beta"] for row in ordered]
        colors = ["#009E73" if value < 0 else "#D55E00" for value in values]
        axis.barh(range(len(ordered)), values, color=colors, height=0.7)
        axis.set_yticks(range(len(ordered)), [row["pair"] for row in ordered])
        axis.axvline(0, color="#2E3440", linewidth=0.8)
        axis.set_xlabel(r"Constrained additive score $\beta_i$ (nats/token)")
        axis.set_ylabel("Removed adjacent H16 boundary")
        axis.set_title("Relative boundary-removal scores at k=4", weight="bold")
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_boundary_effects.png", dpi=300)
        fig.savefig(output_dir / "fig_boundary_effects.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), sharex=True, sharey=True)
        models = [
            ("Constrained additive", additive_prediction, "#0072B2"),
            ("Prediction-only ridge", ridge_prediction, "#E69F00"),
        ]
        limits = [min(target.min(), additive_prediction.min(), ridge_prediction.min()),
                  max(target.max(), additive_prediction.max(), ridge_prediction.max())]
        for axis, (title, prediction, color) in zip(axes, models):
            metric = regression_metrics(target, prediction)
            axis.scatter(target, prediction, s=10, color=color, alpha=0.65, linewidths=0)
            axis.plot(limits, limits, color="#4B5563", linestyle="--", linewidth=0.8)
            axis.set_title(title, weight="bold")
            axis.set_xlabel(r"Measured $\Delta$NLL")
            axis.text(0.04, 0.96,
                      f"$R^2$={metric['r2']:.3f}\n$\\rho$={metric['spearman']:.3f}",
                      transform=axis.transAxes, ha="left", va="top", fontsize=8)
            axis.grid(color="#D1D5DB", linewidth=0.5, alpha=0.5)
        axes[0].set_ylabel(r"Out-of-fold predicted $\Delta$NLL")
        fig.suptitle("Five-fold k=4 surrogate validation", weight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_k4_oof_diagnostics.png", dpi=300)
        fig.savefig(output_dir / "fig_k4_oof_diagnostics.pdf")
        plt.close(fig)


def add_wandb_arguments(parser):
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)


def wandb_run(args, *, job_type, config):
    if args.wandb_mode == "disabled":
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=args.wandb_group, name=args.wandb_run_name,
        job_type=job_type, mode=args.wandb_mode, config=config,
        tags=["experiment-2", "boundary-model", "step-2000", job_type],
    )


def fit_command(args):
    discovery_path = Path(args.discovery_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed, indicators, target, native_nll = load_k4_design(discovery_path)
    source_hash = sha256_file(discovery_path)
    additive_cv = additive_cross_validation(
        indicators, target, folds=args.folds, seed=args.seed)
    ridge_cv = ridge_nested_cross_validation(
        indicators, target, folds=args.folds,
        inner_folds=args.inner_folds, seed=args.seed)
    model = fit_additive(indicators, target)
    effects = boundary_effect_rows(mixed, indicators, target, model["beta"])
    effect_fields = [
        "relative_rank", "boundary", "pair", "beta", "all_frequency",
        "top10_frequency", "top20_frequency", "top50_frequency",
        "mean_delta_when_merged", "mean_delta_when_not_merged",
        "descriptive_not_merged_minus_merged",
    ]
    effects_path = output_dir / "boundary_effects.csv"
    write_csv(effects_path, effects, effect_fields)
    effect_hash = sha256_file(effects_path)

    oof_rows = []
    for index, row in enumerate(mixed):
        oof_rows.append({
            "partition_id": row["partition_id"],
            "actual_delta_nll": target[index],
            "additive_oof_prediction": additive_cv["predicted"][index],
            "ridge_oof_prediction": ridge_cv["predicted"][index],
            "fold": int(additive_cv["fold_ids"][index]),
        })
    write_csv(
        output_dir / "k4_oof_predictions.csv", oof_rows,
        ["partition_id", "actual_delta_nll", "additive_oof_prediction",
         "ridge_oof_prediction", "fold"])

    selection_paths = {}
    ranking_paths = {}
    for num_merges, expected_count in ((3, 286), (5, 462)):
        rankings = candidate_rankings(model["beta"], num_merges)
        if len(rankings) != expected_count:
            raise AssertionError((num_merges, len(rankings), expected_count))
        ranking_path = output_dir / f"k{num_merges}_predicted_rankings.csv"
        write_csv(
            ranking_path, rankings,
            ["predicted_rank", "partition_id", "predicted_score", "merged_boundaries"])
        manifest = selection_manifest(
            rankings, num_merges=num_merges, source_hash=source_hash,
            score_hash=effect_hash, seed=args.seed, uniform_size=args.uniform_size)
        selection_path = output_dir / f"k{num_merges}_selection.json"
        frozen_json(selection_path, manifest)
        selection_paths[num_merges] = selection_path
        ranking_paths[num_merges] = ranking_path

    summary = {
        "created_at": utc_now(),
        "source_commit": git_commit(),
        "source_discovery_results": str(discovery_path),
        "source_discovery_results_sha256": source_hash,
        "native_h16_nll": native_nll,
        "k4_partition_count": len(mixed),
        "mean_k4_delta_nll": float(target.mean()),
        "additive_constraint_sum_beta": float(model["beta"].sum()),
        "additive_cv": {
            "metrics": additive_cv["metrics"],
            "baseline_metrics": additive_cv["baseline_metrics"],
            "fold_count": args.folds,
            "seed": args.seed,
        },
        "ridge_nested_cv": {
            key: value for key, value in ridge_cv.items()
            if key not in ("predicted", "fold_ids")
        },
        "boundary_effects_sha256": effect_hash,
        "leading_boundaries": [row["pair"] for row in effects[:4]],
        "selection_manifests": {
            f"k{k}": {"path": str(path), "sha256": sha256_file(path)}
            for k, path in selection_paths.items()
        },
    }
    atomic_write_json(output_dir / "model_summary.json", summary)
    write_fit_figures(
        output_dir, effects, target,
        additive_cv["predicted"], ridge_cv["predicted"])
    report = [
        "# Experiment 2 boundary contribution model",
        "",
        f"- Additive CV R2: {additive_cv['metrics']['r2']:.6f}",
        f"- Additive CV Spearman: {additive_cv['metrics']['spearman']:.6f}",
        f"- Ridge CV R2: {ridge_cv['metrics']['r2']:.6f}",
        f"- Ridge CV Spearman: {ridge_cv['metrics']['spearman']:.6f}",
        f"- Leading relative boundaries: {', '.join(summary['leading_boundaries'])}",
        "",
        "Negative additive scores mean less damaging than the average k=4 boundary, "
        "not better than native H16. Ridge interactions are prediction-only.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    run = wandb_run(args, job_type="boundary-model-fit", config=summary)
    if run is not None:
        import wandb
        table = wandb.Table(columns=effect_fields)
        for row in effects:
            table.add_data(*[row[field] for field in effect_fields])
        run.log({
            "additive/cv_r2": additive_cv["metrics"]["r2"],
            "additive/cv_spearman": additive_cv["metrics"]["spearman"],
            "ridge/cv_r2": ridge_cv["metrics"]["r2"],
            "ridge/cv_spearman": ridge_cv["metrics"]["spearman"],
            "analysis/boundary_effects": table,
            "figures/boundary_effects": wandb.Image(str(output_dir / "fig_boundary_effects.png")),
            "figures/k4_oof_diagnostics": wandb.Image(
                str(output_dir / "fig_k4_oof_diagnostics.png")),
        })
        artifact = wandb.Artifact(
            f"mhar-exp2-boundary-model-{source_hash[:12]}", type="analysis")
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        run.summary.update({
            "source_discovery_results_sha256": source_hash,
            "boundary_effects_sha256": effect_hash,
            "k3_selection_sha256": sha256_file(selection_paths[3]),
            "k5_selection_sha256": sha256_file(selection_paths[5]),
        })
        run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_transfer_rows(results_path: Path, selection_path: Path, num_merges: int):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["num_merges"] != num_merges:
        raise ValueError("selection k mismatch")
    selected = {row["partition_id"]: row for row in selection["candidates"]}
    results = load_jsonl(results_path)
    by_id = {row["partition_id"]: row for row in results}
    if len(by_id) != len(results) or "native_h16" not in by_id:
        raise ValueError("transfer results require unique rows and native_h16")
    if set(by_id) != {"native_h16", *selected}:
        raise ValueError("transfer results do not exactly match the frozen manifest")
    native_nll = float(by_id["native_h16"]["nll"])
    rows = []
    for identifier, candidate in selected.items():
        result = by_id[identifier]
        rows.append({
            "k": num_merges,
            "partition_id": identifier,
            "roles": candidate["roles"],
            "predicted_rank": candidate["predicted_rank"],
            "predicted_score": candidate["predicted_score"],
            "actual_nll": result["nll"],
            "actual_delta_nll": result["nll"] - native_nll,
            "merged_boundaries": candidate["merged_boundaries"],
        })
    return rows, native_nll, selection


def transfer_summary(rows):
    uniform = [row for row in rows if "uniform_transfer" in row["roles"]]
    predicted = np.asarray([row["predicted_score"] for row in uniform])
    actual = np.asarray([row["actual_delta_nll"] for row in uniform])
    calibration = np.linalg.lstsq(
        np.column_stack([np.ones(len(predicted)), predicted]), actual, rcond=None)[0]
    calibrated = calibration[0] + calibration[1] * predicted
    groups = {}
    for group in ("top", "middle", "bottom"):
        selected = [row["actual_delta_nll"] for row in rows if f"target_{group}" in row["roles"]]
        groups[group] = {
            "count": len(selected),
            "mean_actual_delta_nll": statistics.fmean(selected),
            "median_actual_delta_nll": statistics.median(selected),
        }
    return {
        "uniform_count": len(uniform),
        "uniform_spearman": spearman(actual, predicted),
        "uniform_affine_calibration": {
            "intercept": float(calibration[0]), "slope": float(calibration[1]),
            "descriptive_in_sample_rmse": float(np.sqrt(np.square(actual - calibrated).mean())),
            "descriptive_in_sample_mae": float(np.abs(actual - calibrated).mean()),
        },
        "targeted": groups,
        "top_minus_middle_mean_delta": (
            groups["top"]["mean_actual_delta_nll"] - groups["middle"]["mean_actual_delta_nll"]),
        "top_minus_bottom_mean_delta": (
            groups["top"]["mean_actual_delta_nll"] - groups["bottom"]["mean_actual_delta_nll"]),
        "directional_conditions_met": (
            spearman(actual, predicted) > 0
            and groups["top"]["mean_actual_delta_nll"] < groups["middle"]["mean_actual_delta_nll"]
            and groups["top"]["mean_actual_delta_nll"] < groups["bottom"]["mean_actual_delta_nll"]
        ),
    }


def write_transfer_figure(output_dir, by_k, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.1))
        for axis, num_merges in zip(axes, (3, 5)):
            rows = by_k[num_merges]
            uniform = [row for row in rows if "uniform_transfer" in row["roles"]]
            axis.scatter(
                [row["predicted_score"] for row in uniform],
                [row["actual_delta_nll"] for row in uniform],
                color="#7B8794", s=22, alpha=0.8, label="Uniform sample")
            styles = {
                "top": ("#009E73", "^"),
                "middle": ("#E69F00", "s"),
                "bottom": ("#D55E00", "X"),
            }
            for group, (color, marker) in styles.items():
                selected = [row for row in rows if f"target_{group}" in row["roles"]]
                axis.scatter(
                    [row["predicted_score"] for row in selected],
                    [row["actual_delta_nll"] for row in selected],
                    color=color, marker=marker, s=30, alpha=0.9, label=f"Target {group}")
            axis.set_title(
                f"k={num_merges}: uniform $\\rho$={summaries[num_merges]['uniform_spearman']:.3f}",
                weight="bold")
            axis.set_xlabel(r"Additive score $S(P)$")
            axis.grid(color="#D1D5DB", linewidth=0.5, alpha=0.5)
        axes[0].set_ylabel(r"Actual confirmation $\Delta$NLL")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle("Boundary-score transfer to frozen k=3 and k=5 interventions",
                     weight="bold")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output_dir / "fig_transfer_scatter.png", dpi=300)
        fig.savefig(output_dir / "fig_transfer_scatter.pdf")
        plt.close(fig)


def analyze_transfer_command(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_k, native, selections, summaries = {}, {}, {}, {}
    for num_merges in (3, 5):
        rows, native_nll, selection = load_transfer_rows(
            Path(getattr(args, f"k{num_merges}_results")).resolve(),
            Path(getattr(args, f"k{num_merges}_selection")).resolve(), num_merges)
        by_k[num_merges], native[num_merges], selections[num_merges] = rows, native_nll, selection
        summaries[num_merges] = transfer_summary(rows)
    combined = by_k[3] + by_k[5]
    write_csv(
        output_dir / "transfer_candidates.csv", combined,
        ["k", "partition_id", "roles", "predicted_rank", "predicted_score",
         "actual_nll", "actual_delta_nll", "merged_boundaries"])
    summary = {
        "created_at": utc_now(), "source_commit": git_commit(),
        "k3": summaries[3], "k5": summaries[5],
        "native_confirmation_nll": {"k3": native[3], "k5": native[5]},
        "results_sha256": {
            "k3": sha256_file(Path(args.k3_results).resolve()),
            "k5": sha256_file(Path(args.k5_results).resolve()),
        },
        "selection_sha256": {
            "k3": sha256_file(Path(args.k3_selection).resolve()),
            "k5": sha256_file(Path(args.k5_selection).resolve()),
        },
    }
    atomic_write_json(output_dir / "transfer_summary.json", summary)
    write_transfer_figure(output_dir, by_k, summaries)
    lines = ["# Experiment 2 boundary-score transfer", ""]
    for num_merges in (3, 5):
        value = summaries[num_merges]
        lines.extend([
            f"## k={num_merges}", "",
            f"- Uniform Spearman: {value['uniform_spearman']:.6f}",
            f"- Target top mean delta: {value['targeted']['top']['mean_actual_delta_nll']:.8f}",
            f"- Target middle mean delta: {value['targeted']['middle']['mean_actual_delta_nll']:.8f}",
            f"- Target bottom mean delta: {value['targeted']['bottom']['mean_actual_delta_nll']:.8f}",
            f"- Directional conditions met: {value['directional_conditions_met']}", "",
        ])
    (output_dir / "transfer_report.md").write_text("\n".join(lines), encoding="utf-8")

    run = wandb_run(args, job_type="boundary-transfer-analysis", config=summary)
    if run is not None:
        import wandb
        table = wandb.Table(columns=[
            "k", "partition_id", "roles", "predicted_rank", "predicted_score",
            "actual_delta_nll", "merged_boundaries"])
        for row in combined:
            table.add_data(*[row[column] for column in table.columns])
        run.log({
            "k3/uniform_spearman": summaries[3]["uniform_spearman"],
            "k5/uniform_spearman": summaries[5]["uniform_spearman"],
            "analysis/transfer_candidates": table,
            "figures/transfer_scatter": wandb.Image(
                str(output_dir / "fig_transfer_scatter.png")),
        })
        artifact = wandb.Artifact("mhar-exp2-boundary-transfer", type="analysis")
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--discovery-results", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--inner-folds", type=int, default=4)
    fit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fit.add_argument("--uniform-size", type=int, default=30)
    add_wandb_arguments(fit)
    fit.set_defaults(func=fit_command)

    transfer = subparsers.add_parser("analyze-transfer")
    transfer.add_argument("--k3-results", required=True)
    transfer.add_argument("--k3-selection", required=True)
    transfer.add_argument("--k5-results", required=True)
    transfer.add_argument("--k5-selection", required=True)
    transfer.add_argument("--output-dir", required=True)
    add_wandb_arguments(transfer)
    transfer.set_defaults(func=analyze_transfer_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
