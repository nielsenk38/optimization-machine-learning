"""Additional analysis and plotting utilities for Rapport_V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import (
    PRETTY_NAMES,
    SUMMARY_METADATA_COLUMNS,
    load_metrics,
    load_metrics_from_results_dirs,
    load_prediction_distributions,
    save_final_bar_plot,
    save_line_plot,
    summarize_by_epoch,
    summarize_final_test_accuracy,
)
from .utils import ensure_dir, read_json


MAIN_ORDER_MODES = [
    "random",
    "fixed_random",
    "curriculum_easy",
    "curriculum_hard",
    "label_sorted",
    "label_block_random",
]
SWEEP_ORDER_MODES = [
    "random",
    "fixed_random",
    "curriculum_hard",
    "label_sorted",
    "label_block_random",
]


def _load_experiment_metadata(results_dir: str | Path) -> dict[str, object]:
    results_dir = Path(results_dir)
    metadata: dict[str, object] = {
        "output_dir": str(results_dir),
        "experiment_name": results_dir.name,
    }
    args_path = results_dir / "experiment_args.json"
    if args_path.exists():
        args = read_json(args_path)
        metadata.update(
            {
                "learning_rate": args.get("lr"),
                "epochs": args.get("epochs"),
                "max_train_examples": args.get("max_train_examples"),
                "optimizer": args.get("optimizer"),
                "model": args.get("model"),
                "batch_size": args.get("batch_size"),
                "momentum": args.get("momentum"),
                "weight_decay": args.get("weight_decay"),
            }
        )
    return metadata


def _attach_metadata(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for column, value in metadata.items():
        if column not in df.columns:
            df[column] = value
        else:
            df[column] = df[column].where(df[column].notna(), value)
    return df


def _load_csv_pattern(results_dir: str | Path, pattern: str) -> pd.DataFrame:
    results_dir = Path(results_dir)
    raw_dir = results_dir / "raw"
    paths = sorted(raw_dir.glob(pattern))
    if not paths:
        return pd.DataFrame()
    metadata = _load_experiment_metadata(results_dir)
    frames = [_attach_metadata(pd.read_csv(path), metadata) for path in paths]
    return pd.concat(frames, ignore_index=True)


def load_gradient_cosines(results_dir: str | Path) -> pd.DataFrame:
    return _load_csv_pattern(results_dir, "gradient_cosine_*.csv")


def load_class_accuracy(results_dir: str | Path) -> pd.DataFrame:
    return _load_csv_pattern(results_dir, "class_accuracy_*.csv")


def summarize_gradient_cosine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df.dropna(subset=["cosine_similarity"]).copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_step = (
        valid.groupby(["order_mode", "global_step"], dropna=False)["cosine_similarity"]
        .agg(cosine_similarity_mean="mean", cosine_similarity_std="std")
        .reset_index()
    )
    by_order = (
        valid.groupby(["order_mode"], dropna=False)["cosine_similarity"]
        .agg(cosine_similarity_mean="mean", cosine_similarity_std="std", count="count")
        .reset_index()
    )
    total_counts = df.groupby(["order_mode"], dropna=False)["global_step"].count().reset_index(name="total_steps")
    by_order = by_order.merge(total_counts, on="order_mode", how="left")
    by_order["valid_fraction"] = by_order["count"] / by_order["total_steps"]
    last_valid_step = valid.groupby(["order_mode"], dropna=False)["global_step"].max().reset_index(name="last_valid_step")
    by_order = by_order.merge(last_valid_step, on="order_mode", how="left")
    return by_step, by_order


def add_forgetting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["order_mode", "seed", "class_id", "epoch"]).copy()
    df["best_so_far"] = df.groupby(["order_mode", "seed", "class_id"], dropna=False)["class_accuracy"].cummax()
    df["forgetting"] = df["best_so_far"] - df["class_accuracy"]
    return df


def summarize_class_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["order_mode", "epoch", "class_id"], dropna=False)["class_accuracy"]
        .agg(class_accuracy_mean="mean", class_accuracy_std="std")
        .reset_index()
    )
    return summary


def summarize_forgetting(df: pd.DataFrame) -> pd.DataFrame:
    forgetting = add_forgetting(df)
    summary = (
        forgetting.groupby(["order_mode", "epoch"], dropna=False)["forgetting"]
        .agg(forgetting_mean="mean", forgetting_std="std")
        .reset_index()
    )
    return summary


def _collapse_summary(prediction_distribution: pd.DataFrame) -> pd.DataFrame:
    if prediction_distribution.empty:
        return pd.DataFrame(
            columns=["order_mode", "learning_rate", "seed", "collapse_fraction", "is_collapsed"]
        )

    grouped = (
        prediction_distribution.groupby(["order_mode", "learning_rate", "seed"], dropna=False)["num_predictions"]
        .agg(total_predictions="sum", max_predictions="max")
        .reset_index()
    )
    grouped["collapse_fraction"] = grouped["max_predictions"] / grouped["total_predictions"]
    grouped["is_collapsed"] = grouped["collapse_fraction"] >= 0.40
    return grouped


def build_lr_sweep_summary(results_dirs: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = load_metrics_from_results_dirs(results_dirs)
    metrics = metrics[metrics["order_mode"].isin(SWEEP_ORDER_MODES)].copy()
    final_epoch = metrics.groupby(SUMMARY_METADATA_COLUMNS + ["seed"], dropna=False)["epoch"].transform("max")
    final_rows = metrics[metrics["epoch"] == final_epoch].copy()
    final_summary = (
        final_rows.groupby(["order_mode", "learning_rate"], dropna=False)["test_accuracy"]
        .agg(test_accuracy_mean="mean", test_accuracy_std="std", test_accuracy_count="count")
        .reset_index()
    )
    train_loss_summary = (
        final_rows.groupby(["order_mode", "learning_rate"], dropna=False)["train_loss"]
        .agg(final_train_loss_mean="mean", final_train_loss_std="std")
        .reset_index()
    )

    prediction_distribution_frames = []
    for results_dir in results_dirs:
        distribution_df = load_prediction_distributions(results_dir)
        if not distribution_df.empty:
            prediction_distribution_frames.append(distribution_df)
    prediction_distribution = (
        pd.concat(prediction_distribution_frames, ignore_index=True)
        if prediction_distribution_frames
        else pd.DataFrame()
    )
    collapse = _collapse_summary(prediction_distribution)
    collapse_summary = (
        collapse.groupby(["order_mode", "learning_rate"], dropna=False)
        .agg(
            collapse_fraction_mean=("collapse_fraction", "mean"),
            collapse_rate=("is_collapsed", "mean"),
        )
        .reset_index()
    )
    collapse_summary["collapse_label"] = np.where(collapse_summary["collapse_rate"] > 0.0, "yes", "no")

    summary = final_summary.merge(
        train_loss_summary,
        on=["order_mode", "learning_rate"],
        how="left",
    ).merge(
        collapse_summary,
        on=["order_mode", "learning_rate"],
        how="left",
    )
    return metrics, summary


def _save_main_comparison_plots(main_results_dir: Path, figures_dir: Path, tables_dir: Path) -> None:
    metrics = load_metrics(main_results_dir)
    summary = summarize_by_epoch(metrics)
    final_summary = summarize_final_test_accuracy(metrics)
    summary.to_csv(tables_dir / "main_summary_by_epoch.csv", index=False)
    final_summary.to_csv(tables_dir / "main_final_test_accuracy.csv", index=False)

    save_line_plot(summary, "test_accuracy", "Test accuracy", figures_dir / "main_test_accuracy.pdf")
    save_line_plot(summary, "train_loss", "Training loss", figures_dir / "main_train_loss.pdf")
    save_line_plot(summary, "grad_norm_mean", "Mean gradient norm", figures_dir / "main_grad_norm.pdf")
    save_final_bar_plot(metrics, figures_dir / "main_final_test_accuracy.pdf")


def save_gradient_cosine_plot(by_step: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for order_mode, group in by_step.groupby("order_mode", sort=False):
        group = group.sort_values("global_step")
        ax.plot(
            group["global_step"],
            group["cosine_similarity_mean"],
            linewidth=1.5,
            label=PRETTY_NAMES.get(order_mode, order_mode),
        )
        std = group["cosine_similarity_std"].fillna(0.0)
        ax.fill_between(
            group["global_step"],
            group["cosine_similarity_mean"] - std,
            group["cosine_similarity_mean"] + std,
            alpha=0.12,
        )

    ax.set_xlabel("Global optimization step")
    ax.set_ylabel("Gradient cosine similarity")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_accuracy_heatmaps(class_summary: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    image = None
    for ax, order_mode in zip(axes_flat, MAIN_ORDER_MODES):
        group = class_summary[class_summary["order_mode"] == order_mode]
        if group.empty:
            ax.axis("off")
            continue
        pivot = group.pivot(index="class_id", columns="epoch", values="class_accuracy_mean").sort_index()
        image = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Class")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(int))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(int))

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.92, wspace=0.28, hspace=0.32)
    if image is not None:
        colorbar_axis = fig.add_axes([0.90, 0.16, 0.02, 0.68])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Class accuracy")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_accuracy_curves(class_summary: pd.DataFrame, output_path: str | Path) -> None:
    selected_orders = ["random", "curriculum_hard", "label_sorted", "label_block_random"]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, order_mode in zip(axes_flat, selected_orders):
        group = class_summary[class_summary["order_mode"] == order_mode]
        for class_id, class_group in group.groupby("class_id"):
            class_group = class_group.sort_values("epoch")
            ax.plot(class_group["epoch"], class_group["class_accuracy_mean"], linewidth=1.0, label=f"{class_id}")
        ax.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Class accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Class", loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_forgetting_plot(forgetting_summary: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for order_mode, group in forgetting_summary.groupby("order_mode", sort=False):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group["forgetting_mean"], marker="o", linewidth=1.5, label=PRETTY_NAMES.get(order_mode, order_mode))
        std = group["forgetting_std"].fillna(0.0)
        ax.fill_between(group["epoch"], group["forgetting_mean"] - std, group["forgetting_mean"] + std, alpha=0.12)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean forgetting")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_lr_sweep_final_accuracy_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    learning_rates = sorted(summary["learning_rate"].dropna().unique().tolist(), reverse=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(SWEEP_ORDER_MODES))
    width = 0.18

    for index, learning_rate in enumerate(learning_rates):
        group = summary[summary["learning_rate"] == learning_rate].set_index("order_mode").reindex(SWEEP_ORDER_MODES)
        offset = (index - (len(learning_rates) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            group["test_accuracy_mean"],
            width=width,
            yerr=group["test_accuracy_std"].fillna(0.0),
            capsize=2,
            label=f"lr={learning_rate:g}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAMES.get(order_mode, order_mode) for order_mode in SWEEP_ORDER_MODES], rotation=12, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_lr_curve_grid(metrics: pd.DataFrame, metric: str, ylabel: str, output_path: str | Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.0), sharex=True)
    axes_flat = axes.flatten()
    learning_rates = sorted(metrics["learning_rate"].dropna().unique().tolist(), reverse=True)

    for ax, order_mode in zip(axes_flat, SWEEP_ORDER_MODES):
        group = metrics[metrics["order_mode"] == order_mode]
        for learning_rate in learning_rates:
            lr_group = group[group["learning_rate"] == learning_rate]
            if lr_group.empty:
                continue
            summary = (
                lr_group.groupby("epoch", dropna=False)[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("epoch")
            )
            ax.plot(summary["epoch"], summary["mean"], marker="o", linewidth=1.3, label=f"lr={learning_rate:g}")
            std = summary["std"].fillna(0.0)
            ax.fill_between(summary["epoch"], summary["mean"] - std, summary["mean"] + std, alpha=0.12)
        ax.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _stratified_sample_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(labels) <= max_points:
        return np.arange(len(labels))

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max_points // len(classes)
    remainder = max_points % len(classes)
    selected: list[int] = []
    for index, class_id in enumerate(classes):
        class_indices = np.where(labels == class_id)[0]
        take = min(len(class_indices), per_class + (1 if index < remainder else 0))
        selected.extend(rng.choice(class_indices, size=take, replace=False).tolist())
    return np.array(selected, dtype=np.int64)


def _compute_pca_projection(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[-2:]]
    return centered @ components


def save_embedding_pca_grid(
    main_results_dir: str | Path,
    output_path: str | Path,
    seed: int = 0,
    max_points: int = 2000,
) -> None:
    main_results_dir = Path(main_results_dir)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.1))
    axes_flat = axes.flatten()
    scatter = None

    for ax, order_mode in zip(axes_flat, MAIN_ORDER_MODES):
        embedding_path = main_results_dir / "raw" / f"embeddings_{order_mode}_seed{seed}.npz"
        if not embedding_path.exists():
            ax.axis("off")
            continue

        data = np.load(embedding_path)
        embeddings = data["embeddings"]
        labels = data["labels"]
        finite_mask = np.isfinite(embeddings).all(axis=1)
        if not finite_mask.any():
            ax.text(0.5, 0.5, "Non-finite\nembeddings", ha="center", va="center", fontsize=10)
            ax.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        embeddings = embeddings[finite_mask]
        labels = labels[finite_mask]
        sample_indices = _stratified_sample_indices(labels, max_points=max_points, seed=seed)
        projection = _compute_pca_projection(embeddings[sample_indices])
        sampled_labels = labels[sample_indices]

        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=sampled_labels,
            cmap="tab10",
            s=5,
            alpha=0.7,
            linewidths=0,
        )
        ax.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.92, wspace=0.28, hspace=0.32)
    if scatter is not None:
        colorbar_axis = fig.add_axes([0.90, 0.16, 0.02, 0.68])
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("Class")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_gradient_variance_plot(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot per-epoch gradient variance Tr(Cov(g)) for each ordering.

    Measures the sigma^2 term in SGD convergence bounds: E[||g||^2] - ||E[g]||^2.
    Low value = batch gradients are tightly clustered (may indicate bias, not diversity).
    High value = diverse, noisy updates as assumed by SGD theory.
    """
    if "grad_variance" not in metrics.columns:
        return
    valid = metrics.dropna(subset=["grad_variance"]).copy()
    if valid.empty:
        return

    summary = (
        valid.groupby(["order_mode", "epoch"], dropna=False)["grad_variance"]
        .agg(grad_variance_mean="mean", grad_variance_std="std")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for order_mode, group in summary.groupby("order_mode", sort=False):
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["grad_variance_mean"],
            marker="o",
            linewidth=1.5,
            label=PRETTY_NAMES.get(order_mode, order_mode),
        )
        std = group["grad_variance_std"].fillna(0.0)
        ax.fill_between(
            group["epoch"],
            group["grad_variance_mean"] - std,
            group["grad_variance_mean"] + std,
            alpha=0.12,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient variance Tr(Cov(g))")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_optimizer_comparison_plot(results_dirs: list[str | Path], output_path: str | Path) -> None:
    """Bar chart comparing SGD vs AdamW final accuracy across orderings."""
    frames = []
    for results_dir in results_dirs:
        m = load_metrics(results_dir)
        if m.empty:
            continue
        meta = _load_experiment_metadata(results_dir)
        m = _attach_metadata(m, meta)
        frames.append(m)
    if not frames:
        return

    all_metrics = pd.concat(frames, ignore_index=True)
    final_epoch = all_metrics.groupby(["order_mode", "optimizer"], dropna=False)["epoch"].transform("max")
    final = all_metrics[all_metrics["epoch"] == final_epoch].copy()

    summary = (
        final.groupby(["order_mode", "optimizer"], dropna=False)["test_accuracy"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    orders = [o for o in MAIN_ORDER_MODES if o in summary["order_mode"].values]
    optimizers = sorted(summary["optimizer"].dropna().unique().tolist())
    x = np.arange(len(orders))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, opt in enumerate(optimizers):
        group = summary[summary["optimizer"] == opt].set_index("order_mode").reindex(orders)
        offset = (i - (len(optimizers) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            group["mean"].fillna(0.0),
            width=width,
            yerr=group["std"].fillna(0.0),
            capsize=3,
            label=opt.upper(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAMES.get(o, o) for o in orders], rotation=12, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.1, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance (10%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model_comparison_plot(
    cnn_results_dir: str | Path,
    logistic_results_dir: str | Path,
    output_path: str | Path,
) -> None:
    """Bar chart comparing CNN vs logistic regression for all orderings."""
    frames = []
    for results_dir, label in [(cnn_results_dir, "CNN (non-convex)"), (logistic_results_dir, "Logistic (convex)")]:
        m = load_metrics(results_dir)
        if m.empty:
            continue
        meta = _load_experiment_metadata(results_dir)
        m = _attach_metadata(m, meta)
        m["model_label"] = label
        frames.append(m)
    if not frames:
        return

    all_metrics = pd.concat(frames, ignore_index=True)
    final_epoch = all_metrics.groupby(["order_mode", "model_label"], dropna=False)["epoch"].transform("max")
    final = all_metrics[all_metrics["epoch"] == final_epoch].copy()

    summary = (
        final.groupby(["order_mode", "model_label"], dropna=False)["test_accuracy"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    orders = MAIN_ORDER_MODES
    model_labels = ["CNN (non-convex)", "Logistic (convex)"]
    x = np.arange(len(orders))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, model_label in enumerate(model_labels):
        group = summary[summary["model_label"] == model_label].set_index("order_mode").reindex(orders)
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            group["mean"].fillna(0.0),
            width=width,
            yerr=group["std"].fillna(0.0),
            capsize=3,
            label=model_label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAMES.get(o, o) for o in orders], rotation=12, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.1, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance (10%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_v3_outputs(
    main_results_dir: str | Path,
    sweep_results_dirs: list[str | Path],
    output_dir: str | Path,
    adamw_results_dirs: list[str | Path] | None = None,
    logistic_results_dir: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    _save_main_comparison_plots(Path(main_results_dir), figures_dir, tables_dir)

    # Gradient variance (new metric — requires re-running experiments with updated code)
    main_metrics = load_metrics(main_results_dir)
    if not main_metrics.empty and "grad_variance" in main_metrics.columns:
        main_metrics.to_csv(tables_dir / "main_metrics_full.csv", index=False)
        save_gradient_variance_plot(main_metrics, figures_dir / "gradient_variance.pdf")

    gradient_cosines = load_gradient_cosines(main_results_dir)
    class_accuracy = load_class_accuracy(main_results_dir)

    if not gradient_cosines.empty:
        gradient_by_step, gradient_by_order = summarize_gradient_cosine(gradient_cosines)
        gradient_by_step.to_csv(tables_dir / "gradient_cosine_by_step.csv", index=False)
        gradient_by_order.to_csv(tables_dir / "gradient_cosine_summary.csv", index=False)
        save_gradient_cosine_plot(gradient_by_step, figures_dir / "gradient_cosine_similarity.pdf")

    if not class_accuracy.empty:
        class_accuracy_summary = summarize_class_accuracy(class_accuracy)
        forgetting_summary = summarize_forgetting(class_accuracy)
        class_accuracy_summary.to_csv(tables_dir / "class_accuracy_by_epoch.csv", index=False)
        forgetting_summary.to_csv(tables_dir / "forgetting_by_epoch.csv", index=False)
        save_class_accuracy_heatmaps(class_accuracy_summary, figures_dir / "class_accuracy_heatmaps.pdf")
        save_class_accuracy_curves(class_accuracy_summary, figures_dir / "class_accuracy_curves.pdf")
        save_forgetting_plot(forgetting_summary, figures_dir / "forgetting_by_epoch.pdf")

    metrics, lr_summary = build_lr_sweep_summary(sweep_results_dirs)
    lr_summary.to_csv(tables_dir / "lr_sweep_summary.csv", index=False)
    save_lr_sweep_final_accuracy_plot(lr_summary, figures_dir / "lr_sweep_final_accuracy.pdf")
    _save_lr_curve_grid(metrics, "test_accuracy", "Test accuracy", figures_dir / "lr_sweep_accuracy_curves.pdf")
    _save_lr_curve_grid(metrics, "train_loss", "Training loss", figures_dir / "lr_sweep_train_loss_curves.pdf")

    save_embedding_pca_grid(main_results_dir, figures_dir / "embedding_pca_grid.pdf", seed=0)

    # Optional: AdamW vs SGD comparison
    if adamw_results_dirs:
        all_dirs = [main_results_dir, *adamw_results_dirs]
        save_optimizer_comparison_plot(all_dirs, figures_dir / "optimizer_comparison.pdf")

    # Optional: CNN vs logistic regression comparison
    if logistic_results_dir is not None:
        save_model_comparison_plot(main_results_dir, logistic_results_dir, figures_dir / "model_comparison.pdf")
