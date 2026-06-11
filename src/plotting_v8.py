"""Figure generation utilities for Rapport_V8.

These plots are built only from existing result directories. The script does
not rerun experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import PRETTY_NAMES, load_metrics, load_prediction_distributions, summarize_by_epoch
from .plotting_v3 import (
    load_class_accuracy,
    load_gradient_cosines,
    summarize_class_accuracy,
    summarize_forgetting,
    summarize_gradient_cosine,
)
from .utils import ensure_dir


STABLE_ORDER_MODES = [
    "random",
    "fixed_random",
    "curriculum_easy",
    "curriculum_hard",
]
FAILURE_ORDER_MODES = [
    "label_sorted",
    "label_block_random",
]
GLOBAL_ORDER_MODES = [
    "random",
    "fixed_random",
    "curriculum_hard",
    "curriculum_easy",
    "label_sorted",
    "label_block_random",
]

SEED_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
}
LR_COLORS = {
    0.1: "#d62728",
    0.05: "#1f77b4",
    0.001: "#2ca02c",
}
ORDER_COLORS = {
    "random": "#1f77b4",
    "fixed_random": "#4c78a8",
    "curriculum_easy": "#f2a541",
    "curriculum_hard": "#c17c00",
    "label_sorted": "#d62728",
    "label_block_random": "#9467bd",
}


def _final_epoch_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    final_epoch = metrics.groupby(["order_mode", "seed"], dropna=False)["epoch"].transform("max")
    return metrics[metrics["epoch"] == final_epoch].copy()


def _summarize_final_accuracy(metrics: pd.DataFrame) -> pd.DataFrame:
    final_rows = _final_epoch_rows(metrics)
    return (
        final_rows.groupby("order_mode", dropna=False)["test_accuracy"]
        .agg(test_accuracy_mean="mean", test_accuracy_std="std")
        .reset_index()
    )


def _summarize_failure_seed_instability(
    main_results_dir: str | Path,
    rescue_results_dir: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    main_results_dir = Path(main_results_dir)
    rescue_results_dir = Path(rescue_results_dir)

    for order_mode in FAILURE_ORDER_MODES:
        for seed in [0, 1, 2]:
            cosine_path = main_results_dir / "raw" / f"gradient_cosine_{order_mode}_seed{seed}.csv"
            prediction_path = main_results_dir / "raw" / f"prediction_distribution_{order_mode}_seed{seed}.csv"
            rescue_prediction_path = rescue_results_dir / "raw" / f"prediction_distribution_{order_mode}_seed{seed}.csv"

            cosine_df = pd.read_csv(cosine_path)
            prediction_df = pd.read_csv(prediction_path)
            rescue_prediction_df = pd.read_csv(rescue_prediction_path)

            finite_mask = cosine_df["cosine_similarity"].notna()
            total_pairs = int(len(cosine_df))
            finite_pairs = int(finite_mask.sum())
            base_total_predictions = int(prediction_df["num_predictions"].sum())
            rescue_total_predictions = int(rescue_prediction_df["num_predictions"].sum())
            dominant_base = prediction_df.loc[prediction_df["num_predictions"].idxmax()]
            dominant_rescue = rescue_prediction_df.loc[rescue_prediction_df["num_predictions"].idxmax()]

            rows.append(
                {
                    "order_mode": order_mode,
                    "seed": seed,
                    "finite_pairs": finite_pairs,
                    "total_pairs": total_pairs,
                    "nan_pairs": total_pairs - finite_pairs,
                    "last_finite_step": int(cosine_df.loc[finite_mask, "global_step"].max()) if finite_pairs else -1,
                    "dominant_class_lr005": int(dominant_base["class_id"]),
                    "dominant_fraction_lr005": float(dominant_base["num_predictions"]) / max(1, base_total_predictions),
                    "dominant_class_lr0001": int(dominant_rescue["class_id"]),
                    "dominant_fraction_lr0001": float(dominant_rescue["num_predictions"]) / max(1, rescue_total_predictions),
                    "nonzero_classes_lr0001": int((rescue_prediction_df["num_predictions"] > 0).sum()),
                }
            )

    return pd.DataFrame(rows)


def save_global_accuracy_plot(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    metrics = load_metrics(main_results_dir)
    summary = _summarize_final_accuracy(metrics).set_index("order_mode").reindex(GLOBAL_ORDER_MODES).reset_index()

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    y_positions = np.arange(len(summary))
    labels = [PRETTY_NAMES.get(order_mode, order_mode) for order_mode in summary["order_mode"]]
    colors = [ORDER_COLORS[order_mode] for order_mode in summary["order_mode"]]
    ax.barh(
        y_positions,
        summary["test_accuracy_mean"],
        xerr=summary["test_accuracy_std"].fillna(0.0),
        color=colors,
        capsize=3,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(0.103, len(summary) - 0.3, "chance", fontsize=8, va="center")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Final test accuracy")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return summary


def save_accuracy_trajectory_plot(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    metrics = load_metrics(main_results_dir)
    summary = summarize_by_epoch(metrics)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for order_mode in GLOBAL_ORDER_MODES:
        group = summary[summary["order_mode"] == order_mode].sort_values("epoch")
        if group.empty:
            continue
        mean = group["test_accuracy_mean"]
        std = group["test_accuracy_std"].fillna(0.0)
        ax.plot(
            group["epoch"],
            mean,
            marker="o",
            linewidth=1.5,
            label=PRETTY_NAMES.get(order_mode, order_mode),
            color=ORDER_COLORS[order_mode],
        )
        ax.fill_between(group["epoch"], mean - std, mean + std, color=ORDER_COLORS[order_mode], alpha=0.12)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return summary


def save_failure_overview_figure(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    gradient_cosines = load_gradient_cosines(main_results_dir)
    prediction_distribution = load_prediction_distributions(main_results_dir)
    instability_rows: list[dict[str, int | str]] = []

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.9))

    for column, order_mode in enumerate(FAILURE_ORDER_MODES):
        cosine_axis = axes[0, column]
        prediction_axis = axes[1, column]
        order_cosines = gradient_cosines[gradient_cosines["order_mode"] == order_mode].copy()

        for seed in [0, 1, 2]:
            seed_group = order_cosines[order_cosines["seed"] == seed].sort_values("global_step")
            if seed_group.empty:
                continue
            cosine_axis.plot(
                seed_group["global_step"],
                seed_group["cosine_similarity"],
                linewidth=1.2,
                label=f"Seed {seed}",
                color=SEED_COLORS[seed],
            )
            finite_mask = seed_group["cosine_similarity"].notna()
            instability_rows.append(
                {
                    "order_mode": order_mode,
                    "seed": seed,
                    "finite_pairs": int(finite_mask.sum()),
                    "total_pairs": int(len(seed_group)),
                    "last_finite_step": int(seed_group.loc[finite_mask, "global_step"].max()) if finite_mask.any() else -1,
                }
            )

        cosine_axis.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=10)
        cosine_axis.set_xlabel("Global step")
        cosine_axis.set_ylabel("Cosine similarity")
        cosine_axis.set_ylim(-1.0, 1.0)
        cosine_axis.set_xlim(0, 100 if order_mode == "label_sorted" else 1600)
        cosine_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        cosine_axis.grid(True, alpha=0.25)

        order_prediction = prediction_distribution[prediction_distribution["order_mode"] == order_mode].copy()
        pivot = (
            order_prediction.pivot_table(
                index="class_id",
                columns="seed",
                values="num_predictions",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(index=np.arange(10), fill_value=0.0)
            .reindex(columns=[0, 1, 2], fill_value=0.0)
        )
        fractions = pivot.divide(pivot.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
        x = np.arange(10)
        width = 0.24
        for offset_index, seed in enumerate([0, 1, 2]):
            prediction_axis.bar(
                x + (offset_index - 1) * width,
                fractions[seed],
                width=width,
                color=SEED_COLORS[seed],
                label=f"Seed {seed}",
            )
        prediction_axis.set_xlabel("Predicted class")
        prediction_axis.set_ylabel("Prediction fraction")
        prediction_axis.set_xticks(x)
        prediction_axis.set_ylim(0.0, 1.05)
        prediction_axis.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return pd.DataFrame(instability_rows)


def save_failure_lr_trajectories(
    main_results_dir: str | Path,
    lr010_results_dir: str | Path,
    rescue_results_dir: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    metrics = pd.concat(
        [
            load_metrics(main_results_dir),
            load_metrics(lr010_results_dir),
            load_metrics(rescue_results_dir),
        ],
        ignore_index=True,
    )
    metrics = metrics[metrics["order_mode"].isin(FAILURE_ORDER_MODES)].copy()
    metrics = metrics[metrics["learning_rate"].isin([0.1, 0.05, 0.001])].copy()

    final_rows = _final_epoch_rows(metrics)
    summary = (
        final_rows.groupby(["order_mode", "learning_rate"], dropna=False)
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            final_train_loss_mean=("train_loss", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)
    for axis, order_mode in zip(axes, FAILURE_ORDER_MODES):
        order_metrics = metrics[metrics["order_mode"] == order_mode].copy()
        for learning_rate in [0.1, 0.05, 0.001]:
            lr_group = order_metrics[order_metrics["learning_rate"] == learning_rate].copy()
            if lr_group.empty:
                continue
            by_epoch = (
                lr_group.groupby("epoch", dropna=False)["test_accuracy"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("epoch")
            )
            axis.plot(
                by_epoch["epoch"],
                by_epoch["mean"],
                marker="o",
                linewidth=1.5,
                label=f"lr={learning_rate:g}",
                color=LR_COLORS[learning_rate],
            )
            axis.fill_between(
                by_epoch["epoch"],
                by_epoch["mean"] - by_epoch["std"].fillna(0.0),
                by_epoch["mean"] + by_epoch["std"].fillna(0.0),
                color=LR_COLORS[learning_rate],
                alpha=0.12,
            )
        axis.axhline(0.1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Test accuracy")
        axis.set_ylim(0.0, 0.6)
        axis.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return summary


def save_failure_class_heatmaps(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    class_accuracy = load_class_accuracy(main_results_dir)
    class_accuracy = class_accuracy[class_accuracy["order_mode"].isin(FAILURE_ORDER_MODES)].copy()
    summary = summarize_class_accuracy(class_accuracy)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), sharex=True, sharey=True)
    image = None
    for axis, order_mode in zip(axes, FAILURE_ORDER_MODES):
        group = summary[summary["order_mode"] == order_mode]
        pivot = group.pivot(index="class_id", columns="epoch", values="class_accuracy_mean").sort_index()
        image = axis.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Class")
        axis.set_xticks(np.arange(pivot.shape[1]))
        axis.set_xticklabels(pivot.columns.astype(int))
        axis.set_yticks(np.arange(pivot.shape[0]))
        axis.set_yticklabels(pivot.index.astype(int))

    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.15, top=0.90, wspace=0.28)
    if image is not None:
        colorbar_axis = fig.add_axes([0.90, 0.18, 0.02, 0.65])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Class accuracy")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return summary


def save_stable_cosine_panels(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    gradient_cosines = load_gradient_cosines(main_results_dir)
    by_step, by_order = summarize_gradient_cosine(gradient_cosines)
    stable_by_step = by_step[by_step["order_mode"].isin(STABLE_ORDER_MODES)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.6), sharex=True, sharey=True)
    for axis, order_mode in zip(axes.flatten(), STABLE_ORDER_MODES):
        group = stable_by_step[stable_by_step["order_mode"] == order_mode].sort_values("global_step")
        axis.plot(
            group["global_step"],
            group["cosine_similarity_mean"],
            linewidth=1.4,
            color=ORDER_COLORS[order_mode],
        )
        axis.fill_between(
            group["global_step"],
            group["cosine_similarity_mean"] - group["cosine_similarity_std"].fillna(0.0),
            group["cosine_similarity_mean"] + group["cosine_similarity_std"].fillna(0.0),
            color=ORDER_COLORS[order_mode],
            alpha=0.15,
        )
        axis.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=10)
        axis.set_xlabel("Global step")
        axis.set_ylabel("Cosine similarity")
        axis.set_ylim(-0.2, 1.0)
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return by_order[by_order["order_mode"].isin(STABLE_ORDER_MODES)].copy()


def save_stable_class_accuracy_curves(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    class_accuracy = load_class_accuracy(main_results_dir)
    class_accuracy = class_accuracy[class_accuracy["order_mode"].isin(STABLE_ORDER_MODES)].copy()
    summary = summarize_class_accuracy(class_accuracy)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8), sharex=True, sharey=True)
    for axis, order_mode in zip(axes.flatten(), STABLE_ORDER_MODES):
        group = summary[summary["order_mode"] == order_mode]
        for class_id, class_group in group.groupby("class_id"):
            class_group = class_group.sort_values("epoch")
            axis.plot(class_group["epoch"], class_group["class_accuracy_mean"], linewidth=1.0, label=f"{class_id}")
        axis.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Class accuracy")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Class", loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return summary


def save_stable_forgetting_plot(main_results_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    class_accuracy = load_class_accuracy(main_results_dir)
    class_accuracy = class_accuracy[class_accuracy["order_mode"].isin(STABLE_ORDER_MODES)].copy()
    summary = summarize_forgetting(class_accuracy)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for order_mode in STABLE_ORDER_MODES:
        group = summary[summary["order_mode"] == order_mode].sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["forgetting_mean"],
            marker="o",
            linewidth=1.5,
            label=PRETTY_NAMES.get(order_mode, order_mode),
            color=ORDER_COLORS[order_mode],
        )
        ax.fill_between(
            group["epoch"],
            group["forgetting_mean"] - group["forgetting_std"].fillna(0.0),
            group["forgetting_mean"] + group["forgetting_std"].fillna(0.0),
            color=ORDER_COLORS[order_mode],
            alpha=0.12,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean forgetting")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return summary


def make_v8_outputs(
    main_results_dir: str | Path,
    lr010_results_dir: str | Path,
    rescue_results_dir: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    global_accuracy = save_global_accuracy_plot(main_results_dir, figures_dir / "global_final_accuracy.pdf")
    global_accuracy.to_csv(tables_dir / "global_final_accuracy.csv", index=False)

    accuracy_by_epoch = save_accuracy_trajectory_plot(main_results_dir, figures_dir / "global_accuracy_trajectories.pdf")
    accuracy_by_epoch.to_csv(tables_dir / "global_accuracy_trajectories.csv", index=False)

    failure_instability = save_failure_overview_figure(main_results_dir, figures_dir / "failure_overview.pdf")
    failure_instability.to_csv(tables_dir / "failure_seed_instability.csv", index=False)

    failure_lr_summary = save_failure_lr_trajectories(
        main_results_dir,
        lr010_results_dir,
        rescue_results_dir,
        figures_dir / "failure_lr_trajectories.pdf",
    )
    failure_lr_summary.to_csv(tables_dir / "failure_lr_summary.csv", index=False)

    failure_heatmaps = save_failure_class_heatmaps(main_results_dir, figures_dir / "failure_class_heatmaps.pdf")
    failure_heatmaps.to_csv(tables_dir / "failure_class_accuracy_by_epoch.csv", index=False)

    stable_cosines = save_stable_cosine_panels(main_results_dir, figures_dir / "stable_cosine_panels.pdf")
    stable_cosines.to_csv(tables_dir / "stable_cosine_summary.csv", index=False)

    stable_class_accuracy = save_stable_class_accuracy_curves(main_results_dir, figures_dir / "stable_class_accuracy_curves.pdf")
    stable_class_accuracy.to_csv(tables_dir / "stable_class_accuracy_by_epoch.csv", index=False)

    stable_forgetting = save_stable_forgetting_plot(main_results_dir, figures_dir / "stable_forgetting.pdf")
    stable_forgetting.to_csv(tables_dir / "stable_forgetting.csv", index=False)

    failure_collapse = _summarize_failure_seed_instability(main_results_dir, rescue_results_dir)
    failure_collapse.to_csv(tables_dir / "failure_collapse_summary.csv", index=False)
