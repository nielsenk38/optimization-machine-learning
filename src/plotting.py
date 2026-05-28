"""Plotting utilities for the data-order experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, read_json


PRETTY_NAMES = {
    "random": "Random reshuffle",
    "fixed_random": "Fixed random",
    "label_sorted": "Label sorted",
    "label_block_random": "Random label blocks",
    "curriculum_easy": "Easy-to-hard",
    "curriculum_hard": "Hard-to-easy",
}

SUMMARY_METADATA_COLUMNS = [
    "output_dir",
    "experiment_name",
    "order_mode",
    "learning_rate",
    "epochs",
    "max_train_examples",
    "optimizer",
    "model",
    "batch_size",
    "momentum",
    "weight_decay",
]
PLOT_CONFIGURATION_COLUMNS = [
    "output_dir",
    "experiment_name",
    "learning_rate",
    "epochs",
    "max_train_examples",
    "optimizer",
    "model",
    "batch_size",
    "momentum",
    "weight_decay",
]
LR_SENSITIVITY_COMPATIBILITY_COLUMNS = [
    "epochs",
    "max_train_examples",
    "optimizer",
    "model",
    "batch_size",
    "momentum",
    "weight_decay",
]
LABEL_ORDER_MODES = ["label_sorted", "label_block_random"]
MAIN_REPORT_FILTERS = {
    "learning_rate": 0.05,
    "epochs": 10,
    "max_train_examples": 20000,
    "optimizer": "sgd",
}


def _load_experiment_metadata(results_dir: str | Path) -> dict[str, object]:
    """Read experiment-level metadata used to backfill summary tables."""
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
    """Add missing experiment metadata columns and fill empty values."""
    df = df.copy()
    for column, value in metadata.items():
        if column not in df.columns:
            df[column] = value
        else:
            df[column] = df[column].where(df[column].notna(), value)
    return df


def _seed_string(values: pd.Series) -> str:
    unique_values = sorted({str(value) for value in values.tolist()})
    return ",".join(unique_values)


def _metrics_raw_dir(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    raw_dir = results_dir / "raw"
    return raw_dir if raw_dir.exists() else results_dir


def load_metrics(results_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all metrics CSV files from one results directory."""
    results_dir = Path(results_dir)
    raw_dir = _metrics_raw_dir(results_dir)
    paths = sorted(raw_dir.glob("metrics_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics_*.csv files found in {raw_dir}")

    metadata = _load_experiment_metadata(results_dir)
    frames = [_attach_metadata(pd.read_csv(path), metadata) for path in paths]
    return pd.concat(frames, ignore_index=True)


def load_metrics_from_results_dirs(results_dirs: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate metrics across multiple results directories."""
    frames = [load_metrics(results_dir) for results_dir in results_dirs]
    if not frames:
        raise FileNotFoundError("No results directories were provided.")
    return pd.concat(frames, ignore_index=True)


def load_prediction_distributions(results_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all per-run prediction-distribution CSV files."""
    results_dir = Path(results_dir)
    raw_dir = _metrics_raw_dir(results_dir)
    paths = sorted(raw_dir.glob("prediction_distribution_*.csv"))
    if not paths:
        return pd.DataFrame(
            columns=[
                "order_mode",
                "seed",
                "experiment_name",
                "output_dir",
                "optimizer",
                "epochs",
                "max_train_examples",
                "learning_rate",
                "model",
                "batch_size",
                "momentum",
                "weight_decay",
                "class_id",
                "num_predictions",
            ]
        )

    metadata = _load_experiment_metadata(results_dir)
    frames = [_attach_metadata(pd.read_csv(path), metadata) for path in paths]
    return pd.concat(frames, ignore_index=True)


def summarize_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation across seeds, without mixing configs."""
    metrics = [
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "grad_norm_mean",
        "grad_norm_std",
        "grad_cosine_mean",
        "grad_cosine_std",
        "epoch_time_sec",
    ]
    metrics = [metric for metric in metrics if metric in df.columns]
    group_columns = SUMMARY_METADATA_COLUMNS + ["epoch"]
    summary = df.groupby(group_columns, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    seed_info = (
        df.groupby(group_columns, dropna=False)["seed"]
        .agg(seed=_seed_string, seed_count="nunique")
        .reset_index()
    )
    return seed_info.merge(summary, on=group_columns, how="left")


def summarize_final_test_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final test accuracy while preserving experiment metadata."""
    final_epoch = df.groupby(SUMMARY_METADATA_COLUMNS + ["seed"], dropna=False)["epoch"].transform("max")
    final = df[df["epoch"] == final_epoch]

    group_columns = SUMMARY_METADATA_COLUMNS
    summary = (
        final.groupby(group_columns, dropna=False)["test_accuracy"]
        .agg(test_accuracy_mean="mean", test_accuracy_std="std", test_accuracy_count="count")
        .reset_index()
    )
    seed_info = (
        final.groupby(group_columns, dropna=False)["seed"]
        .agg(seed=_seed_string, seed_count="nunique")
        .reset_index()
    )
    return seed_info.merge(summary, on=group_columns, how="left")


def _assert_single_configuration(df: pd.DataFrame, columns: list[str], plot_name: str) -> None:
    configurations = df[columns].drop_duplicates()
    if len(configurations) > 1:
        raise ValueError(
            f"{plot_name} found multiple incompatible experiment configurations in one results directory. "
            f"Regenerate summaries from a clean output directory to avoid mixing runs."
        )


def _assert_compatible_across_results(df: pd.DataFrame, columns: list[str], plot_name: str) -> None:
    configurations = df[columns].drop_duplicates()
    if len(configurations) > 1:
        raise ValueError(
            f"{plot_name} requires matching hyperparameters across source directories, "
            f"except for the intended comparison variables."
        )


def _drop_duplicate_run_rows(df: pd.DataFrame) -> pd.DataFrame:
    subset = [
        "order_mode",
        "seed",
        "learning_rate",
        "epochs",
        "max_train_examples",
        "optimizer",
        "model",
        "batch_size",
        "momentum",
        "weight_decay",
        "epoch",
    ]
    return df.drop_duplicates(subset=subset, keep="first").copy()


def _filter_rows(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    filtered = df.copy()
    for column, value in filters.items():
        filtered = filtered[filtered[column] == value]
    return filtered


def _load_main_report_plot_df(results_dir: Path) -> pd.DataFrame:
    """Build the intended main comparison from compatible source directories only."""
    source_dirs = [results_dir]
    lr005_dir = results_dir.parent / "results_lr005"
    if lr005_dir.exists():
        source_dirs.append(lr005_dir)

    combined = load_metrics_from_results_dirs(source_dirs)
    filtered = _filter_rows(combined, MAIN_REPORT_FILTERS)
    if filtered.empty:
        raise ValueError("No compatible 10-epoch lr=0.05 runs were found for the main report plots.")

    filtered = _drop_duplicate_run_rows(filtered)
    filtered["output_dir"] = str(results_dir)
    filtered["experiment_name"] = results_dir.name
    return filtered


def save_line_plot(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Save a line plot with mean +/- one standard deviation across seeds."""
    _assert_single_configuration(summary, PLOT_CONFIGURATION_COLUMNS, str(output_path))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for order_mode, group in summary.groupby("order_mode", sort=False):
        group = group.sort_values("epoch")
        label = PRETTY_NAMES.get(order_mode, order_mode)
        mean = group[f"{metric}_mean"]
        std = group[f"{metric}_std"].fillna(0.0)
        epochs = group["epoch"]
        ax.plot(epochs, mean, marker="o", linewidth=1.8, label=label)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_final_bar_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a final-test-accuracy bar plot."""
    _assert_single_configuration(df, PLOT_CONFIGURATION_COLUMNS, str(output_path))

    final_epoch = df.groupby(SUMMARY_METADATA_COLUMNS + ["seed"], dropna=False)["epoch"].transform("max")
    final = df[df["epoch"] == final_epoch]
    grouped = (
        final.groupby("order_mode", dropna=False)["test_accuracy"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [PRETTY_NAMES.get(mode, mode) for mode in grouped.index]
    ax.barh(labels, grouped["mean"], xerr=grouped["std"].fillna(0.0), capsize=3)
    ax.set_xlabel("Final test accuracy")
    ax.set_xlim(max(0.0, grouped["mean"].min() - 0.05), min(1.0, grouped["mean"].max() + 0.03))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_lr_sensitivity_plot(results_dirs: list[str | Path], output_path: str | Path) -> None:
    """Compare final test accuracy across learning rates for label-based orders."""
    df = load_metrics_from_results_dirs(results_dirs)
    df = df[df["order_mode"].isin(LABEL_ORDER_MODES)].copy()
    if df.empty:
        return

    _assert_compatible_across_results(df, LR_SENSITIVITY_COMPATIBILITY_COLUMNS, str(output_path))
    final_epoch = df.groupby(SUMMARY_METADATA_COLUMNS + ["seed"], dropna=False)["epoch"].transform("max")
    final = df[df["epoch"] == final_epoch]
    summary = (
        final.groupby(["order_mode", "learning_rate"], dropna=False)["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    learning_rates = [lr for lr in [0.05, 0.001] if lr in summary["learning_rate"].tolist()]
    if not learning_rates:
        learning_rates = sorted(summary["learning_rate"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(LABEL_ORDER_MODES))
    width = 0.34

    for index, learning_rate in enumerate(learning_rates):
        group = summary[summary["learning_rate"] == learning_rate].set_index("order_mode").reindex(LABEL_ORDER_MODES)
        ax.bar(
            x + (index - (len(learning_rates) - 1) / 2) * width,
            group["mean"],
            width=width,
            yerr=group["std"].fillna(0.0),
            capsize=3,
            label=f"lr={learning_rate:g}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAMES.get(mode, mode) for mode in LABEL_ORDER_MODES])
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_prediction_distribution_plot(
    prediction_distribution_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot the mean predicted-class counts for the label-based orders."""
    df = pd.read_csv(prediction_distribution_path)
    df = df[df["order_mode"].isin(LABEL_ORDER_MODES)].copy()
    if df.empty:
        return

    _assert_single_configuration(df, PLOT_CONFIGURATION_COLUMNS, str(output_path))
    summary = (
        df.groupby(["order_mode", "class_id"], dropna=False)["num_predictions"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, order_mode in zip(axes, LABEL_ORDER_MODES):
        group = summary[summary["order_mode"] == order_mode].sort_values("class_id")
        ax.bar(
            group["class_id"].astype(int).astype(str),
            group["mean"],
            yerr=group["std"].fillna(0.0),
            capsize=2,
        )
        ax.set_title(PRETTY_NAMES.get(order_mode, order_mode), fontsize=10)
        ax.set_xlabel("Predicted class")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean test predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_all_plots(results_dir: str | Path) -> None:
    """Generate all plots and summary CSV files."""
    results_dir = Path(results_dir)
    figures_dir = ensure_dir(results_dir / "figures")
    tables_dir = ensure_dir(results_dir / "tables")

    df = load_metrics(results_dir)
    summary = summarize_by_epoch(df)
    summary.to_csv(tables_dir / "summary_by_epoch.csv", index=False)

    prediction_distribution = load_prediction_distributions(results_dir)
    prediction_distribution.to_csv(tables_dir / "prediction_distribution.csv", index=False)

    final_summary = summarize_final_test_accuracy(df)
    final_summary.to_csv(tables_dir / "final_test_accuracy.csv", index=False)

    plot_df = df
    plot_prediction_distribution_path: Path = tables_dir / "prediction_distribution.csv"

    if results_dir.name == "results":
        plot_df = _load_main_report_plot_df(results_dir)
        lr005_prediction_table = results_dir.parent / "results_lr005" / "tables" / "prediction_distribution.csv"
        if lr005_prediction_table.exists():
            plot_prediction_distribution_path = lr005_prediction_table

        lr_dirs = [
            results_dir.parent / "results_lr005",
            results_dir.parent / "results_lr0001",
        ]
        if all((lr_dir / "raw").exists() for lr_dir in lr_dirs):
            save_lr_sensitivity_plot(
                lr_dirs,
                figures_dir / "lr_sensitivity_label_orders.pdf",
            )

    plot_summary = summarize_by_epoch(plot_df)
    save_line_plot(plot_summary, "test_accuracy", "Test accuracy", figures_dir / "test_accuracy.png")
    save_line_plot(plot_summary, "train_loss", "Training loss", figures_dir / "train_loss.png")
    save_line_plot(plot_summary, "grad_norm_mean", "Mean gradient norm", figures_dir / "grad_norm.png")
    save_final_bar_plot(plot_df, figures_dir / "final_test_accuracy.png")

    save_line_plot(plot_summary, "test_accuracy", "Test accuracy", figures_dir / "test_accuracy.pdf")
    save_line_plot(plot_summary, "train_loss", "Training loss", figures_dir / "train_loss.pdf")
    save_line_plot(plot_summary, "grad_norm_mean", "Mean gradient norm", figures_dir / "grad_norm.pdf")
    save_final_bar_plot(plot_df, figures_dir / "final_test_accuracy.pdf")

    save_prediction_distribution_plot(
        plot_prediction_distribution_path,
        figures_dir / "prediction_distribution_label_orders.pdf",
    )
