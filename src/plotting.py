"""Plotting utilities for the data-order experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


PRETTY_NAMES = {
    "random": "Random reshuffle",
    "fixed_random": "Fixed random",
    "label_sorted": "Label sorted",
    "label_block_random": "Random label blocks",
    "curriculum_easy": "Easy-to-hard",
    "curriculum_hard": "Hard-to-easy",
}


def load_metrics(raw_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all metrics CSV files from a raw results directory."""
    paths = sorted(Path(raw_dir).glob("metrics_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics_*.csv files found in {raw_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def summarize_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation across seeds."""
    metrics = [
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "grad_norm_mean",
        "grad_norm_std",
        "epoch_time_sec",
    ]
    summary = df.groupby(["order_mode", "epoch"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    return summary


def save_line_plot(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Save a line plot with mean +/- one standard deviation across seeds."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for order_mode, group in summary.groupby("order_mode"):
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
    final_epoch = int(df["epoch"].max())
    final = df[df["epoch"] == final_epoch]
    grouped = final.groupby("order_mode")["test_accuracy"].agg(["mean", "std"]).sort_values("mean")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [PRETTY_NAMES.get(mode, mode) for mode in grouped.index]
    ax.barh(labels, grouped["mean"], xerr=grouped["std"].fillna(0.0), capsize=3)
    ax.set_xlabel("Final test accuracy")
    ax.set_xlim(max(0.0, grouped["mean"].min() - 0.05), min(1.0, grouped["mean"].max() + 0.03))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_all_plots(results_dir: str | Path) -> None:
    """Generate all plots and summary CSV files."""
    results_dir = Path(results_dir)
    raw_dir = results_dir / "raw"
    figures_dir = ensure_dir(results_dir / "figures")
    tables_dir = ensure_dir(results_dir / "tables")

    df = load_metrics(raw_dir)
    summary = summarize_by_epoch(df)
    summary.to_csv(tables_dir / "summary_by_epoch.csv", index=False)

    final_epoch = int(df["epoch"].max())
    final = df[df["epoch"] == final_epoch]
    final_summary = final.groupby("order_mode")["test_accuracy"].agg(["mean", "std", "count"]).reset_index()
    final_summary.to_csv(tables_dir / "final_test_accuracy.csv", index=False)

    save_line_plot(summary, "test_accuracy", "Test accuracy", figures_dir / "test_accuracy.png")
    save_line_plot(summary, "train_loss", "Training loss", figures_dir / "train_loss.png")
    save_line_plot(summary, "grad_norm_mean", "Mean gradient norm", figures_dir / "grad_norm.png")
    save_final_bar_plot(df, figures_dir / "final_test_accuracy.png")

    # Also save PDF versions, which are convenient for LaTeX reports.
    save_line_plot(summary, "test_accuracy", "Test accuracy", figures_dir / "test_accuracy.pdf")
    save_line_plot(summary, "train_loss", "Training loss", figures_dir / "train_loss.pdf")
    save_line_plot(summary, "grad_norm_mean", "Mean gradient norm", figures_dir / "grad_norm.pdf")
    save_final_bar_plot(df, figures_dir / "final_test_accuracy.pdf")
