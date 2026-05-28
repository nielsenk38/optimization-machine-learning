"""Run the additional analyses used in Rapport_V3.

This script keeps the original project intact and writes new outputs to
dedicated V3 directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_fashion_mnist, make_test_loader, make_train_loader
from src.plotting import make_all_plots
from src.plotting_v3 import MAIN_ORDER_MODES, SWEEP_ORDER_MODES, make_v3_outputs
from src.train import TrainConfig, get_run_artifact_paths, run_single_experiment
from src.utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the extended V3 data-order analyses.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-root", type=str, default="results_v3")
    parser.add_argument("--main-output-dir", type=str, default="results_v3_main")
    parser.add_argument(
        "--sweep-learning-rates",
        nargs="+",
        type=float,
        default=[0.1, 0.01, 0.001],
        help="Additional learning rates beyond the main 0.05 run.",
    )
    parser.add_argument(
        "--extra-sweep-dirs",
        nargs="*",
        default=[],
        help="Existing result directories to include when building the LR sweep summaries.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-examples", type=int, default=20000)
    parser.add_argument("--subset-seed", type=int, default=123)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--model", choices=["small_cnn", "logistic_regression"], default="small_cnn")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    return parser.parse_args()


def lr_tag(learning_rate: float) -> str:
    return f"{learning_rate:g}".replace(".", "p")


def run_experiment_directory(
    *,
    output_dir: Path,
    experiment_name: str,
    bundle,
    test_loader,
    order_modes: list[str],
    seeds: list[int],
    learning_rate: float,
    args: argparse.Namespace,
    save_embeddings: bool,
) -> None:
    raw_dir = ensure_dir(output_dir / "raw")
    save_json(
        {
            "data_dir": args.data_dir,
            "output_dir": str(output_dir),
            "orders": order_modes,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_train_examples": args.max_train_examples,
            "subset_seed": args.subset_seed,
            "optimizer": args.optimizer,
            "model": args.model,
            "lr": learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "num_workers": args.num_workers,
            "deterministic": args.deterministic,
            "save_embeddings": save_embeddings,
            "experiment_name": experiment_name,
        },
        output_dir / "experiment_args.json",
    )

    max_train_examples_record = len(bundle.train_dataset)
    device = get_device(args.device)
    for order_mode in order_modes:
        for seed in seeds:
            config = TrainConfig(
                order_mode=order_mode,
                seed=seed,
                experiment_name=experiment_name,
                output_dir=str(output_dir),
                model=args.model,
                optimizer=args.optimizer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_train_examples=max_train_examples_record,
                learning_rate=learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                device=str(device),
                deterministic=args.deterministic,
                save_embeddings=save_embeddings,
            )
            artifact_paths = get_run_artifact_paths(config, raw_dir)
            if all(path.exists() for path in artifact_paths.values()) and not args.force:
                print(f"Skipping existing run: {artifact_paths['metrics']}")
                continue

            train_loader, train_sampler = make_train_loader(
                bundle=bundle,
                order_mode=order_mode,
                seed=seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            print(f"\nRunning dir={experiment_name}, order={order_mode}, lr={learning_rate:g}, seed={seed}")
            run_single_experiment(
                config=config,
                train_loader=train_loader,
                train_sampler=train_sampler,
                test_loader=test_loader,
                output_dir=raw_dir,
                show_progress=not args.no_progress,
            )

    make_all_plots(output_dir)


def main() -> None:
    args = parse_args()
    results_root = ensure_dir(args.results_root)
    main_output_dir = Path(args.main_output_dir)
    sweep_output_dirs = [Path(f"results_v3_lr{lr_tag(lr)}") for lr in args.sweep_learning_rates]

    if not args.plot_only:
        device = get_device(args.device)
        print(f"Using device: {device}")
        print("Loading Fashion-MNIST once for all V3 runs...")
        max_train_examples = None if args.max_train_examples <= 0 else args.max_train_examples
        bundle = load_fashion_mnist(
            data_dir=args.data_dir,
            max_train_examples=max_train_examples,
            subset_seed=args.subset_seed,
        )
        test_loader = make_test_loader(bundle, batch_size=args.batch_size, num_workers=args.num_workers)

        if not args.skip_main:
            run_experiment_directory(
                output_dir=main_output_dir,
                experiment_name=main_output_dir.name,
                bundle=bundle,
                test_loader=test_loader,
                order_modes=list(MAIN_ORDER_MODES),
                seeds=list(args.seeds),
                learning_rate=0.05,
                args=args,
                save_embeddings=True,
            )

        if not args.skip_sweep:
            for learning_rate, output_dir in zip(args.sweep_learning_rates, sweep_output_dirs):
                run_experiment_directory(
                    output_dir=output_dir,
                    experiment_name=output_dir.name,
                    bundle=bundle,
                    test_loader=test_loader,
                    order_modes=list(SWEEP_ORDER_MODES),
                    seeds=list(args.seeds),
                    learning_rate=learning_rate,
                    args=args,
                    save_embeddings=False,
                )
    else:
        if not main_output_dir.exists():
            raise FileNotFoundError(f"Main V3 results directory not found: {main_output_dir}")

    sweep_dirs_for_summary = [main_output_dir] + [output_dir for output_dir in sweep_output_dirs if output_dir.exists()]
    sweep_dirs_for_summary.extend(Path(path) for path in args.extra_sweep_dirs if Path(path).exists())
    print("Generating Rapport_V3 figures and tables...")
    make_v3_outputs(main_output_dir, sweep_dirs_for_summary, results_root)
    print(f"Done. V3 results are in: {results_root}")


if __name__ == "__main__":
    main()
