"""Reproduce the data-order mini-project experiments.

Example:
    python run.py --epochs 8 --seeds 0 1 2 --max-train-examples 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import ORDER_MODES, load_fashion_mnist, make_test_loader, make_train_loader
from src.plotting import make_all_plots
from src.train import TrainConfig, run_single_experiment
from src.utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study how training-data order affects optimization.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory where Fashion-MNIST is stored.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for CSV files and figures.")
    parser.add_argument(
        "--orders",
        nargs="+",
        default=[
            "random",
            "fixed_random",
            "label_sorted",
            "label_block_random",
            "curriculum_easy",
            "curriculum_hard",
        ],
        choices=ORDER_MODES,
        help="Data-order strategies to compare.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs per run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=20000,
        help="Use a stratified subset for faster iteration. Set to 60000 for the full train set.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=123,
        help="Seed used only to choose the optional stratified subset.",
    )
    parser.add_argument("--model", choices=["small_cnn", "logistic_regression"], default="small_cnn")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Use 0 for reproducibility.")
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch algorithms.")
    parser.add_argument("--plot-only", action="store_true", help="Skip training and only regenerate plots.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing run CSV files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars and epoch prints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    save_json(vars(args), output_dir / "experiment_args.json")

    if not args.plot_only:
        device = get_device(args.device)
        print(f"Using device: {device}")
        print("Loading Fashion-MNIST...")
        max_train_examples = None if args.max_train_examples <= 0 else args.max_train_examples
        bundle = load_fashion_mnist(
            data_dir=args.data_dir,
            max_train_examples=max_train_examples,
            subset_seed=args.subset_seed,
        )
        test_loader = make_test_loader(bundle, batch_size=args.batch_size, num_workers=args.num_workers)

        for order_mode in args.orders:
            for seed in args.seeds:
                metrics_path = Path(raw_dir) / f"metrics_{order_mode}_seed{seed}.csv"
                distribution_path = Path(raw_dir) / f"prediction_distribution_{order_mode}_seed{seed}.csv"
                if metrics_path.exists() and distribution_path.exists() and not args.force:
                    print(f"Skipping existing run: {metrics_path}")
                    continue

                train_loader, train_sampler = make_train_loader(
                    bundle=bundle,
                    order_mode=order_mode,
                    seed=seed,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                config = TrainConfig(
                    order_mode=order_mode,
                    seed=seed,
                    model=args.model,
                    optimizer=args.optimizer,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    device=str(device),
                    deterministic=args.deterministic,
                )
                print(f"\nRunning order={order_mode}, seed={seed}")
                run_single_experiment(
                    config=config,
                    train_loader=train_loader,
                    train_sampler=train_sampler,
                    test_loader=test_loader,
                    output_dir=raw_dir,
                    show_progress=not args.no_progress,
                )

    print("Generating plots and summary tables...")
    make_all_plots(output_dir)
    print(f"Done. Results are in: {output_dir}")


if __name__ == "__main__":
    main()
