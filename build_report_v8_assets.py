"""Build report-only figures and tables for the final report from existing outputs."""

from __future__ import annotations

import argparse

from src.plotting_v8 import make_v8_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-report assets from existing result directories.")
    parser.add_argument("--main-results-dir", type=str, default="results_v3_main")
    parser.add_argument("--lr010-results-dir", type=str, default="results_v3_lr0p1")
    parser.add_argument("--rescue-results-dir", type=str, default="results_lr0001")
    parser.add_argument("--output-dir", type=str, default="results_v8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_v8_outputs(
        main_results_dir=args.main_results_dir,
        lr010_results_dir=args.lr010_results_dir,
        rescue_results_dir=args.rescue_results_dir,
        output_dir=args.output_dir,
    )
    print(f"Final report assets written to: {args.output_dir}")


if __name__ == "__main__":
    main()
