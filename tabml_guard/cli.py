"""TabML-Guard Command Line Interface.

Usage:
    tabml-guard run --config config.yaml
    tabml-guard run --data sales.csv --target revenue --time date
    tabml-guard init
    tabml-guard index --results ./results
    tabml-guard version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .__version__ import __version__
from .config import Config, DatasetConfig, ReportConfig, TuningConfig, load_config
from .pipeline import TabMLGuard


def cmd_run(args: argparse.Namespace) -> int:
    """Run the TabML-Guard AutoML and diagnostic pipeline."""
    if args.config:
        cfg = load_config(args.config)
        guard = TabMLGuard(config=cfg)
    else:
        if not args.data:
            print("Error: Specify either --config <path.yaml> or --data <path.csv>", file=sys.stderr)
            return 1
        if not args.target:
            print("Error: --target is required when using --data", file=sys.stderr)
            return 1

        guard = TabMLGuard(
            target=args.target,
            time_col=args.time,
            task=args.task or "auto",
            output_dir=args.output or "results",
            drop_cols=args.drop or [],
            budget_trials=args.budget or 40,
            title=f"TabML-Guard — {Path(args.data).stem}",
        )
        guard.config.dataset.path = args.data
        guard.config.dataset.source = "csv"

    print(f"Starting TabML-Guard v{__version__}...")
    guard.fit()

    print("\n" + "=" * 70)
    print("Run completed successfully!")
    print(f"HTML Report: {guard.html_report_path_}")
    print(f"PDF Report:  {guard.pdf_report_path_}")
    print("=" * 70)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold a fresh config.yaml in the working directory."""
    dest = Path(args.output or "config.yaml")
    if dest.exists() and not args.force:
        print(f"Error: '{dest}' already exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    template_path = Path(__file__).parent / "configs" / "default.yaml"
    if template_path.exists():
        content = template_path.read_text(encoding="utf-8")
    else:
        content = (
            "dataset:\n"
            "  source: csv\n"
            "  path: data/my_data.csv\n"
            "  target: target_col\n"
            "  time_col: null\n"
            "task: auto\n"
            "output_dir: results\n"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"Scaffolded config template at: {dest.resolve()}")
    print(f"To run: tabml-guard run --config {dest.name}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Regenerate cross-run INDEX.md comparison table."""
    from .make_index import build_index

    results_dir = Path(args.results or "results")
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found.", file=sys.stderr)
        return 1

    idx_content = build_index(results_dir)
    idx_path = results_dir / "INDEX.md"
    idx_path.write_text(idx_content, encoding="utf-8")
    print(f"Updated results index: {idx_path.resolve()}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print package version and environment info."""
    print(f"TabML-Guard v{__version__}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tabml-guard",
        description="TabML-Guard: Explainable, Drift-Aware AutoML for Time-Ordered Tabular Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tabml-guard run --config config.yaml
  tabml-guard run --data transactions.csv --target is_fraud --time timestamp
  tabml-guard init
  tabml-guard index --results ./results
""",
    )
    parser.add_argument("-v", "--version", action="version", version=f"tabml-guard {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Command: run
    p_run = subparsers.add_parser("run", help="Run the AutoML diagnostic pipeline")
    g = p_run.add_mutually_exclusive_group()
    g.add_argument("--config", metavar="FILE", help="Path to YAML config file")
    g.add_argument("--data", metavar="CSV", help="Path to raw CSV dataset")

    p_run.add_argument("--target", metavar="COL", help="Target column to predict")
    p_run.add_argument("--time", metavar="COL", help="Time column for chronological order")
    p_run.add_argument("--drop", metavar="COL", nargs="*", help="Columns to drop (leakage/IDs)")
    p_run.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    p_run.add_argument("--budget", type=int, default=40, help="Tuning budget trials (default: 40)")
    p_run.add_argument("--output", metavar="DIR", default="results", help="Output directory")

    # Command: init
    p_init = subparsers.add_parser("init", help="Scaffold a new config.yaml")
    p_init.add_argument("--output", metavar="FILE", default="config.yaml", help="Destination file path")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")

    # Command: index
    p_index = subparsers.add_parser("index", help="Rebuild cross-dataset results/INDEX.md")
    p_index.add_argument("--results", metavar="DIR", default="results", help="Results directory to scan")

    # Command: version
    subparsers.add_parser("version", help="Show TabML-Guard version")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    dispatch = {
        "run": cmd_run,
        "init": cmd_init,
        "index": cmd_index,
        "version": cmd_version,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

