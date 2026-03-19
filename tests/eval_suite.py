"""Thin wrapper — runs the eval suite via verra.eval.

Can be invoked directly:
    .venv/bin/python tests/eval_suite.py
    .venv/bin/python tests/eval_suite.py --category financial
    .venv/bin/python tests/eval_suite.py --json

Or via the CLI:
    verra eval
"""

from __future__ import annotations

import sys


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verra evaluation suite")
    parser.add_argument(
        "--category",
        metavar="CAT",
        default=None,
        help="Only run cases in this category",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    from verra.eval import run_eval

    results = run_eval(
        category_filter=args.category,
        output_json=args.output_json,
    )

    # Exit with non-zero if any cases failed (useful in CI)
    failures = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
