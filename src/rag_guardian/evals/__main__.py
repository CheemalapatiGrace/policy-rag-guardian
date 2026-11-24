"""CLI: ``python -m rag_guardian.evals`` - prints a report, exits 1 when any gate fails (CI quality gate)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from ..config import Settings
from ..pipeline import build_pipeline
from ..storage import Database
from .runner import (
    DEFAULT_THRESHOLDS,
    compute_metrics,
    evaluate_thresholds,
    format_report,
    load_cases,
    run_cases,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m rag_guardian.evals", description=__doc__)
    ap.add_argument("--cases", type=Path, help="golden JSONL (default: data/evals/golden.jsonl)")
    ap.add_argument("--regression", type=Path, help="HITL regression JSONL (default: data/evals/regression.jsonl)")
    ap.add_argument("--no-regression", action="store_true", help="skip the reviewer-fed regression set")
    ap.add_argument("--retrieval-mode", choices=["hybrid", "bm25", "vector"], help="ablation: override retrieval mode")
    ap.add_argument("--no-rerank", action="store_true", help="ablation: keep fusion order (no reranking)")
    ap.add_argument("--threshold", action="append", default=[], metavar="METRIC=VALUE",
                    help=f"override a gate; metrics: {', '.join(DEFAULT_THRESHOLDS)}")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of the table")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = Settings.from_env()
    overrides: dict[str, float] = {}
    if os.environ.get("RAG_EVAL_THRESHOLDS"):
        overrides.update(json.loads(os.environ["RAG_EVAL_THRESHOLDS"]))
    for item in args.threshold:
        name, _, value = item.partition("=")
        if name not in DEFAULT_THRESHOLDS:
            print(f"unknown metric {name!r}", file=sys.stderr)
            return 2
        overrides[name] = float(value)

    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(base, db_path=Path(tmp) / "eval.db", regression_path=Path(tmp) / "reg.jsonl", index_dir=None,
                           retrieval_mode=args.retrieval_mode or base.retrieval_mode, rerank=not args.no_rerank and base.rerank)
        pipeline = build_pipeline(settings, db=Database(settings.db_path))
        cases = load_cases(args.cases or base.golden_path)
        if not args.no_regression:
            cases += load_cases(args.regression or base.regression_path)
        if not cases:
            print("no eval cases found", file=sys.stderr)
            return 2
        results = run_cases(pipeline, cases)
        metrics = compute_metrics(pipeline, results)
        gates = evaluate_thresholds(metrics, overrides)
        failed = [g for g in gates if not g["passed"]]
        if args.json:
            print(json.dumps({"metrics": metrics, "gates": gates, "passed": not failed}, indent=2))
        else:
            print(format_report(metrics, gates, results, args.verbose))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
