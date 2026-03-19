"""Automated evaluation suite for Verra One.

Runs a fixed set of question-answer pairs through the full retrieval + chat
pipeline and scores each answer on whether it contains the expected facts.

Usage:
    verra eval                     # run all cases
    verra eval --category financial  # run a single category
    verra eval --json              # machine-readable output
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Eval cases
# ---------------------------------------------------------------------------

EVAL_CASES: list[dict[str, Any]] = [
    {
        "question": "Which clients have overdue invoices?",
        "must_contain": ["Cedar Valley Medical"],
        "must_not_contain": [],
        "category": "cross-document",
    },
    {
        "question": "What's our total revenue for Q4 2025?",
        "must_contain": ["2,800,420", "Q4_2025_Financial_Summary"],
        "must_not_contain": [],
        "category": "financial",
    },
    {
        "question": "Who is Jennifer Walsh and what's the status of her deal?",
        "must_contain": ["Meridian Health", "Patient Flow Analytics", "March 12", "board"],
        "must_not_contain": [],
        "category": "entity",
    },
    {
        "question": "What caused the November 2025 outage?",
        "must_contain": ["TLS", "certificate", "CloudFront", "expired"],
        "must_not_contain": [],
        "category": "incident",
    },
    {
        "question": "What's our org structure?",
        "must_contain": ["Jake Morrison", "CEO", "Sarah Chen", "David Park"],
        "must_not_contain": [],
        "category": "org",
    },
    {
        "question": "Find all mentions of DataForge",
        "must_contain": ["DataForge", "competitor"],
        "must_not_contain": [],
        "category": "search",
    },
    {
        "question": "What contracts are up for renewal soon?",
        "must_contain": ["renewal", "2025"],
        "must_not_contain": [],
        "category": "contracts",
    },
    {
        "question": "Are any vendors due for contract renewal?",
        "must_contain": ["HubSpot", "Datadog", "WeWork"],
        "must_not_contain": [],
        "category": "vendors",
    },
    {
        "question": "What are our biggest risks right now?",
        "must_contain": ["risk"],
        "must_not_contain": ["I don't have"],
        "category": "risk",
    },
    {
        "question": "What action items came out of the last leadership meeting?",
        "must_contain": ["leadership"],
        "must_not_contain": ["I don't have"],
        "category": "meetings",
    },
    {
        "question": "What are the top P1 support ticket patterns?",
        "must_contain": ["P1", "TICKET"],
        "must_not_contain": ["I don't have"],
        "category": "support",
    },
    {
        "question": "Summarize our deployment and incident response process",
        "must_contain": ["deployment", "incident"],
        "must_not_contain": ["I don't have"],
        "category": "ops",
    },
    {
        "question": "How much PTO did the engineering team take in 2025?",
        "must_contain": ["David Park", "days"],
        "must_not_contain": [],
        "category": "hr",
    },
    {
        "question": "What's our monthly AWS spend trend?",
        "must_contain": ["EC2", "RDS"],
        "must_not_contain": ["I don't have"],
        "category": "cloud",
    },
    {
        "question": "Who works on the Greenfield account?",
        "must_contain": ["Sarah Chen"],
        "must_not_contain": ["I don't have"],
        "category": "team",
    },
    {
        "question": "Compare our Q3 and Q4 2025 financial performance",
        "must_contain": ["Q3", "Q4", "revenue"],
        "must_not_contain": ["What would you like"],
        "category": "comparative",
    },
    {
        "question": "Which sales rep has the best win rate?",
        "must_contain": ["win", "rate"],
        "must_not_contain": [],
        "category": "sales",
    },
    {
        "question": "Summarize the Greenfield Holdings account",
        "must_contain": ["Greenfield", "contract", "retainer"],
        "must_not_contain": [],
        "category": "account",
    },
    {
        "question": "What is the Pinnacle Retail project scope?",
        "must_contain": ["Pinnacle", "inventory"],
        "must_not_contain": ["I don't have"],
        "category": "project",
    },
    {
        "question": "Who is our CTO?",
        "must_contain": ["David Park"],
        "must_not_contain": [],
        "category": "org",
    },
    {
        "question": "What HIPAA compliance measures do we have?",
        "must_contain": ["HIPAA"],
        "must_not_contain": ["I don't have"],
        "category": "compliance",
    },
    {
        "question": "What is the Cedar Valley Medical case study about?",
        "must_contain": ["Cedar Valley", "patient flow", "wait time"],
        "must_not_contain": [],
        "category": "marketing",
    },
    {
        "question": "What is our remote work policy?",
        "must_contain": ["remote", "days"],
        "must_not_contain": ["I don't have"],
        "category": "policy",
    },
    {
        "question": "What happened with the ML model accuracy drop in January 2026?",
        "must_contain": ["accuracy", "Houston", "POS"],
        "must_not_contain": [],
        "category": "incident",
    },
    {
        "question": "What is InsightWorks and how do they compare to us?",
        "must_contain": ["InsightWorks", "Austin"],
        "must_not_contain": ["I don't have"],
        "category": "competitive",
    },
]


# ---------------------------------------------------------------------------
# Engine bootstrap
# ---------------------------------------------------------------------------

def _build_engine() -> Any:
    """Initialise and return a ChatEngine using the user's live data."""
    from verra.agent.chat import ChatEngine
    from verra.agent.llm import LLMClient
    from verra.config import VERRA_HOME, ensure_data_dir, load_config
    from verra.store.db import DatabaseManager
    from verra.store.memory import MemoryStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    config = load_config()

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")
    memory_store = MemoryStore.from_connection(db.core)
    llm = LLMClient(model=config.agent.model)

    entity_store = None
    try:
        from verra.store.entities import EntityStore
        entity_store = EntityStore.from_connection(db.core)
    except Exception:
        pass

    engine = ChatEngine(
        llm=llm,
        metadata_store=metadata_store,
        vector_store=vector_store,
        memory_store=memory_store,
        entity_store=entity_store,
    )

    # Attach db so the caller can close it
    engine._eval_db = db  # type: ignore[attr-defined]
    return engine


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _score_case(
    case: dict[str, Any],
    engine: Any,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a single eval case and return a result dict."""
    question = case["question"]
    must_contain: list[str] = case.get("must_contain", [])
    must_not_contain: list[str] = case.get("must_not_contain", [])
    category: str = case.get("category", "general")

    t0 = time.monotonic()
    try:
        response = engine.ask(question)
        answer = response.answer
    except Exception as exc:
        answer = f"[ERROR: {exc}]"
    elapsed = time.monotonic() - t0

    answer_lower = answer.lower()

    # Check must_contain (case-insensitive)
    hits = sum(1 for term in must_contain if term.lower() in answer_lower)
    total = len(must_contain)

    # Check must_not_contain
    violations = [term for term in must_not_contain if term.lower() in answer_lower]

    passed = (hits == total) and not violations

    # Score: fraction of must_contain terms found, 0 if any violations
    if violations:
        score = 0.0
    elif total > 0:
        score = hits / total
    else:
        score = 1.0

    result: dict[str, Any] = {
        "question": question,
        "category": category,
        "passed": passed,
        "score": score,
        "hits": hits,
        "total": total,
        "violations": violations,
        "elapsed_s": round(elapsed, 1),
        "answer_preview": answer[:200].replace("\n", " "),
    }

    if verbose:
        status = "PASS" if passed else "FAIL"
        label = f"[{status}]"
        miss_info = ""
        if not passed:
            missing = [t for t in must_contain if t.lower() not in answer_lower]
            if missing:
                miss_info = f" | missing: {', '.join(missing[:3])}"
            if violations:
                miss_info += f" | violations: {', '.join(violations[:2])}"
        print(
            f"  {label:<6} {question[:60]:<62}  ({hits}/{total}){miss_info}"
        )

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(
    category_filter: str | None = None,
    output_json: bool = False,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the full eval suite and return results.

    Parameters
    ----------
    category_filter:
        If set, only run cases whose ``category`` matches this string.
    output_json:
        If True, print a JSON summary to stdout instead of the table.
    verbose:
        If True (default), print per-case PASS/FAIL lines as they run.
    """
    cases = EVAL_CASES
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]
        if not cases:
            print(
                f"  No eval cases for category '{category_filter}'. "
                f"Available: {sorted({c['category'] for c in EVAL_CASES})}"
            )
            return []

    engine = _build_engine()

    if verbose:
        print()
        print(f"  Running {len(cases)} eval cases...")
        print()

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            result = _score_case(case, engine, verbose=verbose)
            results.append(result)
    finally:
        # Always close the DB, even if the run is interrupted
        try:
            engine._eval_db.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_cases = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / total_cases if total_cases else 0.0
    avg_elapsed = sum(r["elapsed_s"] for r in results) / total_cases if total_cases else 0.0

    if output_json:
        summary = {
            "passed": passed_count,
            "total": total_cases,
            "pct": round(avg_score * 100, 1),
            "avg_elapsed_s": round(avg_elapsed, 1),
            "results": results,
        }
        print(json.dumps(summary, indent=2))
        return results

    if verbose:
        print()
        _print_summary(results, passed_count, total_cases, avg_score, avg_elapsed)

    return results


def _print_summary(
    results: list[dict[str, Any]],
    passed_count: int,
    total_cases: int,
    avg_score: float,
    avg_elapsed: float,
) -> None:
    """Print a human-readable summary table."""
    pct = avg_score * 100
    bar_filled = int(pct / 5)  # 20-char bar
    bar = "#" * bar_filled + "-" * (20 - bar_filled)

    print(f"  Score:   {passed_count}/{total_cases} passed  [{bar}] {pct:.0f}%")
    print(f"  Avg latency: {avg_elapsed:.1f}s per question")
    print()

    # Per-category breakdown
    cats: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r["passed"])

    print("  By category:")
    for cat, passes in sorted(cats.items()):
        cat_passed = sum(passes)
        cat_total = len(passes)
        cat_pct = cat_passed / cat_total * 100
        bar_f = int(cat_pct / 10)  # 10-char bar
        cat_bar = "#" * bar_f + "-" * (10 - bar_f)
        indicator = "OK" if cat_pct == 100 else ("  " if cat_pct >= 50 else "!!")
        print(
            f"    {indicator}  {cat:<16} {cat_passed}/{cat_total}  "
            f"[{cat_bar}] {cat_pct:.0f}%"
        )
    print()

    # List failures for quick debugging
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"  {len(failures)} failing case(s):")
        for r in failures:
            missing = []
            for term in EVAL_CASES:
                if term["question"] == r["question"]:
                    missing = [
                        t for t in term.get("must_contain", [])
                        if t.lower() not in r.get("answer_preview", "").lower()
                    ]
                    break
            missing_str = f"  missing: {', '.join(missing[:4])}" if missing else ""
            print(f"    - {r['question'][:70]}{missing_str}")
        print()
