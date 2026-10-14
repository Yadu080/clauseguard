"""Evaluate ClauseGuard on data/benchmark.csv.

Matching definition
-------------------
Pair level (main metric)
    predicted_drift = the engine produced at least one HIGH or MEDIUM finding.
    REVIEW findings do NOT count as a drift prediction; they mean "look again".
    A "strict" variant where REVIEW also counts is reported alongside.

Category level
    For each category c in (negation, modality, condition, quantifier, numeric):
    gold   = c is listed in expected_categories
    pred   = the engine produced a HIGH/MEDIUM finding with category c
    Counted per pair, so one pair contributes at most one TP/FP/FN per category.
    The category 'other' (changes outside the five categories) has no
    predictions by design and is only counted in the pair-level metric.

Usage
    python scripts/evaluate.py                      # all splits, writes reports/
    python scripts/evaluate.py --split dev          # development split only
    python scripts/evaluate.py --split dev --no-save
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzer import analyze  # noqa: E402
from src.schemas import CATEGORIES, ENGINE_VERSION  # noqa: E402

DATA = ROOT / "data" / "benchmark.csv"
REPORTS = ROOT / "reports"


def _safe_div(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def prf(tp: int, fp: int, fn: int, tn: int | None = None) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (round(2 * precision * recall / (precision + recall), 4)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    if tn is not None:
        out["tn"] = tn
        out["accuracy"] = _safe_div(tp + tn, tp + fp + fn + tn)
    return out


def split_categories(value) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {c.strip() for c in value.split("|") if c.strip()}


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run the same engine as the app on every pair."""
    rows = []
    for rec in df.to_dict("records"):
        result = analyze(rec["original_text"], rec["rewritten_text"])
        rows.append({
            **rec,
            "predicted_drift": int(result.has_drift()),
            "predicted_drift_strict": int(result.has_drift(include_review=True)),
            "predicted_categories": "|".join(sorted(result.predicted_categories())),
            "status": result.status.value,
            "findings": "; ".join(f"{f.severity.value}:{f.rule_id}" for f in result.findings),
        })
    return pd.DataFrame(rows)


def pair_metrics(pred: pd.DataFrame, column: str = "predicted_drift") -> dict:
    gold, p = pred["expected_drift"].astype(int), pred[column].astype(int)
    return prf(int(((gold == 1) & (p == 1)).sum()), int(((gold == 0) & (p == 1)).sum()),
               int(((gold == 1) & (p == 0)).sum()), int(((gold == 0) & (p == 0)).sum()))


def category_metrics(pred: pd.DataFrame) -> dict:
    out = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for cat in CATEGORIES:
        tp = fp = fn = 0
        for gold_s, pred_s in zip(pred["expected_categories"], pred["predicted_categories"]):
            g, p = cat in split_categories(gold_s), cat in split_categories(pred_s)
            tp += g and p
            fp += p and not g
            fn += g and not p
        m = prf(tp, fp, fn)
        m["support"] = tp + fn
        out[cat] = m
        for k in totals:
            totals[k] += m[k]
    out["micro_avg"] = {**prf(totals["tp"], totals["fp"], totals["fn"]),
                        "support": totals["tp"] + totals["fn"]}
    return out


def mutation_breakdown(pred: pd.DataFrame) -> dict:
    out = {}
    for mutation, group in pred.groupby("mutation_type"):
        flagged = int(group["predicted_drift"].sum())
        out[mutation] = {"pairs": len(group), "flagged_as_drift": flagged,
                         "flag_rate": _safe_div(flagged, len(group))}
    return out


def evaluate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pred = predict(df)
    splits = {}
    for name in ["dev", "test"]:
        part = pred[pred["split"] == name]
        if part.empty:
            continue
        splits[name] = {
            "pairs": len(part),
            "pair_level": pair_metrics(part),
            "pair_level_strict": pair_metrics(part, "predicted_drift_strict"),
            "category_level": category_metrics(part),
            "by_mutation_type": mutation_breakdown(part),
        }
    if len(splits) > 1:
        splits["all"] = {
            "pairs": len(pred),
            "pair_level": pair_metrics(pred),
            "pair_level_strict": pair_metrics(pred, "predicted_drift_strict"),
            "category_level": category_metrics(pred),
            "by_mutation_type": mutation_breakdown(pred),
        }
    errors = pred[pred["expected_drift"] != pred["predicted_drift"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": ENGINE_VERSION,
        "dataset": str(DATA.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "matching_definition": {
            "pair_level": "predicted drift = at least one HIGH or MEDIUM finding",
            "pair_level_strict": "predicted drift = at least one finding of any severity (incl. REVIEW)",
            "category_level": "category predicted = at least one HIGH/MEDIUM finding in that category; counted once per pair",
        },
        "splits": splits,
        "errors": [
            {k: rec[k] for k in ("example_id", "split", "original_text", "rewritten_text", "expected_drift",
                                 "expected_categories", "predicted_drift", "predicted_categories",
                                 "mutation_type", "findings")}
            for rec in errors.fillna("").to_dict("records")
        ],
    }
    return report, pred


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:5.3f}"


def print_report(report: dict) -> None:
    print(f"ClauseGuard {report['engine_version']} on {report['dataset']}")
    for name, s in report["splits"].items():
        pl, st = s["pair_level"], s["pair_level_strict"]
        print(f"\n== {name.upper()} ({s['pairs']} pairs) ==")
        print(f"pair level   P {_fmt(pl['precision'])}  R {_fmt(pl['recall'])}  F1 {_fmt(pl['f1'])}  "
              f"acc {_fmt(pl['accuracy'])}   TP {pl['tp']} FP {pl['fp']} FN {pl['fn']} TN {pl['tn']}")
        print(f"strict(+REV) P {_fmt(st['precision'])}  R {_fmt(st['recall'])}  F1 {_fmt(st['f1'])}  "
              f"acc {_fmt(st['accuracy'])}   TP {st['tp']} FP {st['fp']} FN {st['fn']} TN {st['tn']}")
        print(f"{'category':<12} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
        for cat, m in s["category_level"].items():
            print(f"{cat:<12} {_fmt(m['precision']):>6} {_fmt(m['recall']):>6} {_fmt(m['f1']):>6} {m['support']:>8}")
    print(f"\nMisclassified pairs (pair level): {len(report['errors'])}")
    for e in report["errors"]:
        kind = "FN" if e["expected_drift"] == 1 else "FP"
        print(f"  [{kind}] {e['example_id']} ({e['split']}, {e['mutation_type']}): {e['rewritten_text']}")
        print(f"        findings: {e['findings'] or '-'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--no-save", action="store_true", help="print only; do not write reports/")
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    if args.split != "all":
        df = df[df["split"] == args.split]
    report, pred = evaluate(df)
    print_report(report)

    if not args.no_save:
        REPORTS.mkdir(exist_ok=True)
        suffix = "" if args.split == "all" else f"_{args.split}"
        (REPORTS / f"evaluation_results{suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        pred.to_csv(REPORTS / f"evaluation_predictions{suffix}.csv", index=False)
        print(f"\nSaved reports/evaluation_results{suffix}.json and reports/evaluation_predictions{suffix}.csv")


if __name__ == "__main__":
    main()
