"""
Break down results_<condition>.jsonl accuracy by eval axis (axis_terminology,
axis_structure, axis_time, difficulty), instead of one aggregate number.

Motivation (Phase 0 protocol): a single P0/P1/P2 accuracy number hides
whether extra schema/glossary context helps or hurts for specific question
types. The iid_easy_03 case showed P2 regressing vs. P1 on a `literal`
question because column-trap documentation made deleted_at more salient
than the question warranted -- that's exactly the kind of effect this
script is meant to surface systematically, instead of by manual inspection
of single JSONL lines.

Usage:
    python3 analyze_by_axis.py eval_v1.jsonl results_p0.jsonl results_p1.jsonl results_p2.jsonl
    python3 analyze_by_axis.py eval_v1.jsonl results_p0.jsonl results_p1.jsonl results_p2.jsonl --axis axis_terminology
    python3 analyze_by_axis.py eval_v1.jsonl results_p0.jsonl results_p1.jsonl results_p2.jsonl --errors-only
"""

import argparse
import json
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_path", help="eval_v1.jsonl (source of axis labels)"
    )
    parser.add_argument(
        "result_paths", nargs="+", help="one or more results_p*.jsonl files"
    )
    parser.add_argument(
        "--axis",
        default="axis_terminology",
        choices=[
            "axis_terminology",
            "axis_structure",
            "axis_time",
            "difficulty",
        ],
        help="which axis to break accuracy down by (default: axis_terminology)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="print each example that was wrong (result_equivalent == false) instead of the summary table",
    )
    args = parser.parse_args()

    axis_by_id = {}
    for row in load_jsonl(args.eval_path):
        axis_by_id[row["example_id"]] = row

    for result_path in args.result_paths:
        results = load_jsonl(result_path)
        condition = results[0]["condition"] if results else "?"

        if args.errors_only:
            print(
                f"\n=== {result_path} (condition={condition}) -- wrong examples ==="
            )
            for r in results:
                if not r.get("result_equivalent", False):
                    ex = axis_by_id.get(r["example_id"], {})
                    print(
                        f"  {r['example_id']:15s} "
                        f"terminology={ex.get('axis_terminology', '?'):10s} "
                        f"structure={ex.get('axis_structure', '?'):10s} "
                        f"time={ex.get('axis_time', '?'):12s} "
                        f"difficulty={ex.get('difficulty', '?'):6s} "
                        f"error_class={r.get('error_class')}"
                    )
                    print(f"      question: {ex.get('question_de', '?')}")
            continue

        buckets = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in results:
            ex = axis_by_id.get(r["example_id"])
            if ex is None:
                continue
            key = ex[args.axis]
            buckets[key]["total"] += 1
            if r.get("result_equivalent", False):
                buckets[key]["correct"] += 1

        print(
            f"\n=== {result_path} (condition={condition}) -- accuracy by {args.axis} ==="
        )
        for key in sorted(buckets):
            b = buckets[key]
            acc = b["correct"] / b["total"] if b["total"] else 0.0
            print(
                f"  {key:12s}  {b['correct']:3d}/{b['total']:<3d}  {acc:.3f}"
            )
        total_correct = sum(b["correct"] for b in buckets.values())
        total_n = sum(b["total"] for b in buckets.values())
        overall = total_correct / total_n if total_n else 0.0
        print(
            f"  {'ALL':12s}  {total_correct:3d}/{total_n:<3d}  {overall:.3f}"
        )


if __name__ == "__main__":
    main()
