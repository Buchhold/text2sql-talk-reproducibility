"""Validate SFT candidates against the frozen ecom-v1 evaluation set."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

REQUIRED = {"example_id", "seed_id", "schema", "difficulty", "axis_structure", "axis_time", "axis_terminology", "question_de", "gold_sql"}
DYNAMIC_TIME = re.compile(r"\bCURRENT_(?:DATE|TIMESTAMP)\s*\(", re.I)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def validate(train: list[dict], evaluation: list[dict], require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    eval_seeds = {row["seed_id"] for row in evaluation}
    eval_questions = {normalize(row["question_de"]) for row in evaluation}
    eval_sql = {normalize(row["gold_sql"]) for row in evaluation}
    ids: set[str] = set()
    seeds: set[str] = set()
    questions: set[str] = set()
    sqls: set[str] = set()
    for line_number, row in enumerate(train, 1):
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"line {line_number}: missing {sorted(missing)}")
            continue
        if row["schema"] != "ecom-v1":
            errors.append(f"line {line_number}: unexpected schema")
        if row["difficulty"] not in {"easy", "medium", "hard"}:
            errors.append(f"line {line_number}: invalid difficulty")
        if not row["seed_id"].startswith("train_"):
            errors.append(f"line {line_number}: seed_id must start with train_")
        for label, value, seen in (("example_id", row["example_id"], ids), ("seed_id", row["seed_id"], seeds), ("question", normalize(row["question_de"]), questions), ("gold_sql", normalize(row["gold_sql"]), sqls)):
            if value in seen:
                errors.append(f"line {line_number}: duplicate {label}")
            seen.add(value)
        if row["seed_id"] in eval_seeds:
            errors.append(f"line {line_number}: eval seed leakage")
        if normalize(row["question_de"]) in eval_questions:
            errors.append(f"line {line_number}: eval question leakage")
        if normalize(row["gold_sql"]) in eval_sql:
            errors.append(f"line {line_number}: eval SQL leakage")
        if DYNAMIC_TIME.search(row["gold_sql"]):
            errors.append(f"line {line_number}: dynamic time function is forbidden")
    if require_ready:
        counts = Counter(row.get("difficulty") for row in train)
        if len(train) != 1000 or counts["easy"] != 334 or counts["medium"] != 333 or counts["hard"] != 333:
            errors.append("ready dataset requires exactly 1,000 examples: 334 easy, 333 medium, 333 hard")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--eval", type=Path, default=Path("eval_v1.jsonl"))
    parser.add_argument("--known", type=Path, action="append", default=[], help="existing train batch; include it in duplicate checks")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    train = load_jsonl(args.data)
    for known_path in args.known:
        train = load_jsonl(known_path) + train
    errors = validate(train, load_jsonl(args.eval), args.require_ready)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {args.data} against {args.eval} without direct leakage.")


if __name__ == "__main__":
    main()