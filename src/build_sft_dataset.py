"""Build leakage-safe chat-format SFT records from reviewed ecom-v1 examples."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "data" / "evaluation" / "prompt_sql-only-schema-v1.txt"
P0_CONTEXT_PATH = ROOT / "data" / "evaluation" / "schema_context_p0_nodoc.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_prompt(question: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    context = P0_CONTEXT_PATH.read_text(encoding="utf-8")
    return template.replace("{{schema_context}}", context).replace("{{question_de}}", question)


def build_record(example: dict) -> dict:
    return {
        "example_id": example["example_id"],
        "seed_id": example["seed_id"],
        "schema": example["schema"],
        "difficulty": example["difficulty"],
        "messages": [
            {"role": "user", "content": render_prompt(example["question_de"])},
            {"role": "assistant", "content": example["gold_sql"]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [build_record(example) for example in load_jsonl(args.input)]
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {args.output}")
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"Wrote {len(records)} SFT records to {args.output}")


if __name__ == "__main__":
    main()