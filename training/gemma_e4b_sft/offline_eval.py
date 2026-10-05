"""Generate all P0/P1/P2 SQL answers using a saved Gemma 4 LoRA adapter."""
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from eval_metadata import batch_file_stem, configured_metadata, gemma_text_messages, selected_conditions

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/gemma-4-E4B-it")
BUNDLE_PATH = Path(os.environ["EVAL_BUNDLE_PATH"])
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip()
OUTPUT_URI = os.environ.get("AIP_MODEL_DIR", "/gcs/output/model")
OUTPUT_DIR = Path("/gcs/" + OUTPUT_URI.removeprefix("gs://") if OUTPUT_URI.startswith("gs://") else OUTPUT_URI)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1536"))
EVAL_RECORD_LIMIT = int(os.environ.get("EVAL_RECORD_LIMIT", "0"))
EVAL_MAX_RUNTIME_SECONDS = int(os.environ.get("EVAL_MAX_RUNTIME_SECONDS", "0"))
PROCESS_STARTED = time.monotonic()
STOP_REQUESTED = False


def request_stop(_signal_number, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def should_stop() -> bool:
    return STOP_REQUESTED or (
        EVAL_MAX_RUNTIME_SECONDS > 0
        and time.monotonic() - PROCESS_STARTED >= EVAL_MAX_RUNTIME_SECONDS
    )


def strip_sql_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def completed_example_keys(result_path: Path, conditions: tuple[str, ...]) -> set[tuple[str, str]]:
    completed = set()
    if not result_path.exists():
        return completed
    for line in result_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["condition"], record["example_id"])
        if key[0] not in conditions or key in completed:
            raise ValueError(f"Invalid existing evaluation record in {result_path}: {key}")
        completed.add(key)
    return completed


def append_record(result_path: Path, record: dict[str, object]) -> None:
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if not torch.cuda.is_available():
        raise RuntimeError("Offline evaluation requires a CUDA GPU.")
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    examples = bundle["examples"]
    prompt_template = bundle["prompt_template"]
    contexts = bundle["contexts"]
    conditions = selected_conditions()
    if set(contexts) != {"P0", "P1", "P2"} or not examples:
        raise ValueError("Expected a non-empty P0/P1/P2 eval bundle.")
    if EVAL_RECORD_LIMIT < 0 or EVAL_MAX_RUNTIME_SECONDS < 0:
        raise ValueError("EVAL_RECORD_LIMIT and EVAL_MAX_RUNTIME_SECONDS cannot be negative.")
    if ADAPTER_PATH and not Path(ADAPTER_PATH).is_dir():
        raise ValueError(f"Adapter path does not exist: {ADAPTER_PATH}")
    if EVAL_RECORD_LIMIT:
        examples = examples[:EVAL_RECORD_LIMIT]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = batch_file_stem(conditions)
    result_path = OUTPUT_DIR / f"offline_eval_results_{stem}.jsonl"
    manifest_path = OUTPUT_DIR / f"offline_eval_manifest_{stem}.json"
    completed = completed_example_keys(result_path, conditions)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quantization, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    if ADAPTER_PATH:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model_mode = "adapter"
    else:
        model = base
        model_mode = "base"
    model.eval()

    run_id, model_name = configured_metadata()
    total = len(conditions) * len(examples)
    stopped_early = False
    for condition in conditions:
        for example in examples:
            if should_stop():
                stopped_early = True
                print("Stopping before the Cloud Run deadline; completed records are resumable.", flush=True)
                break
            key = (condition, example["example_id"])
            if key in completed:
                continue
            prompt = prompt_template.replace("{{schema_context}}", contexts[condition]).replace(
                "{{question_de}}", example["question_de"]
            )
            inputs = processor.apply_chat_template(
                gemma_text_messages(prompt), tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt",
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True, pad_token_id=processor.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            generated = processor.decode(output[0][input_length:], skip_special_tokens=True)
            record = {
                "run_id": run_id,
                "eval_version": bundle["eval_version"],
                "schema": "ecom-v1",
                "split": example.get("split"),
                "difficulty": example.get("difficulty"),
                "example_id": example["example_id"],
                "model": model_name,
                "base_model": MODEL_ID,
                "prompt_version": bundle["prompt_version"],
                "schema_context_version": f"ecom-v1-{condition.lower()}",
                "condition": condition,
                "generated_sql": strip_sql_fences(generated),
                "latency_ms": round((time.monotonic() - started) * 1000),
                "max_new_tokens": MAX_NEW_TOKENS,
            }
            append_record(result_path, record)
            completed.add(key)
            print(f"[{condition}] {example['example_id']} ({len(completed)}/{total})", flush=True)
        if stopped_early:
            break

    manifest = {
        "run_id": run_id, "model": model_name,
        "records": len(completed), "conditions": list(conditions),
        "result_file": result_path.name,
        "base_model": MODEL_ID, "bundle_sha256": hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
        "adapter_path": ADAPTER_PATH or None, "model_mode": model_mode,
        "max_new_tokens": MAX_NEW_TOKENS, "record_limit": EVAL_RECORD_LIMIT,
        "max_runtime_seconds": EVAL_MAX_RUNTIME_SECONDS,
        "complete": not stopped_early and len(completed) == total,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()