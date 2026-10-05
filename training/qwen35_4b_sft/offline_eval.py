"""Generate P0/P1/P2 SQL answers with Qwen3.5-4B in Q4, with optional thinking."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig

from eval_metadata import batch_file_stem, configured_metadata, selected_conditions

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")
BUNDLE_PATH = Path(os.environ["EVAL_BUNDLE_PATH"])
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip()
OUTPUT_URI = os.environ.get("AIP_MODEL_DIR", "/gcs/output/model")
OUTPUT_DIR = Path("/gcs/" + OUTPUT_URI.removeprefix("gs://") if OUTPUT_URI.startswith("gs://") else OUTPUT_URI)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1536"))
EVAL_RECORD_LIMIT = int(os.environ.get("EVAL_RECORD_LIMIT", "0"))
EVAL_RECORD_OFFSET = int(os.environ.get("EVAL_RECORD_OFFSET", "0"))
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "false").strip().lower() == "true"
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "1536"))
SEED = int(os.environ.get("SEED", "20260922"))
REQUIRE_ADAPTER = os.environ.get("REQUIRE_ADAPTER", "false").strip().lower() == "true"
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


def model_factory():
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        factory = getattr(transformers, name, None)
        if factory is not None:
            return factory
    raise RuntimeError("Transformers exposes no supported Qwen3.5 multimodal auto-model factory.")


def strip_sql_fences(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def completed_example_keys(result_path: Path, conditions: tuple[str, ...]) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not result_path.exists():
        return completed
    for line in result_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["condition"], record["example_id"])
        if key[0] not in conditions or key in completed:
            raise ValueError(f"Invalid existing evaluation record: {key}")
        completed.add(key)
    return completed


def last_subsequence_index(tokens: list[int], marker: list[int]) -> int:
    if not marker:
        raise ValueError("Marker tokens must not be empty.")
    for index in range(len(tokens) - len(marker), -1, -1):
        if tokens[index:index + len(marker)] == marker:
            return index
    return -1


def generate_sql(model, processor, inputs: dict[str, torch.Tensor], generation_kwargs: dict) -> tuple[str, int, int]:
    input_length = inputs["input_ids"].shape[-1]
    if not ENABLE_THINKING:
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        output_ids = generated_ids[0][input_length:].tolist()
        return processor.decode(output_ids, skip_special_tokens=True), 0, len(output_ids)

    generated_ids = model.generate(
        **inputs,
        **generation_kwargs,
        max_new_tokens=THINKING_BUDGET,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    output_ids = generated_ids[0][input_length:].tolist()
    eos_token_id = processor.tokenizer.eos_token_id
    think_end = processor.tokenizer.encode("</think>", add_special_tokens=False)
    if eos_token_id not in output_ids and last_subsequence_index(output_ids, think_end) < 0:
        early_stopping = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        early_ids = processor.tokenizer(
            early_stopping,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(model.device)
        continuation_ids = torch.cat([generated_ids, early_ids], dim=-1)
    else:
        continuation_ids = generated_ids
    remaining_tokens = MAX_NEW_TOKENS - (continuation_ids.shape[-1] - input_length)
    if eos_token_id not in output_ids and remaining_tokens <= 0:
        raise ValueError("MAX_NEW_TOKENS must exceed THINKING_BUDGET plus the early-stopping prompt.")
    if eos_token_id not in output_ids:
        generated_ids = model.generate(
            input_ids=continuation_ids,
            attention_mask=torch.ones_like(continuation_ids, dtype=torch.int64),
            **generation_kwargs,
            max_new_tokens=remaining_tokens,
            use_cache=True,
            pad_token_id=eos_token_id,
        )
        output_ids = generated_ids[0][input_length:].tolist()
    think_end_index = last_subsequence_index(output_ids, think_end)
    final_ids = output_ids[think_end_index + len(think_end):] if think_end_index >= 0 else output_ids
    return processor.decode(final_ids, skip_special_tokens=True), min(THINKING_BUDGET, len(output_ids)), len(output_ids)


def main() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if not torch.cuda.is_available():
        raise RuntimeError("Offline evaluation requires a CUDA GPU.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if MAX_NEW_TOKENS <= 0 or EVAL_RECORD_LIMIT < 0 or EVAL_RECORD_OFFSET < 0 or EVAL_MAX_RUNTIME_SECONDS < 0:
        raise ValueError("MAX_NEW_TOKENS must be positive and record limit/offset/runtime non-negative.")
    if ENABLE_THINKING and (THINKING_BUDGET < 1024 or THINKING_BUDGET >= MAX_NEW_TOKENS):
        raise ValueError("Thinking requires 1024 <= THINKING_BUDGET < MAX_NEW_TOKENS.")
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    examples = bundle["examples"]
    contexts = bundle["contexts"]
    prompt_template = bundle["prompt_template"]
    conditions = selected_conditions()
    if set(contexts) != {"P0", "P1", "P2"} or not examples:
        raise ValueError("Expected a non-empty P0/P1/P2 eval bundle.")
    examples = examples[EVAL_RECORD_OFFSET:]
    if EVAL_RECORD_LIMIT:
        examples = examples[:EVAL_RECORD_LIMIT]
    if not examples:
        raise ValueError("Selected evaluation shard contains no examples.")
    if REQUIRE_ADAPTER and not ADAPTER_PATH:
        raise ValueError("REQUIRE_ADAPTER=true but ADAPTER_PATH is empty.")
    if ADAPTER_PATH and not Path(ADAPTER_PATH).is_dir():
        raise ValueError(f"Adapter path does not exist: {ADAPTER_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / f"offline_eval_results_{batch_file_stem(conditions)}.jsonl"
    manifest_path = OUTPUT_DIR / f"offline_eval_manifest_{batch_file_stem(conditions)}.json"
    completed = completed_example_keys(result_path, conditions)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = model_factory().from_pretrained(MODEL_ID, quantization_config=quantization, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if ADAPTER_PATH:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model_mode = "adapter"
    else:
        model = base
        model_mode = "base"
    model.eval()
    generation_kwargs = (
        {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        if ENABLE_THINKING
        else {"do_sample": False}
    )
    run_id, model_name = configured_metadata()

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
            prompt = prompt_template.replace("{{schema_context}}", contexts[condition]).replace("{{question_de}}", example["question_de"])
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(model.device) for name, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            started = time.monotonic()
            with torch.inference_mode():
                generated, thinking_tokens, generated_tokens = generate_sql(model, processor, inputs, generation_kwargs)
            torch.cuda.synchronize()
            record = {
                "run_id": run_id, "eval_version": bundle["eval_version"], "schema": "ecom-v1",
                "split": example.get("split"), "difficulty": example.get("difficulty"), "example_id": example["example_id"],
                "model": model_name, "base_model": MODEL_ID, "adapter_path": ADAPTER_PATH or None, "model_mode": model_mode, "prompt_version": bundle["prompt_version"],
                "schema_context_version": f"ecom-v1-{condition.lower()}", "condition": condition,
                "generated_sql": strip_sql_fences(generated), "latency_ms": round((time.monotonic() - started) * 1000),
                "max_new_tokens": MAX_NEW_TOKENS, "quantization": "nf4-4bit", "thinking": ENABLE_THINKING,
                "thinking_budget": THINKING_BUDGET if ENABLE_THINKING else 0,
                "thinking_tokens": thinking_tokens, "generated_tokens": generated_tokens,
            }
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            completed.add(key)
            print(f"[{condition}] {example['example_id']} ({len(completed)}/{len(conditions) * len(examples)})", flush=True)
        if stopped_early:
            break
    manifest = {
        "run_id": run_id, "model": model_name, "base_model": MODEL_ID, "records": len(completed),
        "conditions": list(conditions), "result_file": result_path.name, "bundle_sha256": hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
        "adapter_path": ADAPTER_PATH or None, "model_mode": model_mode,
        "adapter_required": REQUIRE_ADAPTER,
        "max_new_tokens": MAX_NEW_TOKENS, "record_limit": EVAL_RECORD_LIMIT, "record_offset": EVAL_RECORD_OFFSET,
        "quantization": "nf4-4bit", "thinking": ENABLE_THINKING,
        "thinking_budget": THINKING_BUDGET if ENABLE_THINKING else 0,
        "max_runtime_seconds": EVAL_MAX_RUNTIME_SECONDS,
        "complete": not stopped_early and len(completed) == len(conditions) * len(examples),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
