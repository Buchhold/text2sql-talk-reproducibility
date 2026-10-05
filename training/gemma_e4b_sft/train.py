"""Run a single, reproducible Gemma 4 E4B QLoRA SFT job."""
import hashlib
import json
import os
import random
from pathlib import Path

# An open Unsloth Gemma 4 issue reports silent zero-gradient training with
# sparse response-only labels on the fused-loss path. Force the verified
# standard-logits workaround before importing Unsloth.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/gemma-4-E4B-it")
DATA_PATH = Path(os.environ["TRAIN_DATA_PATH"])
OUTPUT_URI = os.environ.get("AIP_MODEL_DIR", "/gcs/output/model")
OUTPUT_DIR = Path("/gcs/" + OUTPUT_URI.removeprefix("gs://") if OUTPUT_URI.startswith("gs://") else OUTPUT_URI)
SEED = int(os.environ.get("SEED", "20260918"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
TRAIN_RECORD_LIMIT = int(os.environ.get("TRAIN_RECORD_LIMIT", "0"))


def load_messages(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(set(row) < {"example_id", "messages"} for row in rows):
        raise ValueError("TRAIN_DATA_PATH must contain non-empty chat-format JSONL records.")
    if any([message["role"] for message in row["messages"]] != ["user", "assistant"] for row in rows):
        raise ValueError("Every record must contain exactly user then assistant messages.")
    return rows


def validate_cuda_stack() -> None:
    import bitsandbytes as bnb

    expected_cuda = os.environ.get("EXPECTED_CUDA", "13.0")
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(
            f"PyTorch CUDA mismatch: expected {expected_cuda}, installed {torch.version.cuda}."
        )
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bitsandbytes": bnb.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
    }
    print("CUDA preflight: " + json.dumps(report), flush=True)
    probe = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    bnb.functional.quantize_4bit(probe, quant_type="nf4")
    del probe
    torch.cuda.empty_cache()
    print("CUDA preflight: bitsandbytes NF4 operation passed", flush=True)


def validate_runtime_options() -> None:
    if MAX_LENGTH <= 0 or MAX_STEPS == 0 or TRAIN_RECORD_LIMIT < 0:
        raise ValueError("MAX_LENGTH must be positive, MAX_STEPS cannot be zero, and TRAIN_RECORD_LIMIT cannot be negative.")


def main() -> None:
    validate_runtime_options()
    if not torch.cuda.is_available():
        raise RuntimeError("This training container requires a CUDA GPU.")
    validate_cuda_stack()
    torch._dynamo.config.recompile_limit = 64
    random.seed(SEED)
    set_seed(SEED)

    source_rows = load_messages(DATA_PATH)
    rows = source_rows[:TRAIN_RECORD_LIMIT] if TRAIN_RECORD_LIMIT else source_rows
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Training mode: records={len(rows)}/{len(source_rows)}, max_steps={MAX_STEPS}, max_length={MAX_LENGTH}",
        flush=True,
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        dtype=None,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
    )

    def render(row: dict) -> dict:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text.removeprefix("<bos>")}

    dataset = Dataset.from_list(rows).map(render, remove_columns=list(rows[0]))
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            dataset_text_field="text",
            max_length=MAX_LENGTH,
            num_train_epochs=3,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=1 if MAX_STEPS > 0 else 5,
            save_strategy="no" if MAX_STEPS > 0 else "epoch",
            save_total_limit=1,
            report_to="none",
            seed=SEED,
        ),
    )
    trainer = train_on_responses_only(trainer)
    first_labels = trainer.train_dataset[0]["labels"]
    supervised_tokens = sum(label != -100 for label in first_labels)
    if supervised_tokens == 0:
        raise RuntimeError("Response-only masking removed every target token from the first example.")
    print(f"Response-only masking: first example has {supervised_tokens} supervised tokens", flush=True)
    lora_b_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    ]
    if not lora_b_parameters:
        raise RuntimeError("No trainable LoRA-B parameter was found for the update probe.")
    update_probe_name, update_probe_parameter = lora_b_parameters[0]
    update_probe_before = update_probe_parameter.detach().float().cpu().clone()

    result = trainer.train()
    update_probe_delta = (
        update_probe_parameter.detach().float().cpu() - update_probe_before
    ).abs().max().item()
    if update_probe_delta == 0:
        raise RuntimeError(f"LoRA update probe did not change {update_probe_name}; training was ineffective.")
    print(
        f"LoRA update probe: {update_probe_name} max_abs_delta={update_probe_delta:.8g}",
        flush=True,
    )
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    manifest = {
        "base_model": MODEL_ID,
        "framework": "unsloth",
        "records": len(rows),
        "source_records": len(source_rows),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "lora_update_probe": {"parameter": update_probe_name, "max_abs_delta": update_probe_delta},
        "qlora": {"r": 32, "alpha": 64, "dropout": 0.0, "quantization": "nf4-4bit"},
        "train_metrics": result.metrics,
    }
    (OUTPUT_DIR / "sft_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()