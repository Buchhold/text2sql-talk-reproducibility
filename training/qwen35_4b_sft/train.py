"""Run reproducible text-only QLoRA SFT for Qwen3.5-4B."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

# Keep the verified standard-logits path used by the Gemma training container.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

QWEN_EOS_TOKEN = "<|im_end|>"
GRADIENT_ACCUMULATION_STEPS = 8


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    data_path: Path
    output_dir: Path
    seed: int
    max_length: int
    max_steps: int
    train_record_limit: int
    num_train_epochs: float


def output_path(uri: str) -> Path:
    return Path("/gcs/" + uri.removeprefix("gs://") if uri.startswith("gs://") else uri)


def configuration() -> TrainingConfig:
    return TrainingConfig(
        model_id=os.environ.get("MODEL_ID", "unsloth/Qwen3.5-4B"),
        data_path=Path(os.environ["TRAIN_DATA_PATH"]),
        output_dir=output_path(os.environ.get("AIP_MODEL_DIR", "/gcs/output/model")),
        seed=int(os.environ.get("SEED", "20260922")),
        max_length=int(os.environ.get("MAX_LENGTH", "2048")),
        max_steps=int(os.environ.get("MAX_STEPS", "-1")),
        train_record_limit=int(os.environ.get("TRAIN_RECORD_LIMIT", "0")),
        num_train_epochs=float(os.environ.get("NUM_TRAIN_EPOCHS", "3")),
    )


def load_messages(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(set(row) < {"example_id", "messages"} for row in rows):
        raise ValueError("TRAIN_DATA_PATH must contain non-empty chat-format JSONL records.")
    for row in rows:
        messages = row["messages"]
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError("Every record must contain exactly user then assistant messages.")
        if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
            raise ValueError("Every training message must contain non-empty string content.")
    return rows


def typed_messages(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


def render_conversation(processor, messages: list[dict]) -> str:
    text = processor.apply_chat_template(
        typed_messages(messages),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    required_markers = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    if not isinstance(text, str) or any(marker not in text for marker in required_markers):
        raise RuntimeError("Qwen3.5 chat template did not expose the expected user/assistant markers.")
    return text


def configure_qwen_tokenizer(tokenizer) -> int:
    """Select Qwen's real EOS token instead of Unsloth's template placeholder."""
    if QWEN_EOS_TOKEN not in tokenizer.get_vocab():
        raise RuntimeError(f"Qwen EOS token {QWEN_EOS_TOKEN!r} is absent from the tokenizer vocabulary.")
    tokenizer.eos_token = QWEN_EOS_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = QWEN_EOS_TOKEN
    eos_token_id = tokenizer.convert_tokens_to_ids(QWEN_EOS_TOKEN)
    if eos_token_id is None or eos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Qwen EOS token {QWEN_EOS_TOKEN!r} could not be resolved to a token ID.")
    return eos_token_id


def training_warmup_steps(record_count: int, max_steps: int, num_train_epochs: float) -> int:
    planned_steps = (
        max_steps
        if max_steps > 0
        else math.ceil(math.ceil(record_count / GRADIENT_ACCUMULATION_STEPS) * num_train_epochs)
    )
    # Tiny smoke/probe runs need their first optimizer step to update LoRA weights.
    return 0 if planned_steps < 20 else max(1, round(planned_steps * 0.03))


def validate_options(config: TrainingConfig) -> None:
    if config.max_length <= 0 or config.max_steps == 0:
        raise ValueError("MAX_LENGTH must be positive and MAX_STEPS cannot be zero.")
    if config.train_record_limit < 0 or config.num_train_epochs <= 0:
        raise ValueError("TRAIN_RECORD_LIMIT must be non-negative and NUM_TRAIN_EPOCHS positive.")


def validate_cuda_stack(torch) -> None:
    import bitsandbytes as bnb

    expected_cuda = os.environ.get("EXPECTED_CUDA", "13.0")
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(f"PyTorch CUDA mismatch: expected {expected_cuda}, installed {torch.version.cuda}.")
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


def main() -> None:
    config = configuration()
    validate_options(config)

    import torch
    # Unsloth must patch Transformers/TRL before either library is imported.
    from unsloth import FastVisionModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("This training container requires a CUDA GPU.")
    validate_cuda_stack(torch)
    torch._dynamo.config.recompile_limit = 64
    random.seed(config.seed)
    set_seed(config.seed)

    source_rows = load_messages(config.data_path)
    rows = source_rows[:config.train_record_limit] if config.train_record_limit else source_rows
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Training mode: "
        f"records={len(rows)}/{len(source_rows)}, max_steps={config.max_steps}, "
        f"epochs={config.num_train_epochs}, max_length={config.max_length}",
        flush=True,
    )

    model, processor = FastVisionModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_length,
        load_in_4bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    FastVisionModel.for_training(model)
    text_tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = configure_qwen_tokenizer(text_tokenizer)
    model.config.eos_token_id = eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = eos_token_id
    warmup_steps = training_warmup_steps(len(rows), config.max_steps, config.num_train_epochs)
    print(
        f"Tokenizer contract: eos_token={QWEN_EOS_TOKEN!r}, eos_token_id={eos_token_id}, "
        f"pad_token_id={text_tokenizer.pad_token_id}, warmup_steps={warmup_steps}",
        flush=True,
    )

    rendered_rows = [{"text": render_conversation(processor, row["messages"])} for row in rows]
    dataset = Dataset.from_list(rendered_rows)
    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        dataset_text_field="text",
        max_length=config.max_length,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        eos_token=QWEN_EOS_TOKEN,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1 if config.max_steps > 0 else 5,
        save_strategy="no" if config.max_steps > 0 else "epoch",
        save_total_limit=1,
        report_to="none",
        seed=config.seed,
    )
    if training_args.eos_token != QWEN_EOS_TOKEN:
        raise RuntimeError(
            f"Unsloth/TRL replaced eos_token with {training_args.eos_token!r}; import patching is broken."
        )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=text_tokenizer,
        args=training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
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
    print(f"LoRA update probe: {update_probe_name} max_abs_delta={update_probe_delta:.8g}", flush=True)

    model.save_pretrained(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))
    manifest = {
        "base_model": config.model_id,
        "framework": "unsloth-fast-vision-model-text-only",
        "records": len(rows),
        "source_records": len(source_rows),
        "dataset_sha256": hashlib.sha256(config.data_path.read_bytes()).hexdigest(),
        "seed": config.seed,
        "max_length": config.max_length,
        "max_steps": config.max_steps,
        "num_train_epochs": config.num_train_epochs,
        "vision_layers_trainable": False,
        "lora_update_probe": {"parameter": update_probe_name, "max_abs_delta": update_probe_delta},
        "qlora": {"r": 32, "alpha": 64, "dropout": 0.0, "quantization": "nf4-4bit"},
        "train_metrics": result.metrics,
    }
    (config.output_dir / "sft_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
