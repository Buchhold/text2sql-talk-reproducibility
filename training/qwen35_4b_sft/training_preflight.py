"""GPU-free build-time checks for Qwen3.5 text-only QLoRA training."""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import subprocess
from pathlib import Path

EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.13.0",
    "bitsandbytes": "0.49.2",
    "datasets": "4.3.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "timm": "1.0.29",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
    "torchao": "0.16.0",
    "torchaudio": "2.10.0",
    "torchcodec": "0.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.5.0",
    "trl": "0.24.0",
    "unsloth": "2026.9.7",
    "unsloth-zoo": "2026.9.6",
    "xformers": "0.0.34",
}
ALLOWED_RUNTIME_DEPENDENCIES = {"libcuda.so.1"}
LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def unresolved_dependencies(ldd_output: str) -> set[str]:
    return {line.strip().split()[0] for line in ldd_output.splitlines() if "=> not found" in line}


def model_factory():
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        factory = getattr(transformers, name, None)
        if factory is not None:
            return name, factory
    raise RuntimeError("Transformers exposes no supported Qwen3.5 multimodal auto-model factory.")


def check_native_stack(expected_cuda: str) -> dict:
    import timm
    import torch
    torchao_output = io.StringIO()
    with contextlib.redirect_stdout(torchao_output), contextlib.redirect_stderr(torchao_output):
        import torchao
    if "Skipping import of cpp extensions" in torchao_output.getvalue():
        raise RuntimeError("TorchAO disabled its C++ extensions: " + torchao_output.getvalue().strip())
    import torchaudio
    import torchcodec
    import torchvision

    del timm, torchao, torchaudio, torchcodec, torchvision
    versions = {package: importlib.metadata.version(package) for package in EXPECTED_PACKAGE_VERSIONS}
    mismatches = {
        package: {"expected": expected, "installed": versions[package]}
        for package, expected in EXPECTED_PACKAGE_VERSIONS.items()
        if versions[package].split("+", 1)[0] != expected
    }
    if mismatches:
        raise RuntimeError(f"Package-family mismatch: {mismatches}")
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(f"PyTorch CUDA mismatch: expected {expected_cuda}, installed {torch.version.cuda}.")

    bnb_spec = importlib.util.find_spec("bitsandbytes")
    if bnb_spec is None or bnb_spec.origin is None:
        raise RuntimeError("bitsandbytes is not installed.")
    suffix = expected_cuda.replace(".", "")
    libraries = list(Path(bnb_spec.origin).parent.rglob(f"libbitsandbytes_cuda{suffix}.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one CUDA {suffix} bitsandbytes library, found {len(libraries)}.")
    ldd = subprocess.run(["ldd", str(libraries[0])], check=True, capture_output=True, text=True)
    missing = unresolved_dependencies(ldd.stdout)
    unexpected = missing - ALLOWED_RUNTIME_DEPENDENCIES
    if unexpected:
        raise RuntimeError(f"Unexpected unresolved native dependencies: {sorted(unexpected)}")
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": versions,
        "native_library": str(libraries[0]),
        "runtime_only_unresolved": sorted(missing),
    }


def check_training_import_contract(training_script: Path) -> dict:
    source = training_script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(training_script))
    main_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    if main_node is None:
        raise RuntimeError("Training script has no main() function.")
    first_import_lines: dict[str, int] = {}
    for node in ast.walk(main_node):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            first_import_lines[root] = min(first_import_lines.get(root, node.lineno), node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                first_import_lines[root] = min(first_import_lines.get(root, node.lineno), node.lineno)
    missing = {"unsloth", "transformers", "trl"} - set(first_import_lines)
    if missing:
        raise RuntimeError(f"Training imports are incomplete: {sorted(missing)}")
    if not first_import_lines["unsloth"] < first_import_lines["transformers"]:
        raise RuntimeError("Unsloth must be imported before Transformers.")
    if not first_import_lines["unsloth"] < first_import_lines["trl"]:
        raise RuntimeError("Unsloth must be imported before TRL.")
    if "training_args.eos_token != QWEN_EOS_TOKEN" not in source:
        raise RuntimeError("Training script lacks the runtime EOS replacement guard.")
    return {name: first_import_lines[name] for name in ("unsloth", "transformers", "trl")}


def check_training_api_contract(training_script: Path) -> dict:
    # Importing Unsloth itself requires a GPU in this release, so Cloud Build
    # validates its import order statically and validates unpatched TRL here.
    from trl import SFTConfig, SFTTrainer

    import_contract = check_training_import_contract(training_script)
    trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
    required_trainer_parameters = {"model", "train_dataset", "processing_class", "args"}
    missing_parameters = required_trainer_parameters - set(trainer_parameters)
    if missing_parameters:
        raise RuntimeError(f"TRL SFTTrainer API changed; missing: {sorted(missing_parameters)}")
    required_fields = {
        "dataset_text_field", "max_length", "max_steps", "num_train_epochs",
        "gradient_accumulation_steps", "optim", "bf16", "save_strategy",
        "warmup_steps", "eos_token",
    }
    missing_fields = required_fields - set(SFTConfig.__dataclass_fields__)
    if missing_fields:
        raise RuntimeError(f"TRL SFTConfig API changed; missing fields: {sorted(missing_fields)}")

    distribution = importlib.metadata.distribution("unsloth")
    installed_files = {str(path).replace("\\", "/") for path in (distribution.files or [])}
    required_files = {
        "unsloth/__init__.py",
        "unsloth/chat_templates.py",
        "unsloth/models/vision.py",
        "unsloth/trainer.py",
    }
    missing_files = required_files - installed_files
    if missing_files:
        raise RuntimeError(f"Unsloth installation is incomplete; missing: {sorted(missing_files)}")
    return {
        "training_import_lines": import_contract,
        "sft_trainer_parameters": sorted(required_trainer_parameters),
        "sft_config_fields": sorted(required_fields),
        "unsloth_files": sorted(required_files),
    }


def check_model_contract(model_id: str) -> dict:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor

    factory_name, factory = model_factory()
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token = "<|im_end|>"
    if eos_token not in tokenizer.get_vocab():
        raise RuntimeError(f"Qwen EOS token {eos_token!r} is absent from the tokenizer vocabulary.")
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_token_id is None or eos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Qwen EOS token {eos_token!r} could not be resolved to a token ID.")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Antworte ausschließlich mit SQL."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "SELECT 1;"}]},
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    markers = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    if not isinstance(rendered, str) or any(marker not in rendered for marker in markers):
        raise RuntimeError("Qwen3.5 chat template did not expose response-masking markers.")

    with init_empty_weights(), torch.device("meta"):
        model = factory.from_config(config)
    module_names = {name for name, _ in model.named_modules()}
    if not any("visual" in name for name in module_names):
        raise RuntimeError("Qwen3.5 visual modules were not detected.")
    targets = sorted(
        name for name in module_names
        if "language_model" in name and name.endswith(LORA_SUFFIXES)
    )
    if not targets:
        raise RuntimeError("No language-model LoRA targets were detected.")
    if any("visual" in name for name in targets):
        raise RuntimeError("Language-only LoRA targets leaked into visual modules.")
    return {
        "model_type": config.model_type,
        "model_factory": factory_name,
        "response_markers_checked": list(markers),
        "eos_token": eos_token,
        "eos_token_id": eos_token_id,
        "language_lora_target_count": len(targets),
        "language_lora_target_examples": targets[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda", default=os.environ.get("EXPECTED_CUDA", "13.0"))
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(json.dumps({"native_stack": check_native_stack(args.expected_cuda)}, indent=2), flush=True)
    print(json.dumps({"training_api": check_training_api_contract(Path("train.py"))}, indent=2), flush=True)
    print(json.dumps({"model_contract": check_model_contract(args.model_id)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
