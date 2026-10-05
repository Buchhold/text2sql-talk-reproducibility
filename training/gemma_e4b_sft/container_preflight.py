"""Build-time checks for the Gemma SFT container; no GPU or model weights needed."""
import argparse
import contextlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
from pathlib import Path

EXPECTED_LORA_PATTERN = (
    r"^model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
EXPECTED_MULTIMODAL_MODULES = (
    "model.audio_tower",
    "model.vision_tower",
    "model.embed_audio",
    "model.embed_vision",
)
ALLOWED_RUNTIME_DEPENDENCIES = {"libcuda.so.1"}
EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.13.0",
    "bitsandbytes": "0.49.2",
    "peft": "0.20.0",
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


def unresolved_dependencies(ldd_output: str) -> set[str]:
    return {
        line.strip().split()[0]
        for line in ldd_output.splitlines()
        if "=> not found" in line
    }


def check_native_stack(expected_cuda: str) -> dict:
    import timm
    import torch
    torchao_output = io.StringIO()
    with contextlib.redirect_stdout(torchao_output), contextlib.redirect_stderr(torchao_output):
        import torchao
    torchao_import_output = torchao_output.getvalue()
    if "Skipping import of cpp extensions" in torchao_import_output:
        raise RuntimeError(
            "TorchAO disabled its C++ extensions; Torch and TorchAO are incompatible: "
            + torchao_import_output.strip()
        )
    if torchao_import_output:
        print("TorchAO import output: " + torchao_import_output.strip(), flush=True)
    import torchaudio
    import torchcodec
    import torchvision

    # Imports themselves validate native ABI, image stack, and FFmpeg compatibility.
    del timm, torchao, torchaudio, torchcodec, torchvision
    package_versions = {
        package: importlib.metadata.version(package)
        for package in EXPECTED_PACKAGE_VERSIONS
    }
    mismatches = {
        package: {"expected": expected, "installed": package_versions[package]}
        for package, expected in EXPECTED_PACKAGE_VERSIONS.items()
        if package_versions[package].split("+", 1)[0] != expected
    }
    if mismatches:
        raise RuntimeError(f"PyTorch package-family mismatch: {mismatches}")
    actual_cuda = torch.version.cuda
    if actual_cuda != expected_cuda:
        raise RuntimeError(
            f"PyTorch CUDA mismatch: expected {expected_cuda}, installed {actual_cuda}."
        )
    bnb_spec = importlib.util.find_spec("bitsandbytes")
    if bnb_spec is None or bnb_spec.origin is None:
        raise RuntimeError("bitsandbytes is not installed.")
    cuda_suffix = expected_cuda.replace(".", "")
    bnb_dir = Path(bnb_spec.origin).parent
    candidates = list(bnb_dir.rglob(f"libbitsandbytes_cuda{cuda_suffix}.so"))
    if len(candidates) != 1:
        available = sorted(path.name for path in bnb_dir.rglob("libbitsandbytes_cuda*.so"))
        raise RuntimeError(
            f"Expected one bitsandbytes CUDA {cuda_suffix} library, found {len(candidates)}. "
            f"Available: {available}"
        )
    native_library = candidates[0]
    ldd = subprocess.run(
        ["ldd", str(native_library)], check=True, capture_output=True, text=True
    )
    missing = unresolved_dependencies(ldd.stdout)
    unexpected = missing - ALLOWED_RUNTIME_DEPENDENCIES
    if unexpected:
        raise RuntimeError(
            f"Unresolved native dependencies for {native_library.name}: {sorted(unexpected)}"
        )
    return {
        "python_cuda": actual_cuda,
        "torch_packages": package_versions,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "native_library": str(native_library),
        "runtime_only_unresolved": sorted(missing),
    }


def check_training_api_contract() -> dict:
    from trl import SFTConfig, SFTTrainer

    trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
    required_trainer_parameters = {"model", "train_dataset", "processing_class", "args"}
    missing_parameters = required_trainer_parameters - set(trainer_parameters)
    if missing_parameters:
        raise RuntimeError(
            f"TRL SFTTrainer API changed; missing parameters: {sorted(missing_parameters)}"
        )
    required_config_fields = {
        "dataset_text_field", "max_length", "max_steps", "num_train_epochs",
        "gradient_accumulation_steps", "optim", "bf16", "save_strategy",
    }
    missing_fields = required_config_fields - set(SFTConfig.__dataclass_fields__)
    if missing_fields:
        raise RuntimeError(
            f"TRL SFTConfig API changed; missing fields: {sorted(missing_fields)}"
        )

    unsloth_distribution = importlib.metadata.distribution("unsloth")
    installed_files = {str(path).replace("\\", "/") for path in (unsloth_distribution.files or [])}
    required_files = {"unsloth/__init__.py", "unsloth/chat_templates.py"}
    missing_files = required_files - installed_files
    if missing_files:
        raise RuntimeError(
            f"Unsloth installation is incomplete; missing files: {sorted(missing_files)}"
        )
    return {
        "sft_trainer_parameters": sorted(required_trainer_parameters),
        "sft_config_fields": sorted(required_config_fields),
        "unsloth_files": sorted(required_files),
    }


def check_model_structure(model_id: str) -> dict:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights(), torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    module_names = {name for name, _ in model.named_modules()}
    missing = set(EXPECTED_MULTIMODAL_MODULES) - module_names
    if missing:
        raise RuntimeError(f"Gemma module paths changed; missing: {sorted(missing)}")
    pattern = re.compile(EXPECTED_LORA_PATTERN)
    lora_targets = sorted(name for name in module_names if pattern.fullmatch(name))
    if not lora_targets:
        raise RuntimeError("The LoRA target regex matched no language-model modules.")
    leaked = [name for name in lora_targets if not name.startswith("model.language_model.")]
    if leaked:
        raise RuntimeError(f"The LoRA target regex leaked outside the language model: {leaked[:5]}")
    return {
        "model_type": config.model_type,
        "lora_target_count": len(lora_targets),
        "lora_target_examples": lora_targets[:3],
        "multimodal_modules_detected": list(EXPECTED_MULTIMODAL_MODULES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda", default=os.environ.get("EXPECTED_CUDA", "13.0"))
    parser.add_argument("--model-id")
    args = parser.parse_args()
    native_report = check_native_stack(args.expected_cuda)
    print(json.dumps({"native_stack": native_report}, indent=2), flush=True)
    api_report = check_training_api_contract()
    print(json.dumps({"training_api": api_report}, indent=2), flush=True)
    if args.model_id:
        model_report = check_model_structure(args.model_id)
        print(json.dumps({"model_structure": model_report}, indent=2), flush=True)


if __name__ == "__main__":
    main()