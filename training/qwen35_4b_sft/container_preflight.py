"""Build-time checks for the Qwen3.5-4B text-only Q4 evaluation container."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path

EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.13.0",
    "bitsandbytes": "0.49.2",
    "huggingface-hub": "1.5.0",
    "pillow": "11.3.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.5.0",
}


def model_factory():
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        factory = getattr(transformers, name, None)
        if factory is not None:
            return name, factory
    raise RuntimeError("Transformers exposes neither AutoModelForMultimodalLM nor AutoModelForImageTextToText.")


def unresolved_dependencies(ldd_output: str) -> set[str]:
    return {line.strip().split()[0] for line in ldd_output.splitlines() if "=> not found" in line}


def check_native_stack(expected_cuda: str) -> dict:
    import torch
    import torchvision
    import bitsandbytes

    del torchvision
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
    unexpected = missing - {"libcuda.so.1"}
    if unexpected:
        raise RuntimeError(f"Unexpected unresolved native dependencies: {sorted(unexpected)}")
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "bitsandbytes": bitsandbytes.__version__, "native_library": str(libraries[0])}


def check_model_contract(model_id: str) -> dict:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor

    factory_name, factory = model_factory()
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Antworte ausschließlich mit SQL."}]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError("Qwen chat template did not render a non-empty text-only prompt.")
    with init_empty_weights(), torch.device("meta"):
        model = factory.from_config(config)
    modules = {name for name, _ in model.named_modules()}
    if not any("language_model" in name for name in modules):
        raise RuntimeError("Qwen3.5 model did not expose language-model modules.")
    return {
        "model_type": config.model_type,
        "model_factory": factory_name,
        "text_only_template_checked": True,
        "module_count": len(modules),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda", default=os.environ.get("EXPECTED_CUDA", "13.0"))
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(json.dumps({"native_stack": check_native_stack(args.expected_cuda)}, indent=2), flush=True)
    print(json.dumps({"model_contract": check_model_contract(args.model_id)}, indent=2), flush=True)


if __name__ == "__main__":
    main()