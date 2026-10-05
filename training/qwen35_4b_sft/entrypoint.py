"""Dispatch the shared Qwen3.5 image to evaluation or SFT training."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    mode = os.environ.get("TASK_MODE", "eval").strip().lower()
    scripts = {
        "eval": "offline_eval.py",
        "train": "train.py",
    }
    if mode not in scripts:
        raise ValueError(f"TASK_MODE must be one of {sorted(scripts)}, got {mode!r}.")
    script = Path(__file__).with_name(scripts[mode])
    print(f"Qwen3.5 container mode: {mode}", flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()