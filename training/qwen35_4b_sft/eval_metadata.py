"""Runtime-configurable identifiers and conditions for Qwen offline evaluation."""
import os

DEFAULT_RUN_ID = "qwen35-4b-q4-vanilla-offline-eval"
DEFAULT_MODEL_NAME = "qwen35-4b-q4-vanilla"
SUPPORTED_CONDITIONS = ("P0", "P1", "P2")


def configured_metadata() -> tuple[str, str]:
    run_id = os.environ.get("EVAL_RUN_ID", DEFAULT_RUN_ID).strip()
    model_name = os.environ.get("EVAL_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    if not run_id or not model_name:
        raise ValueError("EVAL_RUN_ID and EVAL_MODEL_NAME must be non-empty.")
    return run_id, model_name


def selected_conditions() -> tuple[str, ...]:
    raw = os.environ.get("EVAL_CONDITIONS", ",".join(SUPPORTED_CONDITIONS))
    conditions = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not conditions or any(condition not in SUPPORTED_CONDITIONS for condition in conditions):
        raise ValueError("EVAL_CONDITIONS must be a non-empty comma-separated subset of P0, P1, P2.")
    if len(set(conditions)) != len(conditions):
        raise ValueError("EVAL_CONDITIONS must not contain duplicates.")
    return conditions


def batch_file_stem(conditions: tuple[str, ...]) -> str:
    return "_".join(condition.lower() for condition in conditions)