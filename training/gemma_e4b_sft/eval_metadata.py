"""Runtime-configurable identifiers for offline evaluation outputs."""
import os

DEFAULT_RUN_ID = "gemma-4-e4b-sft-v3-offline-eval"
DEFAULT_MODEL_NAME = "gemma-4-e4b-sft-v3"


def configured_metadata() -> tuple[str, str]:
    run_id = os.environ.get("EVAL_RUN_ID", DEFAULT_RUN_ID).strip()
    model_name = os.environ.get("EVAL_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    if not run_id or not model_name:
        raise ValueError("EVAL_RUN_ID and EVAL_MODEL_NAME must be non-empty.")
    return run_id, model_name


SUPPORTED_CONDITIONS = ("P0", "P1", "P2")


def gemma_text_messages(prompt: str) -> list[dict[str, object]]:
    """Return Gemma 4's multimodal chat-template representation for plain text."""
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


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