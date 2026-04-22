from pathlib import Path
from typing import Optional, Tuple


_LOCAL_MODEL_FILE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
}


def _is_explicit_local_reference(model_source: str) -> bool:
    normalized = model_source.strip().replace("\\", "/")
    if not normalized:
        return False

    if normalized.startswith(("./", "../", "~/", "/")):
        return True

    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        return True

    if "\\" in model_source:
        return True

    return Path(model_source).suffix.lower() in _LOCAL_MODEL_FILE_SUFFIXES


def resolve_pretrained_source(
    preferred_source: Optional[str],
    *,
    fallback_source: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve a model source with local-first behavior.

    Returns:
        Tuple[str, str]: (resolved_source, source_kind), where source_kind is
        either "local" or "remote".
    """

    if preferred_source:
        candidate = Path(preferred_source).expanduser()
        if candidate.exists():
            return str(candidate.resolve()), "local"

        if not _is_explicit_local_reference(preferred_source):
            return preferred_source, "remote"

    if fallback_source:
        fallback_candidate = Path(fallback_source).expanduser()
        if fallback_candidate.exists():
            return str(fallback_candidate.resolve()), "local"
        return fallback_source, "remote"

    if preferred_source:
        return preferred_source, "remote"

    raise ValueError("At least one model source must be provided.")
