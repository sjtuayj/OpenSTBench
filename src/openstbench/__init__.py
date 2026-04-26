"""Top-level exports for the ``openstbench`` package.

This module keeps package import resilient to optional dependency failures.
Importing ``openstbench`` should succeed even when a specific evaluator's
extra dependencies are missing; the failure is raised only when that symbol is
actually accessed.
"""

from importlib import import_module
from typing import Dict, Tuple

__version__ = "0.3.1"

__all__ = [
    "TranslationEvaluator",
    "EmotionEvaluator",
    "ParalinguisticEvaluator",
    "BaseAudioEventPredictor",
    "BaseAudioEventLocalizer",
    "ClapAudioEventPredictor",
    "ClapSlidingWindowEventLocalizer",
    "EventAlignmentConfig",
    "EventLocalization",
    "EventLocalizationConfig",
    "EventPrediction",
    "EventPredictionConfig",
    "ParalinguisticSample",
    "load_paralinguistic_manifest",
    "load_paralinguistic_samples",
    "load_paralinguistic_audio_from_folder",
    "build_paralinguistic_inputs",
    "evaluate_paralinguistic_dataset",
    "SpeechQualityEvaluator",
    "TemporalConsistencyEvaluator",
    "LatencyEvaluator",
    "SpeakerSimilarityEvaluator",
    "GenericAgent",
    "AgentPipeline",
    "ReadAction",
    "WriteAction",
    "load_text_from_file_or_list",
    "load_audio_from_folder",
    "Dataset",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
    "create_dataset_from_json",
]


_EXPORT_SPECS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "translation_evaluator",
        (
            "TranslationEvaluator",
            "load_audio_from_folder",
            "load_text_from_file_or_list",
        ),
    ),
    ("emotion_evaluator", ("EmotionEvaluator",)),
    (
        "paralinguistic_evaluator",
        (
            "BaseAudioEventPredictor",
            "BaseAudioEventLocalizer",
            "ClapAudioEventPredictor",
            "ClapSlidingWindowEventLocalizer",
            "EventAlignmentConfig",
            "EventLocalization",
            "EventLocalizationConfig",
            "EventPrediction",
            "EventPredictionConfig",
            "ParalinguisticEvaluator",
            "ParalinguisticSample",
            "build_paralinguistic_inputs",
            "evaluate_paralinguistic_dataset",
            "load_audio_from_folder",
            "load_paralinguistic_manifest",
            "load_paralinguistic_samples",
        ),
    ),
    ("speech_quality_evaluator", ("SpeechQualityEvaluator",)),
    ("temporal_consistency_evaluator", ("TemporalConsistencyEvaluator",)),
    ("speaker_similarity_evaluator", ("SpeakerSimilarityEvaluator",)),
    (
        "dataset",
        (
            "Dataset",
            "create_dataset_from_json",
            "get_dataset_info",
            "list_datasets",
            "load_dataset",
        ),
    ),
    ("latency.agent", ("AgentPipeline", "GenericAgent")),
    ("latency.basics", ("ReadAction", "WriteAction")),
    ("latency.cli", ("LatencyEvaluator",)),
)


_IMPORT_ERRORS: Dict[str, Tuple[str, Exception]] = {}
_SYMBOL_TO_MODULE: Dict[str, str] = {}

for _module_name, _names in _EXPORT_SPECS:
    for _name in _names:
        if _module_name == "paralinguistic_evaluator" and _name == "load_audio_from_folder":
            _SYMBOL_TO_MODULE["load_paralinguistic_audio_from_folder"] = _module_name
        else:
            _SYMBOL_TO_MODULE[_name] = _module_name


def _load_module_exports(module_name: str) -> None:
    names = [name for name, owner in _SYMBOL_TO_MODULE.items() if owner == module_name]
    try:
        module = import_module(f".{module_name}", __name__)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        for name in names:
            _IMPORT_ERRORS[name] = (module_name, exc)
        return

    for name in names:
        if module_name == "paralinguistic_evaluator" and name == "load_paralinguistic_audio_from_folder":
            globals()[name] = getattr(module, "load_audio_from_folder")
        else:
            globals()[name] = getattr(module, name)
        _IMPORT_ERRORS.pop(name, None)


def __getattr__(name: str):
    if name in _SYMBOL_TO_MODULE:
        module_name = _SYMBOL_TO_MODULE[name]
        _load_module_exports(module_name)
        if name in globals():
            return globals()[name]
    if name in _IMPORT_ERRORS:
        module_name, exc = _IMPORT_ERRORS[name]
        raise ImportError(
            f"Cannot import '{name}' from 'openstbench' because "
            f"'openstbench.{module_name}' failed to load: {exc}"
        ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | set(__all__))
