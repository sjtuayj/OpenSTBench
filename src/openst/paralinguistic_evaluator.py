import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


LabelNormalizer = Optional[Union[Dict[str, Optional[str]], Callable[[str], Optional[str]]]]


@dataclass(frozen=True)
class ParalinguisticSample:
    sample_id: str
    source_audio: str
    source_text: str = ""
    source_label: Optional[str] = None
    source_onset_ms: Optional[float] = None
    source_offset_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventPrediction:
    label: Optional[str]
    score: Optional[float] = None
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": None if self.score is None else float(self.score),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
        }


@dataclass(frozen=True)
class EventPredictionConfig:
    score_threshold: float = 0.2
    fallback_top1: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_threshold": float(self.score_threshold),
            "fallback_top1": bool(self.fallback_top1),
        }


@dataclass(frozen=True)
class EventLocalization:
    label: Optional[str]
    onset_ms: Optional[float] = None
    offset_ms: Optional[float] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "onset_ms": None if self.onset_ms is None else float(self.onset_ms),
            "offset_ms": None if self.offset_ms is None else float(self.offset_ms),
            "score": None if self.score is None else float(self.score),
        }


@dataclass(frozen=True)
class EventLocalizationConfig:
    window_ms: float = 320.0
    hop_ms: float = 40.0
    score_threshold: Optional[float] = None
    fallback_top1: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_ms": float(self.window_ms),
            "hop_ms": float(self.hop_ms),
            "score_threshold": None if self.score_threshold is None else float(self.score_threshold),
            "fallback_top1": None if self.fallback_top1 is None else bool(self.fallback_top1),
        }


@dataclass(frozen=True)
class EventAlignmentConfig:
    relative_onset_tolerance: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_onset_tolerance": float(self.relative_onset_tolerance),
        }


class BaseAudioEventPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[EventPrediction]:
        raise NotImplementedError


class BaseAudioEventLocalizer(ABC):
    @abstractmethod
    def localize(
        self,
        audio_paths: Sequence[str],
        labels: Sequence[Optional[str]],
        candidate_labels: Sequence[str],
    ) -> List[EventLocalization]:
        raise NotImplementedError


def ensure_existing_audio(path: str) -> str:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")
    return str(audio_path.resolve())


def _to_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_audio_mono(audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    waveform = waveform.squeeze(0).contiguous().float().cpu().numpy()
    return waveform, int(sr)


def _get_audio_duration_ms(audio_path: str) -> float:
    try:
        info = torchaudio.info(str(audio_path))
        if info.sample_rate > 0:
            return float(info.num_frames / info.sample_rate * 1000.0)
    except Exception:
        pass

    waveform, sr = _load_audio_mono(audio_path)
    if sr <= 0:
        return 0.0
    return float(len(waveform) / sr * 1000.0)


def _normalize_text_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def _apply_label_normalizer(label: Optional[str], label_normalizer: LabelNormalizer) -> Optional[str]:
    if label is None:
        return None

    normalized = _normalize_text_label(label)
    if not normalized:
        return None

    if label_normalizer is None:
        mapped = normalized
    elif callable(label_normalizer):
        mapped = label_normalizer(normalized)
    else:
        mapped = label_normalizer.get(normalized, normalized)

    if mapped is None:
        return None
    mapped_text = _normalize_text_label(str(mapped))
    return mapped_text or None


def _coerce_manifest_source_label(raw_label: Any, index: int) -> Optional[str]:
    if raw_label is None:
        return None

    if isinstance(raw_label, list):
        cleaned = []
        for item in raw_label:
            label = _normalize_text_label(str(item))
            if label and label not in cleaned:
                cleaned.append(label)
        if not cleaned:
            return None
        if len(cleaned) > 1:
            raise ValueError(
                f"Manifest item {index} has multiple source labels {cleaned}. "
                "This evaluator expects at most one source event label per sample."
            )
        return cleaned[0]

    label = _normalize_text_label(str(raw_label))
    return label or None


def _coerce_optional_float(raw_value: Any, *, name: str, index: int) -> Optional[float]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{name} for manifest item {index} must be numeric or null.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} for manifest item {index} must be numeric or null.") from exc
    if value < 0.0:
        raise ValueError(f"{name} for manifest item {index} must be non-negative.")
    return value


def _normalize_label_batch(
    labels: Optional[Sequence[Optional[str]]],
    *,
    name: str,
    expected_length: int,
    label_normalizer: LabelNormalizer,
) -> Optional[List[Optional[str]]]:
    if labels is None:
        return None
    if len(labels) != expected_length:
        raise ValueError(f"{name} size mismatch: {len(labels)} vs {expected_length}")

    normalized: List[Optional[str]] = []
    for index, label in enumerate(labels):
        if label is None:
            normalized.append(None)
            continue
        if not isinstance(label, str):
            raise ValueError(f"{name}[{index}] must be a string or None.")
        normalized.append(_apply_label_normalizer(label, label_normalizer))
    return normalized


def _normalize_float_batch(
    values: Optional[Sequence[Optional[Union[int, float]]]],
    *,
    name: str,
    expected_length: int,
) -> Optional[List[Optional[float]]]:
    if values is None:
        return None
    if len(values) != expected_length:
        raise ValueError(f"{name} size mismatch: {len(values)} vs {expected_length}")

    normalized: List[Optional[float]] = []
    for index, value in enumerate(values):
        if value is None:
            normalized.append(None)
            continue
        if isinstance(value, bool):
            raise ValueError(f"{name}[{index}] must be numeric or None.")
        try:
            float_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be numeric or None.") from exc
        if float_value < 0.0:
            raise ValueError(f"{name}[{index}] must be non-negative.")
        normalized.append(float_value)
    return normalized


def _normalize_candidate_labels(
    candidate_labels: Optional[Sequence[str]],
    *,
    label_normalizer: LabelNormalizer,
) -> List[str]:
    if candidate_labels is None:
        return []

    normalized: List[str] = []
    seen = set()
    for label in candidate_labels:
        mapped = _apply_label_normalizer(label, label_normalizer)
        if mapped is None or mapped in seen:
            continue
        normalized.append(mapped)
        seen.add(mapped)
    return normalized


def _resolve_candidate_labels(
    *,
    candidate_labels: Optional[Sequence[str]],
    source_labels: Optional[Sequence[Optional[str]]],
    target_labels: Optional[Sequence[Optional[str]]],
    label_normalizer: LabelNormalizer,
) -> List[str]:
    resolved = _normalize_candidate_labels(candidate_labels, label_normalizer=label_normalizer)
    if resolved:
        return resolved

    seen = set()
    derived: List[str] = []
    for batch in (source_labels, target_labels):
        if batch is None:
            continue
        for label in batch:
            mapped = _apply_label_normalizer(label, label_normalizer)
            if mapped is None or mapped in seen:
                continue
            derived.append(mapped)
            seen.add(mapped)
    return derived


def _load_data_list(data: Union[List[str], str], name: str) -> List[str]:
    if isinstance(data, str):
        path = Path(data)
        if path.exists() and path.is_dir():
            return load_audio_from_folder(str(path))
        return [ensure_existing_audio(data)]
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a path or a list of paths.")
    return [ensure_existing_audio(str(item)) for item in data]


def load_audio_from_folder(folder_path: str) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Expected a directory, got: {folder_path}")

    audio_files: List[Path] = []
    for extension in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        audio_files.extend(folder.glob(f"*{extension}"))
    audio_files = sorted(audio_files, key=lambda item: item.stem)
    if not audio_files:
        raise ValueError(f"No audio files found under: {folder_path}")
    return [str(item.resolve()) for item in audio_files]


def _safe_mean(values: Sequence[float]) -> float:
    return round(float(sum(values) / len(values)), 4) if values else 0.0


def _compute_single_label_metrics(
    reference_labels: Sequence[Optional[str]],
    predicted_labels: Sequence[Optional[str]],
    *,
    class_labels: Sequence[str],
) -> Dict[str, Any]:
    evaluated_indices = [index for index, label in enumerate(reference_labels) if label is not None]
    evaluated_reference = [reference_labels[index] for index in evaluated_indices]
    evaluated_prediction = [predicted_labels[index] for index in evaluated_indices]

    total = len(evaluated_reference)
    correct = sum(
        1
        for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
        if reference_label == predicted_label
    )
    abstained = sum(1 for predicted_label in evaluated_prediction if predicted_label is None)

    unique_class_labels = list(dict.fromkeys([label for label in class_labels if label]))
    if not unique_class_labels:
        unique_class_labels = sorted(
            {
                label
                for label in evaluated_reference + evaluated_prediction
                if label is not None
            }
        )

    per_label: Dict[str, Dict[str, float]] = {}
    macro_f1_values: List[float] = []
    macro_recall_values: List[float] = []
    confusion: Dict[str, Dict[str, int]] = {}

    for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction):
        predicted_key = predicted_label if predicted_label is not None else "__none__"
        confusion.setdefault(str(reference_label), {})
        confusion[str(reference_label)][predicted_key] = confusion[str(reference_label)].get(predicted_key, 0) + 1

    for label in unique_class_labels:
        tp = sum(
            1
            for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
            if reference_label == label and predicted_label == label
        )
        fp = sum(
            1
            for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
            if reference_label != label and predicted_label == label
        )
        fn = sum(
            1
            for reference_label, predicted_label in zip(evaluated_reference, evaluated_prediction)
            if reference_label == label and predicted_label != label
        )
        support = sum(1 for reference_label in evaluated_reference if reference_label == label)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        per_label[label] = {
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        macro_f1_values.append(f1)
        macro_recall_values.append(recall)

    return {
        "num_evaluated": int(total),
        "num_skipped": int(len(reference_labels) - total),
        "num_correct": int(correct),
        "num_abstained": int(abstained),
        "preservation_rate": round(float(correct / total), 4) if total > 0 else 0.0,
        "macro_f1": _safe_mean(macro_f1_values),
        "macro_recall": _safe_mean(macro_recall_values),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def _compute_alignment_metrics(
    reference_labels: Sequence[Optional[str]],
    predicted_labels: Sequence[Optional[str]],
    source_onsets_ms: Sequence[Optional[float]],
    target_onsets_ms: Sequence[Optional[float]],
    source_durations_ms: Sequence[float],
    target_durations_ms: Sequence[float],
    *,
    relative_onset_tolerance: float,
    sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    evaluated_indices = [
        index
        for index, label in enumerate(reference_labels)
        if label is not None and source_onsets_ms[index] is not None and source_durations_ms[index] > 0.0
    ]

    aligned_success = 0
    missing_target_onset = 0
    conditional_errors: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    for index in evaluated_indices:
        reference_label = reference_labels[index]
        predicted_label = predicted_labels[index]
        source_onset_ms = source_onsets_ms[index]
        target_onset_ms = target_onsets_ms[index]
        source_duration_ms = float(source_durations_ms[index])
        target_duration_ms = float(target_durations_ms[index])

        assert reference_label is not None
        assert source_onset_ms is not None

        source_relative_onset = float(source_onset_ms / source_duration_ms) if source_duration_ms > 0.0 else None
        target_relative_onset = (
            float(target_onset_ms / target_duration_ms)
            if target_onset_ms is not None and target_duration_ms > 0.0
            else None
        )
        label_correct = predicted_label is not None and predicted_label == reference_label

        relative_onset_error: Optional[float] = None
        aligned = False
        if label_correct and target_relative_onset is not None and source_relative_onset is not None:
            relative_onset_error = abs(source_relative_onset - target_relative_onset)
            conditional_errors.append(relative_onset_error)
            aligned = relative_onset_error <= relative_onset_tolerance
        elif label_correct and target_onset_ms is None:
            missing_target_onset += 1

        if aligned:
            aligned_success += 1

        sample_records.append(
            {
                "sample_index": int(index),
                "sample_id": sample_ids[index] if sample_ids is not None else str(index),
                "reference_label": reference_label,
                "predicted_label": predicted_label,
                "source_onset_ms": float(source_onset_ms),
                "target_onset_ms": None if target_onset_ms is None else float(target_onset_ms),
                "source_duration_ms": round(source_duration_ms, 4),
                "target_duration_ms": round(target_duration_ms, 4),
                "source_relative_onset": None if source_relative_onset is None else round(source_relative_onset, 4),
                "target_relative_onset": None if target_relative_onset is None else round(target_relative_onset, 4),
                "relative_onset_error": None if relative_onset_error is None else round(relative_onset_error, 4),
                "label_correct": bool(label_correct),
                "aligned": bool(aligned),
            }
        )

    total = len(evaluated_indices)
    return {
        "num_evaluated": int(total),
        "num_skipped": int(len(reference_labels) - total),
        "num_aligned": int(aligned_success),
        "num_missing_target_onset": int(missing_target_onset),
        "num_conditionally_evaluated": int(len(conditional_errors)),
        "aligned_preservation_rate": round(float(aligned_success / total), 4) if total > 0 else 0.0,
        "conditional_relative_onset_error": _safe_mean(conditional_errors),
        "relative_onset_tolerance": round(float(relative_onset_tolerance), 4),
        "samples": sample_records,
    }


class ClapAudioEventPredictor(BaseAudioEventPredictor):
    PROMPT_TEMPLATES = (
        "{label}",
        "the sound of {label}",
        "an audio recording of {label}",
        "a person producing {label}",
    )

    def __init__(
        self,
        *,
        model_path: str = "laion/clap-htsat-fused",
        config: Optional[EventPredictionConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or EventPredictionConfig()
        self.device = _to_device(device)
        self._processor = None
        self._model = None
        self._text_embedding_cache: Dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("CLAP-based event prediction requires `transformers`.") from exc

        self._processor = ClapProcessor.from_pretrained(self.model_path)
        self._model = ClapModel.from_pretrained(self.model_path).to(self.device).eval()

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            return embedding
        return embedding / norm

    def _build_prompts(self, candidate_labels: Sequence[str]) -> List[Tuple[str, str]]:
        prompt_records: List[Tuple[str, str]] = []
        for label in candidate_labels:
            for template in self.PROMPT_TEMPLATES:
                prompt_records.append((label, template.format(label=label)))
        return prompt_records

    def _extract_audio_embeddings_from_waveforms(
        self,
        waveforms: Sequence[np.ndarray],
        *,
        sampling_rate: int,
    ) -> List[np.ndarray]:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        embeddings: List[np.ndarray] = []
        for waveform in waveforms:
            inputs = self._processor(audio=waveform, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._model.get_audio_features(**inputs)
            embeddings.append(features[0].detach().cpu().numpy())
        return embeddings

    def _extract_audio_embeddings(self, audio_paths: Sequence[str]) -> List[np.ndarray]:
        embeddings: List[np.ndarray] = []
        for audio_path in tqdm(audio_paths, desc="Extracting CLAP event embeddings", unit="file"):
            waveform, sr = _load_audio_mono(str(audio_path), target_sr=48000)
            embeddings.extend(self._extract_audio_embeddings_from_waveforms([waveform], sampling_rate=sr))
        return embeddings

    def _extract_text_embeddings(self, texts: Sequence[str]) -> List[np.ndarray]:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        embeddings: List[np.ndarray] = []
        for text in texts:
            cached = self._text_embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
                continue
            inputs = self._processor(text=[str(text)], return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
            embedding = features[0].detach().cpu().numpy()
            self._text_embedding_cache[text] = embedding
            embeddings.append(embedding)
        return embeddings

    def _score_audio_embeddings(
        self,
        audio_embeddings: Sequence[np.ndarray],
        candidate_labels: Sequence[str],
    ) -> List[Dict[str, float]]:
        normalized_candidate_labels = [
            _normalize_text_label(label)
            for label in candidate_labels
            if _normalize_text_label(label)
        ]
        unique_candidate_labels = list(dict.fromkeys(normalized_candidate_labels))
        if not unique_candidate_labels:
            return [{} for _ in audio_embeddings]

        prompt_records = self._build_prompts(unique_candidate_labels)
        normalized_audio_embeddings = [self._normalize_embedding(item) for item in audio_embeddings]
        normalized_text_embeddings = [
            self._normalize_embedding(item)
            for item in self._extract_text_embeddings([prompt for _, prompt in prompt_records])
        ]

        scores_per_audio: List[Dict[str, float]] = []
        for audio_embedding in normalized_audio_embeddings:
            label_scores: Dict[str, float] = {}
            for (label, _prompt), text_embedding in zip(prompt_records, normalized_text_embeddings):
                score = float(np.dot(audio_embedding, text_embedding))
                if label not in label_scores or score > label_scores[label]:
                    label_scores[label] = score
            scores_per_audio.append(label_scores)
        return scores_per_audio

    def score_audio_paths(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[Dict[str, float]]:
        audio_embeddings = self._extract_audio_embeddings(audio_paths)
        return self._score_audio_embeddings(audio_embeddings, candidate_labels)

    def score_waveforms(
        self,
        waveforms: Sequence[np.ndarray],
        *,
        sampling_rate: int,
        candidate_labels: Sequence[str],
    ) -> List[Dict[str, float]]:
        audio_embeddings = self._extract_audio_embeddings_from_waveforms(waveforms, sampling_rate=sampling_rate)
        return self._score_audio_embeddings(audio_embeddings, candidate_labels)

    def predict(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[EventPrediction]:
        label_scores_per_audio = self.score_audio_paths(audio_paths, candidate_labels)

        predictions: List[EventPrediction] = []
        threshold = float(self.config.score_threshold)
        fallback_top1 = bool(self.config.fallback_top1)

        for label_scores in label_scores_per_audio:
            if not label_scores:
                predictions.append(EventPrediction(label=None, score=None, scores={}))
                continue

            top_label = max(label_scores.items(), key=lambda item: item[1])[0]
            top_score = float(label_scores[top_label])
            predicted_label = top_label if top_score >= threshold or fallback_top1 else None

            predictions.append(
                EventPrediction(
                    label=predicted_label,
                    score=top_score,
                    scores={label: round(score, 4) for label, score in sorted(label_scores.items())},
                )
            )

        return predictions


class ClapSlidingWindowEventLocalizer(BaseAudioEventLocalizer):
    def __init__(
        self,
        *,
        model_path: str = "laion/clap-htsat-fused",
        prediction_config: Optional[EventPredictionConfig] = None,
        localization_config: Optional[EventLocalizationConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.predictor = ClapAudioEventPredictor(
            model_path=model_path,
            config=prediction_config,
            device=device,
        )
        self.prediction_config = prediction_config or EventPredictionConfig()
        self.localization_config = localization_config or EventLocalizationConfig()

    def _build_windows(self, waveform: np.ndarray, sampling_rate: int) -> Tuple[List[np.ndarray], List[int]]:
        window_samples = max(1, int(round(self.localization_config.window_ms / 1000.0 * sampling_rate)))
        hop_samples = max(1, int(round(self.localization_config.hop_ms / 1000.0 * sampling_rate)))
        total_samples = int(len(waveform))

        if total_samples <= window_samples:
            return [waveform], [0]

        starts: List[int] = list(range(0, max(total_samples - window_samples + 1, 1), hop_samples))
        last_start = max(total_samples - window_samples, 0)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        segments = [waveform[start : start + window_samples] for start in starts]
        return segments, starts

    def localize(
        self,
        audio_paths: Sequence[str],
        labels: Sequence[Optional[str]],
        candidate_labels: Sequence[str],
    ) -> List[EventLocalization]:
        if len(audio_paths) != len(labels):
            raise ValueError(f"audio_paths size mismatch: {len(audio_paths)} vs labels {len(labels)}")

        normalized_candidate_labels = {
            _normalize_text_label(label)
            for label in candidate_labels
            if _normalize_text_label(label)
        }
        threshold = (
            float(self.localization_config.score_threshold)
            if self.localization_config.score_threshold is not None
            else float(self.prediction_config.score_threshold)
        )
        fallback_top1 = (
            bool(self.localization_config.fallback_top1)
            if self.localization_config.fallback_top1 is not None
            else bool(self.prediction_config.fallback_top1)
        )

        localizations: List[EventLocalization] = []
        for audio_path, label in tqdm(
            list(zip(audio_paths, labels)),
            desc="Localizing acoustic events",
            unit="file",
        ):
            normalized_label = _normalize_text_label(label) if isinstance(label, str) and label else None
            if normalized_label is None or (
                normalized_candidate_labels and normalized_label not in normalized_candidate_labels
            ):
                localizations.append(EventLocalization(label=normalized_label))
                continue

            waveform, sampling_rate = _load_audio_mono(str(audio_path), target_sr=48000)
            if waveform.size == 0:
                localizations.append(EventLocalization(label=normalized_label))
                continue

            segments, starts = self._build_windows(waveform, sampling_rate)
            label_scores_per_window = self.predictor.score_waveforms(
                segments,
                sampling_rate=sampling_rate,
                candidate_labels=[normalized_label],
            )
            best_start = 0
            best_score = None
            best_index = -1
            for index, label_scores in enumerate(label_scores_per_window):
                score = label_scores.get(normalized_label)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = float(score)
                    best_start = starts[index]
                    best_index = index

            if best_score is None or (best_score < threshold and not fallback_top1):
                localizations.append(EventLocalization(label=normalized_label, score=best_score))
                continue

            window_samples = len(segments[best_index]) if best_index >= 0 else 0
            onset_ms = float(best_start / sampling_rate * 1000.0)
            offset_ms = float(min(best_start + window_samples, len(waveform)) / sampling_rate * 1000.0)
            localizations.append(
                EventLocalization(
                    label=normalized_label,
                    onset_ms=onset_ms,
                    offset_ms=offset_ms,
                    score=best_score,
                )
            )

        return localizations


class ParalinguisticEvaluator:
    DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"

    def __init__(
        self,
        use_continuous_fidelity: bool = True,
        use_event_preservation: bool = True,
        use_event_alignment: bool = True,
        clap_model_path: Optional[str] = None,
        event_predictor: Optional[BaseAudioEventPredictor] = None,
        event_localizer: Optional[BaseAudioEventLocalizer] = None,
        event_prediction_config: Optional[Dict[str, Any]] = None,
        event_localization_config: Optional[Dict[str, Any]] = None,
        event_alignment_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.device = _to_device(device)
        self.use_continuous_fidelity = bool(use_continuous_fidelity)
        self.use_event_preservation = bool(use_event_preservation)
        self.use_event_alignment = bool(use_event_alignment)
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL
        self.event_prediction_config = EventPredictionConfig(**(event_prediction_config or {}))
        self.event_localization_config = EventLocalizationConfig(**(event_localization_config or {}))
        self.event_alignment_config = EventAlignmentConfig(**(event_alignment_config or {}))
        self.event_predictor = event_predictor
        self.event_localizer = event_localizer

        self._clap_processor = None
        self._clap_model = None
        self._default_predictor: Optional[ClapAudioEventPredictor] = None
        self._default_localizer: Optional[ClapSlidingWindowEventLocalizer] = None

    def _load_clap(self) -> None:
        if self._clap_model is not None and self._clap_processor is not None:
            return

        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("Paralinguistic_Fidelity_Cosine requires `transformers` to load CLAP.") from exc

        self._clap_processor = ClapProcessor.from_pretrained(self.clap_model_path)
        self._clap_model = ClapModel.from_pretrained(self.clap_model_path).to(self.device).eval()

    def _extract_clap_embeddings(self, audio_paths: Sequence[str]) -> List[np.ndarray]:
        self._load_clap()
        assert self._clap_processor is not None
        assert self._clap_model is not None

        embeddings: List[np.ndarray] = []
        for audio_path in tqdm(audio_paths, desc="Extracting CLAP embeddings", unit="file"):
            waveform, sr = _load_audio_mono(str(audio_path), target_sr=48000)
            inputs = self._clap_processor(audio=waveform, sampling_rate=sr, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._clap_model.get_audio_features(**inputs)
            embeddings.append(features[0].detach().cpu().numpy())
        return embeddings

    @staticmethod
    def _average_cosine(source_embeddings: Sequence[np.ndarray], target_embeddings: Sequence[np.ndarray]) -> float:
        total = 0.0
        count = 0
        for source_embedding, target_embedding in zip(source_embeddings, target_embeddings):
            source_norm = float(np.linalg.norm(source_embedding))
            target_norm = float(np.linalg.norm(target_embedding))
            if source_norm <= 0.0 or target_norm <= 0.0:
                continue
            total += float(np.dot(source_embedding, target_embedding) / (source_norm * target_norm))
            count += 1
        return round(total / count, 4) if count > 0 else 0.0

    def _get_event_predictor(self) -> BaseAudioEventPredictor:
        if self.event_predictor is not None:
            return self.event_predictor
        if self._default_predictor is None:
            self._default_predictor = ClapAudioEventPredictor(
                model_path=self.clap_model_path,
                config=self.event_prediction_config,
                device=self.device,
            )
        return self._default_predictor

    def _get_event_localizer(self) -> BaseAudioEventLocalizer:
        if self.event_localizer is not None:
            return self.event_localizer
        if self._default_localizer is None:
            self._default_localizer = ClapSlidingWindowEventLocalizer(
                model_path=self.clap_model_path,
                prediction_config=self.event_prediction_config,
                localization_config=self.event_localization_config,
                device=self.device,
            )
        return self._default_localizer

    def _predict_labels(
        self,
        *,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
        label_normalizer: LabelNormalizer,
    ) -> List[EventPrediction]:
        predictor = self._get_event_predictor()
        if not hasattr(predictor, "predict"):
            raise TypeError("event_predictor must expose a `predict(audio_paths, candidate_labels)` method.")

        predictions = predictor.predict(audio_paths, candidate_labels)
        if len(predictions) != len(audio_paths):
            raise ValueError(
                "event_predictor returned a different number of predictions than audio inputs: "
                f"{len(predictions)} vs {len(audio_paths)}"
            )

        normalized_predictions: List[EventPrediction] = []
        for prediction in predictions:
            normalized_predictions.append(
                EventPrediction(
                    label=_apply_label_normalizer(prediction.label, label_normalizer),
                    score=prediction.score,
                    scores={
                        _apply_label_normalizer(label, label_normalizer) or label: float(score)
                        for label, score in prediction.scores.items()
                    },
                )
            )
        return normalized_predictions

    def _localize_events(
        self,
        *,
        audio_paths: Sequence[str],
        labels: Sequence[Optional[str]],
        candidate_labels: Sequence[str],
        label_normalizer: LabelNormalizer,
    ) -> List[EventLocalization]:
        localizer = self._get_event_localizer()
        if not hasattr(localizer, "localize"):
            raise TypeError("event_localizer must expose a `localize(audio_paths, labels, candidate_labels)` method.")

        localizations = localizer.localize(audio_paths, labels, candidate_labels)
        if len(localizations) != len(audio_paths):
            raise ValueError(
                "event_localizer returned a different number of localizations than audio inputs: "
                f"{len(localizations)} vs {len(audio_paths)}"
            )

        normalized_localizations: List[EventLocalization] = []
        for localization in localizations:
            normalized_localizations.append(
                EventLocalization(
                    label=_apply_label_normalizer(localization.label, label_normalizer),
                    onset_ms=localization.onset_ms,
                    offset_ms=localization.offset_ms,
                    score=localization.score,
                )
            )
        return normalized_localizations

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        *,
        source_labels: Optional[Sequence[Optional[str]]] = None,
        target_labels: Optional[Sequence[Optional[str]]] = None,
        source_onsets_ms: Optional[Sequence[Optional[Union[int, float]]]] = None,
        target_onsets_ms: Optional[Sequence[Optional[Union[int, float]]]] = None,
        candidate_labels: Optional[Sequence[str]] = None,
        label_normalizer: LabelNormalizer = None,
        sample_ids: Optional[Sequence[str]] = None,
        verbose: bool = True,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
        source_audio_paths = _load_data_list(source_audio, "Source Audio")
        target_audio_paths = _load_data_list(target_audio, "Target Audio")

        if len(source_audio_paths) != len(target_audio_paths):
            raise ValueError(
                f"Source and target size mismatch: {len(source_audio_paths)} vs {len(target_audio_paths)}"
            )
        if not source_audio_paths:
            raise ValueError("No samples found for paralinguistic evaluation.")

        num_samples = len(source_audio_paths)
        normalized_source_labels = _normalize_label_batch(
            source_labels,
            name="source_labels",
            expected_length=num_samples,
            label_normalizer=label_normalizer,
        )
        normalized_target_labels = _normalize_label_batch(
            target_labels,
            name="target_labels",
            expected_length=num_samples,
            label_normalizer=label_normalizer,
        )
        normalized_source_onsets_ms = _normalize_float_batch(
            source_onsets_ms,
            name="source_onsets_ms",
            expected_length=num_samples,
        )
        normalized_target_onsets_ms = _normalize_float_batch(
            target_onsets_ms,
            name="target_onsets_ms",
            expected_length=num_samples,
        )
        resolved_candidate_labels = _resolve_candidate_labels(
            candidate_labels=candidate_labels,
            source_labels=normalized_source_labels,
            target_labels=normalized_target_labels,
            label_normalizer=label_normalizer,
        )

        results: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {
            "num_samples": num_samples,
            "device": self.device,
            "clap_model_path": self.clap_model_path,
        }

        if self.use_continuous_fidelity:
            source_embeddings = self._extract_clap_embeddings(source_audio_paths)
            target_embeddings = self._extract_clap_embeddings(target_audio_paths)
            cosine = self._average_cosine(source_embeddings, target_embeddings)
            results["Paralinguistic_Fidelity_Cosine"] = cosine
            diagnostics["continuous_fidelity"] = {
                "metric": "Paralinguistic_Fidelity_Cosine",
                "num_embeddings": len(source_embeddings),
                "score": cosine,
            }

        source_prediction_records: Optional[List[EventPrediction]] = None
        target_prediction_records: Optional[List[EventPrediction]] = None
        source_label_origin = "provided_source_labels"
        target_label_origin = "provided_target_labels"

        if self.use_event_preservation or self.use_event_alignment:
            if (normalized_source_labels is None or normalized_target_labels is None) and not resolved_candidate_labels:
                raise ValueError(
                    "Event preservation or alignment requires candidate_labels when either source_labels or "
                    "target_labels are not provided."
                )

            if normalized_source_labels is None:
                source_prediction_records = self._predict_labels(
                    audio_paths=source_audio_paths,
                    candidate_labels=resolved_candidate_labels,
                    label_normalizer=label_normalizer,
                )
                normalized_source_labels = [prediction.label for prediction in source_prediction_records]
                source_label_origin = "predicted_source_labels"

            if normalized_target_labels is None:
                target_prediction_records = self._predict_labels(
                    audio_paths=target_audio_paths,
                    candidate_labels=resolved_candidate_labels,
                    label_normalizer=label_normalizer,
                )
                normalized_target_labels = [prediction.label for prediction in target_prediction_records]
                target_label_origin = "predicted_target_labels"

        if self.use_event_preservation:
            assert normalized_source_labels is not None
            assert normalized_target_labels is not None

            metric_payload = _compute_single_label_metrics(
                normalized_source_labels,
                normalized_target_labels,
                class_labels=resolved_candidate_labels,
            )

            use_predicted_reference = source_labels is None
            if use_predicted_reference:
                rate_name = "Predicted_Event_Consistency_Rate"
                macro_f1_name = "Predicted_Event_Consistency_Macro_F1"
                macro_recall_name = "Predicted_Event_Consistency_Macro_Recall"
            else:
                rate_name = "Acoustic_Event_Preservation_Rate"
                macro_f1_name = "Acoustic_Event_Preservation_Macro_F1"
                macro_recall_name = "Acoustic_Event_Preservation_Macro_Recall"

            results[rate_name] = metric_payload["preservation_rate"]
            results[macro_f1_name] = metric_payload["macro_f1"]
            results[macro_recall_name] = metric_payload["macro_recall"]

            diagnostics["event_preservation"] = {
                "candidate_labels": list(resolved_candidate_labels),
                "source_label_origin": source_label_origin,
                "target_label_origin": target_label_origin,
                "config": self.event_prediction_config.to_dict(),
                "num_evaluated": metric_payload["num_evaluated"],
                "num_skipped": metric_payload["num_skipped"],
                "num_abstained": metric_payload["num_abstained"],
                "per_label": metric_payload["per_label"],
                "confusion_matrix": metric_payload["confusion_matrix"],
                "source_predictions": [prediction.to_dict() for prediction in source_prediction_records]
                if source_prediction_records is not None
                else None,
                "target_predictions": [prediction.to_dict() for prediction in target_prediction_records]
                if target_prediction_records is not None
                else None,
                "samples": [
                    {
                        "sample_index": index,
                        "sample_id": sample_ids[index] if sample_ids is not None else str(index),
                        "reference_label": normalized_source_labels[index],
                        "predicted_label": normalized_target_labels[index],
                        "correct": normalized_source_labels[index] is not None
                        and normalized_source_labels[index] == normalized_target_labels[index],
                    }
                    for index in range(num_samples)
                ],
            }

        if self.use_event_alignment and normalized_source_onsets_ms is not None:
            assert normalized_source_labels is not None
            assert normalized_target_labels is not None

            target_localization_records: Optional[List[EventLocalization]] = None
            target_onset_origin = "provided_target_onsets_ms"

            if normalized_target_onsets_ms is None:
                target_localization_records = self._localize_events(
                    audio_paths=target_audio_paths,
                    labels=normalized_target_labels,
                    candidate_labels=resolved_candidate_labels,
                    label_normalizer=label_normalizer,
                )
                normalized_target_onsets_ms = [item.onset_ms for item in target_localization_records]
                target_onset_origin = "localized_target_onsets_ms"

            assert normalized_target_onsets_ms is not None

            source_durations_ms = [_get_audio_duration_ms(path) for path in source_audio_paths]
            target_durations_ms = [_get_audio_duration_ms(path) for path in target_audio_paths]

            alignment_payload = _compute_alignment_metrics(
                normalized_source_labels,
                normalized_target_labels,
                normalized_source_onsets_ms,
                normalized_target_onsets_ms,
                source_durations_ms,
                target_durations_ms,
                relative_onset_tolerance=self.event_alignment_config.relative_onset_tolerance,
                sample_ids=sample_ids,
            )

            if source_labels is None:
                alignment_rate_name = "Predicted_Event_Aligned_Consistency_Rate"
                onset_error_name = "Predicted_Conditional_Relative_Onset_Error"
            else:
                alignment_rate_name = "Event_Aligned_Preservation_Rate"
                onset_error_name = "Conditional_Relative_Onset_Error"

            results[alignment_rate_name] = alignment_payload["aligned_preservation_rate"]
            results[onset_error_name] = alignment_payload["conditional_relative_onset_error"]

            diagnostics["event_alignment"] = {
                "candidate_labels": list(resolved_candidate_labels),
                "source_label_origin": source_label_origin,
                "target_label_origin": target_label_origin,
                "target_onset_origin": target_onset_origin,
                "prediction_config": self.event_prediction_config.to_dict(),
                "localization_config": self.event_localization_config.to_dict(),
                "alignment_config": self.event_alignment_config.to_dict(),
                "source_onsets_ms": normalized_source_onsets_ms,
                "target_onsets_ms": normalized_target_onsets_ms,
                "target_localizations": [item.to_dict() for item in target_localization_records]
                if target_localization_records is not None
                else None,
                "num_evaluated": alignment_payload["num_evaluated"],
                "num_skipped": alignment_payload["num_skipped"],
                "num_aligned": alignment_payload["num_aligned"],
                "num_missing_target_onset": alignment_payload["num_missing_target_onset"],
                "num_conditionally_evaluated": alignment_payload["num_conditionally_evaluated"],
                "relative_onset_tolerance": alignment_payload["relative_onset_tolerance"],
                "samples": alignment_payload["samples"],
            }

        if verbose:
            print("\n[ParalinguisticEvaluator] Summary")
            print(f"  Samples: {num_samples}")
            for metric_name, score in results.items():
                print(f"  {metric_name}: {score}")

        if return_diagnostics:
            return results, diagnostics
        return results


def load_paralinguistic_manifest(path: str) -> List[ParalinguisticSample]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Paralinguistic manifest not found: {path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Paralinguistic manifest must be a JSON list.")

    samples: List[ParalinguisticSample] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {index} must be a dict.")

        source_audio = item.get("source_audio")
        if not source_audio:
            raise ValueError(f"Manifest item {index} is missing source_audio.")

        raw_label = item.get("source_label", item.get("label", item.get("labels")))
        source_label = _coerce_manifest_source_label(raw_label, index)
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        raw_onset = item.get("source_onset_ms", item.get("onset_ms", item.get("start_ms", metadata.get("source_onset_ms"))))
        raw_offset = item.get("source_offset_ms", item.get("offset_ms", item.get("end_ms", metadata.get("source_offset_ms"))))

        samples.append(
            ParalinguisticSample(
                sample_id=str(item.get("id", index)),
                source_audio=ensure_existing_audio(str(source_audio)),
                source_text=str(item.get("source_text", "")).strip(),
                source_label=source_label,
                source_onset_ms=_coerce_optional_float(raw_onset, name="source_onset_ms", index=index),
                source_offset_ms=_coerce_optional_float(raw_offset, name="source_offset_ms", index=index),
                metadata=metadata,
            )
        )
    return samples


def load_paralinguistic_samples(path: str, max_samples: Optional[int] = None) -> List[ParalinguisticSample]:
    samples = load_paralinguistic_manifest(path)
    if max_samples is not None:
        return samples[:max_samples]
    return samples


def build_paralinguistic_inputs(samples: List[ParalinguisticSample]) -> Dict[str, List[Any]]:
    if not samples:
        raise ValueError("No paralinguistic samples provided.")
    return {
        "sample_ids": [sample.sample_id for sample in samples],
        "source_audio": [sample.source_audio for sample in samples],
        "source_text": [sample.source_text for sample in samples],
        "source_labels": [sample.source_label for sample in samples],
        "source_onsets_ms": [sample.source_onset_ms for sample in samples],
        "source_offsets_ms": [sample.source_offset_ms for sample in samples],
        "metadata": [sample.metadata for sample in samples],
    }


def evaluate_paralinguistic_dataset(
    target_audio: List[str],
    *,
    samples: Optional[List[ParalinguisticSample]] = None,
    manifest_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    evaluator: Optional[ParalinguisticEvaluator] = None,
    evaluator_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    return_diagnostics: bool = False,
    sample_transform: Optional[Callable[[List[ParalinguisticSample]], List[ParalinguisticSample]]] = None,
    target_labels: Optional[Sequence[Optional[str]]] = None,
    target_onsets_ms: Optional[Sequence[Optional[Union[int, float]]]] = None,
    candidate_labels: Optional[Sequence[str]] = None,
    label_normalizer: LabelNormalizer = None,
) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
    if samples is None:
        if not manifest_path:
            raise ValueError("manifest_path is required when samples are not provided.")
        samples = load_paralinguistic_manifest(manifest_path)

    if max_samples is not None:
        samples = samples[:max_samples]
    if sample_transform is not None:
        samples = sample_transform(samples)
    if not samples:
        raise ValueError("No paralinguistic samples available.")

    inputs = build_paralinguistic_inputs(samples)
    if len(inputs["source_audio"]) != len(target_audio):
        raise ValueError(
            f"Source/target size mismatch for paralinguistic evaluation: "
            f"{len(inputs['source_audio'])} vs {len(target_audio)}"
        )

    if evaluator is None:
        final_kwargs = dict(evaluator_kwargs or {})
        final_kwargs.setdefault("use_continuous_fidelity", True)
        final_kwargs.setdefault("use_event_preservation", True)
        final_kwargs.setdefault("use_event_alignment", True)
        evaluator = ParalinguisticEvaluator(device=device, **final_kwargs)

    result = evaluator.evaluate_all(
        source_audio=inputs["source_audio"],
        target_audio=target_audio,
        source_labels=inputs["source_labels"],
        target_labels=target_labels,
        source_onsets_ms=inputs["source_onsets_ms"],
        target_onsets_ms=target_onsets_ms,
        candidate_labels=candidate_labels,
        label_normalizer=label_normalizer,
        sample_ids=inputs["sample_ids"],
        verbose=True,
        return_diagnostics=return_diagnostics,
    )

    if return_diagnostics:
        scores, diagnostics = result
        diagnostics["sample_ids"] = inputs["sample_ids"]
        diagnostics["metadata"] = inputs["metadata"]
        return scores, diagnostics

    return result
