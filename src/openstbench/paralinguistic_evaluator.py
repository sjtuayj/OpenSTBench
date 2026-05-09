import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from ._model_loading import resolve_pretrained_source


LabelNormalizer = Optional[Union[Dict[str, Optional[str]], Callable[[str], Optional[str]]]]
DEFAULT_CLAP_MODEL_SOURCE = "laion/clap-htsat-fused"


@dataclass(frozen=True)
class AcousticEvent:
    label: str
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
class ParalinguisticSample:
    sample_id: str
    source_audio: str
    source_text: str = ""
    source_events: Tuple[AcousticEvent, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventPredictionConfig:
    score_threshold: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_threshold": float(self.score_threshold),
        }


@dataclass(frozen=True)
class EventLocalizationConfig:
    window_ms: float = 320.0
    hop_ms: float = 40.0
    merge_gap_ms: float = 80.0
    min_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_ms": float(self.window_ms),
            "hop_ms": float(self.hop_ms),
            "merge_gap_ms": float(self.merge_gap_ms),
            "min_duration_ms": float(self.min_duration_ms),
        }


@dataclass(frozen=True)
class EventMatchingConfig:
    relative_onset_tolerance: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_onset_tolerance": float(self.relative_onset_tolerance),
        }


class BaseAudioEventLocalizer(ABC):
    @abstractmethod
    def localize(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[List[AcousticEvent]]:
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

    try:
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if target_sr is not None and sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        waveform = waveform.squeeze(0).contiguous().float().cpu().numpy()
        return waveform, int(sr)
    except Exception:
        waveform, sr = sf.read(str(path), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if target_sr is not None and int(sr) != int(target_sr):
            tensor = torch.from_numpy(waveform).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, int(sr), int(target_sr))
            waveform = tensor.squeeze(0).contiguous().float().cpu().numpy()
            sr = int(target_sr)
        return waveform.reshape(-1), int(sr)


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


def _coerce_optional_float(raw_value: Any, *, name: str, index: int, event_index: Optional[int] = None) -> Optional[float]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        location = f"manifest item {index}" if event_index is None else f"manifest item {index} event {event_index}"
        raise ValueError(f"{name} for {location} must be numeric or null.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        location = f"manifest item {index}" if event_index is None else f"manifest item {index} event {event_index}"
        raise ValueError(f"{name} for {location} must be numeric or null.") from exc
    if value < 0.0:
        location = f"manifest item {index}" if event_index is None else f"manifest item {index} event {event_index}"
        raise ValueError(f"{name} for {location} must be non-negative.")
    return value


def _normalize_candidate_labels(
    candidate_labels: Sequence[str],
    *,
    label_normalizer: LabelNormalizer,
) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for label in candidate_labels:
        mapped = _apply_label_normalizer(label, label_normalizer)
        if mapped is None or mapped in seen:
            continue
        normalized.append(mapped)
        seen.add(mapped)
    return normalized


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


def _normalize_event(
    raw_event: Dict[str, Any],
    *,
    sample_index: int,
    event_index: int,
    label_normalizer: LabelNormalizer,
) -> AcousticEvent:
    raw_label = raw_event.get("label")
    label = _apply_label_normalizer(None if raw_label is None else str(raw_label), label_normalizer)
    if label is None:
        raise ValueError(f"Manifest item {sample_index} event {event_index} is missing a valid label.")

    onset_ms = _coerce_optional_float(raw_event.get("onset_ms"), name="onset_ms", index=sample_index, event_index=event_index)
    offset_ms = _coerce_optional_float(raw_event.get("offset_ms"), name="offset_ms", index=sample_index, event_index=event_index)
    score = _coerce_optional_float(raw_event.get("score"), name="score", index=sample_index, event_index=event_index)
    if onset_ms is not None and offset_ms is not None and offset_ms < onset_ms:
        raise ValueError(f"offset_ms for manifest item {sample_index} event {event_index} must be >= onset_ms.")

    return AcousticEvent(label=label, onset_ms=onset_ms, offset_ms=offset_ms, score=score)


def _normalize_event_batch(
    events_batch: Sequence[Sequence[Union[AcousticEvent, Dict[str, Any]]]],
    *,
    name: str,
    expected_length: int,
    label_normalizer: LabelNormalizer,
) -> List[List[AcousticEvent]]:
    if len(events_batch) != expected_length:
        raise ValueError(f"{name} size mismatch: {len(events_batch)} vs {expected_length}")

    normalized_batch: List[List[AcousticEvent]] = []
    for sample_index, raw_events in enumerate(events_batch):
        sample_events: List[AcousticEvent] = []
        for event_index, raw_event in enumerate(raw_events):
            if isinstance(raw_event, AcousticEvent):
                mapped_label = _apply_label_normalizer(raw_event.label, label_normalizer)
                if mapped_label is None:
                    raise ValueError(f"{name}[{sample_index}][{event_index}] has an invalid label.")
                sample_events.append(
                    AcousticEvent(
                        label=mapped_label,
                        onset_ms=raw_event.onset_ms,
                        offset_ms=raw_event.offset_ms,
                        score=raw_event.score,
                    )
                )
                continue
            if not isinstance(raw_event, dict):
                raise ValueError(f"{name}[{sample_index}][{event_index}] must be a dict or AcousticEvent.")
            sample_events.append(
                _normalize_event(
                    raw_event,
                    sample_index=sample_index,
                    event_index=event_index,
                    label_normalizer=label_normalizer,
                )
            )
        normalized_batch.append(sample_events)
    return normalized_batch


def _relative_onset(event: AcousticEvent, duration_ms: float) -> Optional[float]:
    if event.onset_ms is None or duration_ms <= 0.0:
        return None
    return float(event.onset_ms / duration_ms)


def _count_events_by_label(events: Sequence[AcousticEvent]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.label] = counts.get(event.label, 0) + 1
    return counts


def _compute_count_metrics(
    reference_batch: Sequence[Sequence[AcousticEvent]],
    predicted_batch: Sequence[Sequence[AcousticEvent]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    tp = 0
    fp = 0
    fn = 0
    per_label_tp: Dict[str, int] = {}
    per_label_fp: Dict[str, int] = {}
    per_label_fn: Dict[str, int] = {}
    sample_records: List[Dict[str, Any]] = []

    for sample_index, (reference_events, predicted_events) in enumerate(zip(reference_batch, predicted_batch)):
        reference_counts = _count_events_by_label(reference_events)
        predicted_counts = _count_events_by_label(predicted_events)
        labels = sorted(set(reference_counts) | set(predicted_counts))
        sample_tp = 0
        sample_fp = 0
        sample_fn = 0

        for label in labels:
            label_tp = min(reference_counts.get(label, 0), predicted_counts.get(label, 0))
            label_fp = max(predicted_counts.get(label, 0) - reference_counts.get(label, 0), 0)
            label_fn = max(reference_counts.get(label, 0) - predicted_counts.get(label, 0), 0)

            tp += label_tp
            fp += label_fp
            fn += label_fn
            sample_tp += label_tp
            sample_fp += label_fp
            sample_fn += label_fn

            per_label_tp[label] = per_label_tp.get(label, 0) + label_tp
            per_label_fp[label] = per_label_fp.get(label, 0) + label_fp
            per_label_fn[label] = per_label_fn.get(label, 0) + label_fn

        sample_records.append(
            {
                "sample_index": int(sample_index),
                "reference_counts": reference_counts,
                "predicted_counts": predicted_counts,
                "tp": int(sample_tp),
                "fp": int(sample_fp),
                "fn": int(sample_fn),
            }
        )

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    per_label: Dict[str, Dict[str, float]] = {}
    for label in sorted(set(per_label_tp) | set(per_label_fp) | set(per_label_fn)):
        label_tp = per_label_tp.get(label, 0)
        label_fp = per_label_fp.get(label, 0)
        label_fn = per_label_fn.get(label, 0)
        label_precision = float(label_tp / (label_tp + label_fp)) if (label_tp + label_fp) > 0 else 0.0
        label_recall = float(label_tp / (label_tp + label_fn)) if (label_tp + label_fn) > 0 else 0.0
        label_f1 = float((2 * label_tp) / (2 * label_tp + label_fp + label_fn)) if (2 * label_tp + label_fp + label_fn) > 0 else 0.0
        per_label[label] = {
            "tp": int(label_tp),
            "fp": int(label_fp),
            "fn": int(label_fn),
            "precision": round(label_precision, 4),
            "recall": round(label_recall, 4),
            "f1": round(label_f1, 4),
        }

    return (
        {
            "Acoustic_Event_Count_F1": round(f1, 4),
        },
        {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "per_label": per_label,
            "samples": sample_records,
        },
    )


def _select_better_match(candidate: Tuple[int, float], current: Optional[Tuple[int, float]]) -> bool:
    if current is None:
        return True
    if candidate[0] != current[0]:
        return candidate[0] > current[0]
    return candidate[1] < current[1]


def _match_same_label_events(
    reference_events: Sequence[AcousticEvent],
    predicted_events: Sequence[AcousticEvent],
    *,
    reference_duration_ms: float,
    predicted_duration_ms: float,
    relative_onset_tolerance: float,
) -> List[Tuple[int, int, float, float]]:
    reference_records = [
        (index, event, _relative_onset(event, reference_duration_ms))
        for index, event in enumerate(reference_events)
        if _relative_onset(event, reference_duration_ms) is not None
    ]
    predicted_records = [
        (index, event, _relative_onset(event, predicted_duration_ms))
        for index, event in enumerate(predicted_events)
        if _relative_onset(event, predicted_duration_ms) is not None
    ]

    if not reference_records or not predicted_records:
        return []

    reference_records.sort(key=lambda item: item[2])
    predicted_records.sort(key=lambda item: item[2])
    num_references = len(reference_records)
    num_predictions = len(predicted_records)
    dp: List[List[Optional[Tuple[int, float]]]] = [[None] * (num_predictions + 1) for _ in range(num_references + 1)]
    decision: List[List[Optional[str]]] = [[None] * (num_predictions + 1) for _ in range(num_references + 1)]

    for i in range(num_references, -1, -1):
        dp[i][num_predictions] = (0, 0.0)
    for j in range(num_predictions, -1, -1):
        dp[num_references][j] = (0, 0.0)

    for i in range(num_references - 1, -1, -1):
        for j in range(num_predictions - 1, -1, -1):
            best = dp[i + 1][j]
            best_decision = "skip_reference"

            skip_prediction = dp[i][j + 1]
            if _select_better_match(skip_prediction, best):
                best = skip_prediction
                best_decision = "skip_prediction"

            reference_relative = float(reference_records[i][2])
            predicted_relative = float(predicted_records[j][2])
            relative_error = abs(reference_relative - predicted_relative)
            if relative_error <= relative_onset_tolerance:
                next_match = dp[i + 1][j + 1]
                candidate = (next_match[0] + 1, next_match[1] + relative_error)
                if _select_better_match(candidate, best):
                    best = candidate
                    best_decision = "match"

            dp[i][j] = best
            decision[i][j] = best_decision

    matches: List[Tuple[int, int, float, float]] = []
    i = 0
    j = 0
    while i < num_references and j < num_predictions:
        current = decision[i][j]
        if current == "match":
            reference_index, reference_event, reference_relative = reference_records[i]
            predicted_index, predicted_event, predicted_relative = predicted_records[j]
            matches.append(
                (
                    reference_index,
                    predicted_index,
                    abs(float(reference_relative) - float(predicted_relative)),
                    abs(float((reference_event.onset_ms or 0.0) - (predicted_event.onset_ms or 0.0))),
                )
            )
            i += 1
            j += 1
        elif current == "skip_prediction":
            j += 1
        else:
            i += 1
    return matches


def _compute_localization_metrics(
    reference_batch: Sequence[Sequence[AcousticEvent]],
    predicted_batch: Sequence[Sequence[AcousticEvent]],
    reference_durations_ms: Sequence[float],
    predicted_durations_ms: Sequence[float],
    *,
    relative_onset_tolerance: float,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    matched_total = 0
    reference_total = sum(len(events) for events in reference_batch)
    predicted_total = sum(len(events) for events in predicted_batch)
    relative_errors: List[float] = []
    absolute_errors_ms: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    for sample_index, (reference_events, predicted_events, reference_duration_ms, predicted_duration_ms) in enumerate(
        zip(reference_batch, predicted_batch, reference_durations_ms, predicted_durations_ms)
    ):
        labels = sorted({event.label for event in reference_events} | {event.label for event in predicted_events})
        matched_reference_indices = set()
        matched_predicted_indices = set()
        matched_pairs: List[Dict[str, Any]] = []

        for label in labels:
            reference_label_events = [
                (index, event)
                for index, event in enumerate(reference_events)
                if event.label == label
            ]
            predicted_label_events = [
                (index, event)
                for index, event in enumerate(predicted_events)
                if event.label == label
            ]
            matches = _match_same_label_events(
                [event for _, event in reference_label_events],
                [event for _, event in predicted_label_events],
                reference_duration_ms=reference_duration_ms,
                predicted_duration_ms=predicted_duration_ms,
                relative_onset_tolerance=relative_onset_tolerance,
            )
            for reference_local_index, predicted_local_index, relative_error, absolute_error_ms in matches:
                reference_index, reference_event = reference_label_events[reference_local_index]
                predicted_index, predicted_event = predicted_label_events[predicted_local_index]
                matched_reference_indices.add(reference_index)
                matched_predicted_indices.add(predicted_index)
                matched_total += 1
                relative_errors.append(relative_error)
                absolute_errors_ms.append(absolute_error_ms)
                matched_pairs.append(
                    {
                        "label": label,
                        "reference_event": reference_event.to_dict(),
                        "predicted_event": predicted_event.to_dict(),
                        "relative_onset_error": round(relative_error, 4),
                        "absolute_onset_error_ms": round(absolute_error_ms, 4),
                    }
                )

        unmatched_reference = [event.to_dict() for index, event in enumerate(reference_events) if index not in matched_reference_indices]
        unmatched_predicted = [event.to_dict() for index, event in enumerate(predicted_events) if index not in matched_predicted_indices]
        sample_records.append(
            {
                "sample_index": int(sample_index),
                "sample_id": sample_ids[sample_index] if sample_ids is not None else str(sample_index),
                "reference_duration_ms": round(float(reference_duration_ms), 4),
                "predicted_duration_ms": round(float(predicted_duration_ms), 4),
                "matched_pairs": matched_pairs,
                "unmatched_reference_events": unmatched_reference,
                "unmatched_predicted_events": unmatched_predicted,
            }
        )

    fp = predicted_total - matched_total
    fn = reference_total - matched_total
    precision = float(matched_total / predicted_total) if predicted_total > 0 else 0.0
    recall = float(matched_total / reference_total) if reference_total > 0 else 0.0
    f1 = float((2 * matched_total) / (2 * matched_total + fp + fn)) if (2 * matched_total + fp + fn) > 0 else 0.0

    return (
        {
            "Acoustic_Event_Localization_F1": round(f1, 4),
            "Acoustic_Event_Onset_Error": _safe_mean(relative_errors),
        },
        {
            "tp": int(matched_total),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "relative_onset_tolerance": round(float(relative_onset_tolerance), 4),
            "mean_relative_onset_error": _safe_mean(relative_errors),
            "mean_absolute_onset_error_ms": _safe_mean(absolute_errors_ms),
            "num_matched_events": int(matched_total),
            "samples": sample_records,
        },
    )


class ClapAudioEventPredictor:
    PROMPT_TEMPLATES = (
        "{label}",
        "the sound of {label}",
        "an audio recording of {label}",
        "a person producing {label}",
    )

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_CLAP_MODEL_SOURCE,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
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
            raise RuntimeError("CLAP-based event localization requires `transformers`.") from exc

        model_source, source_kind = resolve_pretrained_source(
            self.model_path,
            fallback_source=DEFAULT_CLAP_MODEL_SOURCE,
        )
        print(f"Loading CLAP ({source_kind}) from {model_source}...")
        self._processor = ClapProcessor.from_pretrained(model_source)
        self._model = ClapModel.from_pretrained(model_source).to(self.device).eval()

    def _build_prompts(self, candidate_labels: Sequence[str]) -> List[Tuple[str, str]]:
        prompts: List[Tuple[str, str]] = []
        for label in candidate_labels:
            for template in self.PROMPT_TEMPLATES:
                prompts.append((label, template.format(label=label)))
        return prompts

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm > 0.0 else embedding

    def _extract_text_embeddings(self, candidate_labels: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[np.ndarray]]:
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        prompt_records = self._build_prompts(candidate_labels)
        uncached_prompts = [prompt for _, prompt in prompt_records if prompt not in self._text_embedding_cache]
        if uncached_prompts:
            inputs = self._processor(text=uncached_prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            for prompt, embedding in zip(uncached_prompts, text_features.detach().cpu().numpy()):
                self._text_embedding_cache[prompt] = self._normalize_embedding(embedding)

        text_embeddings = [self._text_embedding_cache[prompt] for _, prompt in prompt_records]
        return prompt_records, text_embeddings

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
                audio_features = self._model.get_audio_features(**inputs)
            embeddings.append(self._normalize_embedding(audio_features[0].detach().cpu().numpy()))
        return embeddings

    def score_waveforms(
        self,
        waveforms: Sequence[np.ndarray],
        *,
        sampling_rate: int,
        candidate_labels: Sequence[str],
    ) -> List[Dict[str, float]]:
        prompt_records, text_embeddings = self._extract_text_embeddings(candidate_labels)
        audio_embeddings = self._extract_audio_embeddings_from_waveforms(waveforms, sampling_rate=sampling_rate)
        normalized_text_embeddings = [self._normalize_embedding(embedding) for embedding in text_embeddings]

        scores_per_audio: List[Dict[str, float]] = []
        for audio_embedding in audio_embeddings:
            label_scores: Dict[str, float] = {}
            for (label, _prompt), text_embedding in zip(prompt_records, normalized_text_embeddings):
                score = float(np.dot(audio_embedding, text_embedding))
                if label not in label_scores or score > label_scores[label]:
                    label_scores[label] = score
            scores_per_audio.append(label_scores)
        return scores_per_audio


class ClapSlidingWindowEventLocalizer(BaseAudioEventLocalizer):
    def __init__(
        self,
        *,
        model_path: str = DEFAULT_CLAP_MODEL_SOURCE,
        prediction_config: Optional[EventPredictionConfig] = None,
        localization_config: Optional[EventLocalizationConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.predictor = ClapAudioEventPredictor(model_path=model_path, device=device)
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

    def _merge_label_windows(
        self,
        *,
        label: str,
        windows: Sequence[Tuple[float, float, float]],
    ) -> List[AcousticEvent]:
        if not windows:
            return []

        merge_gap_ms = float(self.localization_config.merge_gap_ms)
        min_duration_ms = float(self.localization_config.min_duration_ms)
        merged_events: List[AcousticEvent] = []

        current_onset, current_offset, current_score = windows[0]
        for onset_ms, offset_ms, score in windows[1:]:
            if onset_ms <= current_offset + merge_gap_ms:
                current_offset = max(current_offset, offset_ms)
                current_score = max(current_score, score)
                continue
            if current_offset - current_onset >= min_duration_ms:
                merged_events.append(
                    AcousticEvent(label=label, onset_ms=current_onset, offset_ms=current_offset, score=current_score)
                )
            current_onset, current_offset, current_score = onset_ms, offset_ms, score

        if current_offset - current_onset >= min_duration_ms:
            merged_events.append(
                AcousticEvent(label=label, onset_ms=current_onset, offset_ms=current_offset, score=current_score)
            )
        return merged_events

    def localize(
        self,
        audio_paths: Sequence[str],
        candidate_labels: Sequence[str],
    ) -> List[List[AcousticEvent]]:
        if not candidate_labels:
            raise ValueError("candidate_labels must not be empty when target events are not provided.")

        threshold = float(self.prediction_config.score_threshold)
        all_localizations: List[List[AcousticEvent]] = []
        for audio_path in tqdm(audio_paths, desc="Localizing acoustic events", unit="file"):
            waveform, sampling_rate = _load_audio_mono(str(audio_path), target_sr=48000)
            if waveform.size == 0:
                all_localizations.append([])
                continue

            segments, starts = self._build_windows(waveform, sampling_rate)
            label_scores_per_window = self.predictor.score_waveforms(
                segments,
                sampling_rate=sampling_rate,
                candidate_labels=candidate_labels,
            )

            active_windows: Dict[str, List[Tuple[float, float, float]]] = {label: [] for label in candidate_labels}
            for start, segment, label_scores in zip(starts, segments, label_scores_per_window):
                onset_ms = float(start / sampling_rate * 1000.0)
                offset_ms = float(min(start + len(segment), len(waveform)) / sampling_rate * 1000.0)
                for label in candidate_labels:
                    score = label_scores.get(label)
                    if score is None or score < threshold:
                        continue
                    active_windows[label].append((onset_ms, offset_ms, float(score)))

            events: List[AcousticEvent] = []
            for label in candidate_labels:
                events.extend(self._merge_label_windows(label=label, windows=active_windows[label]))
            events.sort(key=lambda event: ((event.onset_ms or 0.0), event.label))
            all_localizations.append(events)
        return all_localizations


class ParalinguisticEvaluator:
    DEFAULT_CLAP_MODEL = DEFAULT_CLAP_MODEL_SOURCE

    def __init__(
        self,
        clap_model_path: Optional[str] = None,
        event_localizer: Optional[BaseAudioEventLocalizer] = None,
        event_prediction_config: Optional[Dict[str, Any]] = None,
        event_localization_config: Optional[Dict[str, Any]] = None,
        event_matching_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.device = _to_device(device)
        self.clap_model_path = clap_model_path or self.DEFAULT_CLAP_MODEL
        self.event_prediction_config = EventPredictionConfig(**(event_prediction_config or {}))
        self.event_localization_config = EventLocalizationConfig(**(event_localization_config or {}))
        self.event_matching_config = EventMatchingConfig(**(event_matching_config or {}))
        self.event_localizer = event_localizer
        self._default_localizer: Optional[ClapSlidingWindowEventLocalizer] = None

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

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        *,
        source_events: Sequence[Sequence[Union[AcousticEvent, Dict[str, Any]]]],
        target_events: Optional[Sequence[Sequence[Union[AcousticEvent, Dict[str, Any]]]]] = None,
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
        normalized_source_events = _normalize_event_batch(
            source_events,
            name="source_events",
            expected_length=num_samples,
            label_normalizer=label_normalizer,
        )

        if target_events is None:
            if candidate_labels is None:
                raise ValueError("candidate_labels is required when target_events is not provided.")
            normalized_candidate_labels = _normalize_candidate_labels(candidate_labels, label_normalizer=label_normalizer)
            if not normalized_candidate_labels:
                raise ValueError("candidate_labels resolved to an empty label set.")
            normalized_target_events = self._get_event_localizer().localize(target_audio_paths, normalized_candidate_labels)
            target_event_origin = "localized_target_events"
        else:
            normalized_target_events = _normalize_event_batch(
                target_events,
                name="target_events",
                expected_length=num_samples,
                label_normalizer=label_normalizer,
            )
            normalized_candidate_labels = _normalize_candidate_labels(
                candidate_labels or [event.label for batch in normalized_source_events for event in batch],
                label_normalizer=label_normalizer,
            )
            target_event_origin = "provided_target_events"

        reference_durations_ms = [_get_audio_duration_ms(path) for path in source_audio_paths]
        predicted_durations_ms = [_get_audio_duration_ms(path) for path in target_audio_paths]

        count_results, count_diagnostics = _compute_count_metrics(normalized_source_events, normalized_target_events)
        localization_results, localization_diagnostics = _compute_localization_metrics(
            normalized_source_events,
            normalized_target_events,
            reference_durations_ms,
            predicted_durations_ms,
            relative_onset_tolerance=self.event_matching_config.relative_onset_tolerance,
            sample_ids=sample_ids,
        )

        results = {**count_results, **localization_results}
        diagnostics: Dict[str, Any] = {
            "num_samples": int(num_samples),
            "device": self.device,
            "clap_model_path": self.clap_model_path,
            "candidate_labels": list(normalized_candidate_labels),
            "target_event_origin": target_event_origin,
            "prediction_config": self.event_prediction_config.to_dict(),
            "localization_config": self.event_localization_config.to_dict(),
            "matching_config": self.event_matching_config.to_dict(),
            "count_metrics": count_diagnostics,
            "localization_metrics": localization_diagnostics,
            "source_events": [[event.to_dict() for event in events] for events in normalized_source_events],
            "target_events": [[event.to_dict() for event in events] for events in normalized_target_events],
        }

        if verbose:
            print("\n[ParalinguisticEvaluator] Summary")
            print(f"  Samples: {num_samples}")
            for metric_name, score in results.items():
                print(f"  {metric_name}: {score}")

        if return_diagnostics:
            return results, diagnostics
        return results


def load_paralinguistic_manifest(path: str, *, label_normalizer: LabelNormalizer = None) -> List[ParalinguisticSample]:
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

        raw_events = item.get("source_events")
        if raw_events is None:
            raise ValueError(
                f"Manifest item {index} is missing source_events. "
                "The new paralinguistic evaluator requires explicit source_events."
            )
        if not isinstance(raw_events, list):
            raise ValueError(f"source_events for manifest item {index} must be a list.")

        events = tuple(
            _normalize_event(
                raw_event,
                sample_index=index,
                event_index=event_index,
                label_normalizer=label_normalizer,
            )
            for event_index, raw_event in enumerate(raw_events)
            if isinstance(raw_event, dict)
        )
        if len(events) != len(raw_events):
            raise ValueError(f"source_events for manifest item {index} must contain only dict entries.")

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            ParalinguisticSample(
                sample_id=str(item.get("id", index)),
                source_audio=ensure_existing_audio(str(source_audio)),
                source_text=str(item.get("source_text", "")).strip(),
                source_events=events,
                metadata=metadata,
            )
        )
    return samples


def load_paralinguistic_samples(
    path: str,
    max_samples: Optional[int] = None,
    *,
    label_normalizer: LabelNormalizer = None,
) -> List[ParalinguisticSample]:
    samples = load_paralinguistic_manifest(path, label_normalizer=label_normalizer)
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
        "source_events": [[event.to_dict() for event in sample.source_events] for sample in samples],
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
    target_events: Optional[Sequence[Sequence[Union[AcousticEvent, Dict[str, Any]]]]] = None,
    candidate_labels: Optional[Sequence[str]] = None,
    label_normalizer: LabelNormalizer = None,
) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
    if samples is None:
        if not manifest_path:
            raise ValueError("manifest_path is required when samples are not provided.")
        samples = load_paralinguistic_manifest(manifest_path, label_normalizer=label_normalizer)

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
        evaluator = ParalinguisticEvaluator(device=device, **final_kwargs)

    result = evaluator.evaluate_all(
        source_audio=inputs["source_audio"],
        target_audio=target_audio,
        source_events=inputs["source_events"],
        target_events=target_events,
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
