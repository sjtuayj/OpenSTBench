"""Threshold-independent acoustic-event detection and evaluation.

The detector is the public ``BEATs_strong_1`` AudioSet-Strong model from
PretrainedSED. Dataset ontologies and benchmark policies deliberately live
outside this module; this file only deals with canonical AudioSet class names,
frame scores, and clip-level event tags.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from ._vendor.pretrained_sed import as_strong_train_classes


BEATS_STRONG_1_URL = (
    "https://github.com/fschmid56/PretrainedSED/releases/download/"
    "v0.0.1/BEATs_strong_1.pt"
)
BEATS_STRONG_1_FILENAME = "BEATs_strong_1.pt"
BEATS_STRONG_1_SIZE = 364_091_999
# Computed from the complete upstream v0.0.1 GitHub Release asset.
BEATS_STRONG_1_SHA256 = "db13a79ae90a0cfd0f9911a6a1d8cdb89324322bee642dcfe32de022123b8b54"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 10.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
FRAMES_PER_CHUNK = 250
FRAME_SECONDS = CHUNK_SECONDS / FRAMES_PER_CHUNK


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_dir() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"])
    else:
        root = Path.home() / ".cache"
    return root / "openstbench" / "models"


class _ExclusiveFileLock:
    def __init__(self, path: Path, timeout_seconds: float = 600.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._fd: Optional[int] = None

    def __enter__(self) -> "_ExclusiveFileLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return self
            except FileExistsError:
                try:
                    owner = int(self.path.read_text(encoding="ascii").strip())
                    if not _process_is_alive(owner):
                        self.path.unlink(missing_ok=True)
                        continue
                except (FileNotFoundError, ValueError):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for checkpoint lock: {self.path}")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd is not None:
            os.close(self._fd)
        self.path.unlink(missing_ok=True)


def _process_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_limited_information, False, process_id)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


@dataclass(frozen=True)
class SEDFrameScores:
    sample_id: str
    frame_boundaries: np.ndarray
    scores: np.ndarray
    event_classes: Tuple[str, ...]

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.frame_boundaries, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float32)
        if scores.ndim != 2:
            raise ValueError("scores must have shape [frames, classes]")
        if boundaries.ndim != 1 or len(boundaries) != scores.shape[0] + 1:
            raise ValueError("frame_boundaries must have one more item than scores")
        if scores.shape[1] != len(self.event_classes):
            raise ValueError("event_classes does not match the score columns")
        if len(boundaries) < 2 or boundaries[0] != 0.0 or np.any(np.diff(boundaries) <= 0.0):
            raise ValueError("frame boundaries must be gapless, increasing, and start at zero")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores contain non-finite values")
        object.__setattr__(self, "frame_boundaries", boundaries)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "event_classes", tuple(self.event_classes))

    @property
    def duration(self) -> float:
        return float(self.frame_boundaries[-1])

    def select_classes(self, event_classes: Sequence[str]) -> "SEDFrameScores":
        index = {name: idx for idx, name in enumerate(self.event_classes)}
        missing = [name for name in event_classes if name not in index]
        if missing:
            raise ValueError(f"Unknown event classes: {missing}")
        columns = [index[name] for name in event_classes]
        return SEDFrameScores(
            sample_id=self.sample_id,
            frame_boundaries=self.frame_boundaries.copy(),
            scores=self.scores[:, columns],
            event_classes=tuple(event_classes),
        )


class _BEATsStrongModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from ._vendor.pretrained_sed.beats import BEATs, BEATsConfig

        self.beats = BEATs(BEATsConfig())
        self.strong_head = nn.Linear(768, len(as_strong_train_classes))
        self.weak_head = nn.Linear(768, len(as_strong_train_classes))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            fbank = self.beats.preprocess(waveform.float())
        features = self.beats.extract_features(fbank.unsqueeze(1), do_preprocess=False)[0]
        features = F.adaptive_avg_pool1d(features.transpose(1, 2), FRAMES_PER_CHUNK).transpose(1, 2)
        return self.strong_head(features)


def _remap_checkpoint_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.model."):
            key = "beats." + key[len("model.model.") :]
        elif key.startswith("model.beats."):
            key = "beats." + key[len("model.beats.") :]
        elif key.startswith("model.") and not key.startswith("model.seq_model."):
            key = key[len("model.") :]
        remapped[key] = value
    return remapped


class BEATsStrongEventDetector:
    """Frame-level AudioSet-Strong detector with no task-specific training."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        *,
        cache_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        download: bool = True,
        show_progress: bool = False,
        model_factory: Callable[[], nn.Module] = _BEATsStrongModel,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {self.device}")
        self.cache_dir = (Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()).resolve()
        self.expected_sha256 = BEATS_STRONG_1_SHA256
        self.download_enabled = bool(download)
        self.show_progress = bool(show_progress)
        self._model_factory = model_factory
        self._model: Optional[nn.Module] = None
        self.model_source = "explicit_path" if model_path is not None else "openstbench_cache"
        self.model_path = self._resolve_checkpoint(model_path)
        self.checkpoint_sha256 = _sha256_file(self.model_path)
        if self.expected_sha256 and self.checkpoint_sha256 != self.expected_sha256:
            raise RuntimeError(
                f"BEATs checkpoint checksum mismatch for {self.model_path}: "
                f"expected {self.expected_sha256}, got {self.checkpoint_sha256}"
            )

    def _resolve_checkpoint(self, model_path: Optional[str | Path]) -> Path:
        if model_path is not None:
            path = Path(model_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"BEATs checkpoint not found: {path}")
            return path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / BEATS_STRONG_1_FILENAME
        checksum_sidecar = destination.with_suffix(destination.suffix + ".sha256")
        lock_path = destination.with_suffix(destination.suffix + ".lock")
        with _ExclusiveFileLock(lock_path):
            if destination.is_file():
                self._verify_cached_checkpoint(destination, checksum_sidecar)
                return destination.resolve()
            if not self.download_enabled:
                raise FileNotFoundError(
                    f"BEATs checkpoint is not cached at {destination}. "
                    "Provide model_path or enable downloading."
                )
            fd, temporary_name = tempfile.mkstemp(
                prefix=BEATS_STRONG_1_FILENAME + ".", suffix=".part", dir=str(self.cache_dir)
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                if self.show_progress:
                    progress = tqdm(
                        total=BEATS_STRONG_1_SIZE,
                        desc="BEATs checkpoint",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        file=sys.stderr,
                    )
                    downloaded = 0

                    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
                        nonlocal downloaded
                        if total_size > 0 and progress.total != total_size:
                            progress.total = total_size
                        current = min(block_count * block_size, progress.total or block_count * block_size)
                        progress.update(max(0, current - downloaded))
                        downloaded = current

                    try:
                        urllib.request.urlretrieve(BEATS_STRONG_1_URL, temporary, reporthook)
                    finally:
                        progress.close()
                else:
                    urllib.request.urlretrieve(BEATS_STRONG_1_URL, temporary)
                if temporary.stat().st_size != BEATS_STRONG_1_SIZE:
                    raise RuntimeError(
                        f"Incomplete BEATs checkpoint: expected {BEATS_STRONG_1_SIZE} bytes, "
                        f"got {temporary.stat().st_size}"
                    )
                digest = _sha256_file(temporary)
                if self.expected_sha256 and digest != self.expected_sha256:
                    raise RuntimeError(
                        f"Downloaded BEATs checkpoint checksum mismatch: expected "
                        f"{self.expected_sha256}, got {digest}"
                    )
                os.replace(temporary, destination)
                checksum_sidecar.write_text(digest + "\n", encoding="ascii")
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    "Unable to download BEATs_strong_1. Supply --beats_model_path "
                    "when running in an offline environment."
                ) from exc
        return destination.resolve()

    def _verify_cached_checkpoint(self, path: Path, sidecar: Path) -> None:
        if path.stat().st_size != BEATS_STRONG_1_SIZE:
            raise RuntimeError(f"Cached BEATs checkpoint has an unexpected size: {path}")
        expected = self.expected_sha256
        if expected is None and sidecar.is_file():
            expected = sidecar.read_text(encoding="ascii").strip().lower()
        if expected:
            actual = _sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"Cached BEATs checkpoint checksum mismatch for {path}: "
                    f"expected {expected}, got {actual}"
                )

    def _load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model
        model = self._model_factory()
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("BEATs checkpoint must contain a state dictionary")
        state_dict = _remap_checkpoint_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"weak_head.weight", "weak_head.bias"}
        missing_set = set(missing)
        if missing_set - allowed_missing or unexpected:
            raise RuntimeError(
                f"Incompatible BEATs_strong_1 checkpoint; missing={sorted(missing_set)}, "
                f"unexpected={sorted(unexpected)}"
            )
        model.to(self.device).eval()
        self._model = model
        return model

    @staticmethod
    def _load_audio(path: str | Path) -> torch.Tensor:
        audio_path = Path(path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception:
            array, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(array.T)
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), SAMPLE_RATE)
        waveform = waveform.squeeze(0).contiguous()
        if waveform.numel() == 0:
            raise ValueError(f"Audio is empty: {audio_path}")
        return waveform

    def predict_file(self, audio_path: str | Path, *, sample_id: Optional[str] = None) -> SEDFrameScores:
        waveform = self._load_audio(audio_path)
        model = self._load_model()
        frame_batches = []
        for start in range(0, waveform.numel(), CHUNK_SAMPLES):
            chunk = waveform[start : start + CHUNK_SAMPLES]
            if chunk.numel() < CHUNK_SAMPLES:
                chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.numel()))
            with torch.inference_mode():
                logits = model(chunk.unsqueeze(0).to(self.device))
                probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            if probabilities.shape != (FRAMES_PER_CHUNK, len(as_strong_train_classes)):
                raise RuntimeError(f"Unexpected BEATs output shape: {probabilities.shape}")
            frame_batches.append(probabilities)
        all_scores = np.concatenate(frame_batches, axis=0)
        duration = waveform.numel() / SAMPLE_RATE
        frame_count = max(1, int(math.ceil(duration / FRAME_SECONDS)))
        all_scores = all_scores[:frame_count]
        boundaries = np.minimum(np.arange(frame_count + 1) * FRAME_SECONDS, duration)
        boundaries[-1] = duration
        if frame_count > 1 and boundaries[-1] <= boundaries[-2]:
            all_scores = all_scores[:-1]
            boundaries = boundaries[:-1]
            boundaries[-1] = duration
        return SEDFrameScores(
            sample_id=sample_id or Path(audio_path).stem,
            frame_boundaries=boundaries,
            scores=all_scores,
            event_classes=tuple(as_strong_train_classes),
        )

    def predict_files(
        self,
        audio_paths: Sequence[str | Path],
        *,
        sample_ids: Optional[Sequence[str]] = None,
        progress_description: Optional[str] = None,
    ) -> Dict[str, SEDFrameScores]:
        if sample_ids is None:
            sample_ids = [Path(path).stem for path in audio_paths]
        if len(sample_ids) != len(audio_paths):
            raise ValueError("sample_ids and audio_paths must have the same length")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        iterator = zip(sample_ids, audio_paths)
        if self.show_progress and progress_description:
            progress = tqdm(
                iterator,
                total=len(audio_paths),
                desc=progress_description,
                unit="audio",
                file=sys.stderr,
            )
            try:
                return {
                    sample_id: self.predict_file(path, sample_id=sample_id)
                    for sample_id, path in progress
                }
            finally:
                progress.close()
        return {
            sample_id: self.predict_file(path, sample_id=sample_id)
            for sample_id, path in iterator
        }

    def model_metadata(self) -> Dict[str, object]:
        return {
            "name": "BEATs_strong_1",
            "source": self.model_source,
            "checkpoint_path": str(self.model_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_url": BEATS_STRONG_1_URL if self.model_source == "openstbench_cache" else None,
            "sample_rate": SAMPLE_RATE,
            "frame_seconds": FRAME_SECONDS,
            "num_classes": len(as_strong_train_classes),
        }


def _import_sed_scores_eval():
    try:
        import sed_scores_eval
    except ImportError as exc:
        raise RuntimeError(
            "Acoustic-event metrics require sed-scores-eval==0.0.4. "
            "Install OpenSTBench with the acoustic-events extra."
        ) from exc
    return sed_scores_eval


def _score_dataframes(
    frame_scores: Mapping[str, SEDFrameScores], event_classes: Sequence[str]
) -> Dict[str, object]:
    _import_sed_scores_eval()
    from sed_scores_eval.base_modules.scores import create_score_dataframe

    dataframes: Dict[str, object] = {}
    for sample_id, item in frame_scores.items():
        selected = item.select_classes(event_classes)
        dataframes[sample_id] = create_score_dataframe(
            selected.scores,
            timestamps=selected.frame_boundaries,
            event_classes=list(selected.event_classes),
        )
    return dataframes


def compute_event_content_map(
    frame_scores: Mapping[str, SEDFrameScores],
    reference_tags: Mapping[str, Sequence[str]],
    *,
    event_classes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    if not frame_scores:
        raise ValueError("frame_scores must not be empty")
    if set(frame_scores) != set(reference_tags):
        raise ValueError("frame_scores and reference_tags must have identical sample ids")
    available = set(next(iter(frame_scores.values())).event_classes)
    requested = list(dict.fromkeys(event_classes or sorted({label for tags in reference_tags.values() for label in tags})))
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Reference classes are unavailable in the detector: {unknown}")
    class_stats: Dict[str, Dict[str, object]] = {}
    evaluable = []
    for label in requested:
        positives = sum(label in set(tags) for tags in reference_tags.values())
        negatives = len(reference_tags) - positives
        eligible = positives > 0 and negatives > 0
        class_stats[label] = {
            "positives": int(positives),
            "negatives": int(negatives),
            "eligible": eligible,
        }
        if eligible:
            evaluable.append(label)
    if not evaluable:
        return {
            "event_content_map": None,
            "per_class": class_stats,
            "eligible_classes": [],
            "num_eligible_classes": 0,
            "num_requested_classes": len(requested),
            "class_coverage": 0.0,
            "warnings": ["no_class_has_both_positive_and_negative_clips"],
        }
    scores = _score_dataframes(frame_scores, evaluable)
    ground_truth = {
        sample_id: [label for label in reference_tags[sample_id] if label in evaluable]
        for sample_id in frame_scores
    }
    sed_scores_eval = _import_sed_scores_eval()
    average_precision, _ = sed_scores_eval.clip_based.average_precision(scores, ground_truth)
    for label in evaluable:
        class_stats[label]["average_precision"] = float(average_precision[label])
    return {
        "event_content_map": float(average_precision["mean"]),
        "per_class": class_stats,
        "eligible_classes": evaluable,
        "num_eligible_classes": len(evaluable),
        "num_requested_classes": len(requested),
        "class_coverage": len(evaluable) / len(requested) if requested else 0.0,
        "warnings": [],
    }


class AcousticEventEvaluator:
    """Small orchestration layer around detection and content mAP."""

    def __init__(self, detector: BEATsStrongEventDetector) -> None:
        self.detector = detector

    def detect(
        self, audio_paths: Sequence[str | Path], *, sample_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, SEDFrameScores]:
        return self.detector.predict_files(audio_paths, sample_ids=sample_ids)

    def evaluate(
        self,
        frame_scores: Mapping[str, SEDFrameScores],
        reference_tags: Mapping[str, Sequence[str]],
        *,
        event_classes: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        return compute_event_content_map(
            frame_scores, reference_tags, event_classes=event_classes
        )


__all__ = [
    "AcousticEventEvaluator",
    "BEATsStrongEventDetector",
    "SEDFrameScores",
    "compute_event_content_map",
]
