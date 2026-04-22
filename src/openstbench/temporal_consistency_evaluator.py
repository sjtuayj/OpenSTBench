import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import torchaudio


class TemporalConsistencyEvaluator:
    """Evaluate source-target temporal consistency at the utterance level."""

    def __init__(
        self,
        thresholds: Sequence[float] = (0.2, 0.4),
        use_relative_error: bool = True,
        use_log_ratio_error: bool = True,
    ) -> None:
        if not thresholds:
            raise ValueError("thresholds must contain at least one value.")

        normalized_thresholds: List[float] = []
        for threshold in thresholds:
            try:
                value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError("thresholds must be numeric.") from exc
            if value < 0.0 or value >= 1.0:
                raise ValueError("each threshold must satisfy 0 <= threshold < 1.")
            normalized_thresholds.append(value)

        self.thresholds = tuple(sorted(set(normalized_thresholds)))
        self.use_relative_error = bool(use_relative_error)
        self.use_log_ratio_error = bool(use_log_ratio_error)

    def _load_audio_from_folder(self, folder_path: str, name: str) -> List[str]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"{name} not found: {folder_path}")
        if not folder.is_dir():
            raise ValueError(f"{name} must be a directory when using folder input: {folder_path}")

        audio_files: List[Path] = []
        for extension in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            audio_files.extend(folder.glob(f"*{extension}"))
        audio_files = sorted(audio_files, key=lambda item: item.stem)
        if not audio_files:
            raise ValueError(f"No audio files found under {name}: {folder_path}")
        return [str(item.resolve()) for item in audio_files]

    def _resolve_audio_paths(self, values: Sequence[str], name: str) -> List[str]:
        resolved: List[str] = []
        for index, value in enumerate(values):
            path = Path(str(value))
            if not path.exists():
                raise FileNotFoundError(f"{name}[{index}] not found: {value}")
            resolved.append(str(path.resolve()))
        return resolved

    def _load_audio_list(self, data: Union[List[str], str], name: str) -> List[str]:
        if isinstance(data, list):
            return self._resolve_audio_paths(data, name)

        if not isinstance(data, str):
            raise ValueError(f"{name} must be a path or a list of paths.")

        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {data}")

        if path.is_dir():
            return self._load_audio_from_folder(data, name)

        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            candidate_keys = ("audio", "path", "file", "wav", "mp3", "source_audio", "target_audio")
            if isinstance(payload, list):
                if not payload:
                    return []
                if isinstance(payload[0], str):
                    return self._resolve_audio_paths(payload, name)
                if isinstance(payload[0], dict):
                    for key in candidate_keys:
                        if key in payload[0]:
                            return self._resolve_audio_paths([str(item[key]) for item in payload], name)
                raise ValueError(f"{name} JSON list must contain strings or dicts with one of {candidate_keys}.")

            if isinstance(payload, dict):
                plural_keys = tuple(f"{key}s" for key in candidate_keys)
                for key in plural_keys:
                    if key in payload:
                        return self._resolve_audio_paths(payload[key], name)
                raise ValueError(f"{name} JSON object must contain one of {plural_keys}.")

            raise ValueError(f"{name} JSON payload must be a list or dict.")

        if suffix == ".txt":
            with open(path, "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file if line.strip()]
            return self._resolve_audio_paths(lines, name)

        return self._resolve_audio_paths([data], name)

    def _get_audio_duration_ms(self, audio_path: str) -> float:
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate > 0 and info.num_frames >= 0:
                return float(info.num_frames / info.sample_rate * 1000.0)
        except Exception:
            pass

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate <= 0:
            return 0.0
        return float(waveform.shape[-1] / sample_rate * 1000.0)

    @staticmethod
    def _threshold_suffix(threshold: float) -> str:
        return f"{threshold:.1f}"

    def _compute_metrics(
        self,
        source_durations_ms: Sequence[float],
        target_durations_ms: Sequence[float],
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        metrics: Dict[str, float] = {}
        sample_records: List[Dict[str, Any]] = []
        valid_ratios: List[float] = []
        relative_errors: List[float] = []
        log_ratio_errors: List[float] = []
        hit_counts = {threshold: 0 for threshold in self.thresholds}
        skipped = 0

        for index, (source_ms, target_ms) in enumerate(zip(source_durations_ms, target_durations_ms)):
            sample_id = sample_ids[index] if sample_ids is not None else str(index)
            record: Dict[str, Any] = {
                "sample_index": int(index),
                "sample_id": str(sample_id),
                "source_duration_ms": round(float(source_ms), 4),
                "target_duration_ms": round(float(target_ms), 4),
            }

            if source_ms <= 0.0 or target_ms <= 0.0:
                skipped += 1
                record["valid"] = False
                record["skip_reason"] = "non_positive_duration"
                sample_records.append(record)
                continue

            ratio = float(target_ms / source_ms)
            relative_error = abs(ratio - 1.0)
            log_ratio_abs = abs(math.log(ratio))

            valid_ratios.append(ratio)
            relative_errors.append(relative_error)
            log_ratio_errors.append(log_ratio_abs)

            slc_hits: Dict[str, bool] = {}
            for threshold in self.thresholds:
                hit = (1.0 - threshold) <= ratio <= (1.0 + threshold)
                slc_hits[self._threshold_suffix(threshold)] = bool(hit)
                if hit:
                    hit_counts[threshold] += 1

            record.update(
                {
                    "valid": True,
                    "duration_ratio": round(ratio, 6),
                    "relative_duration_error": round(relative_error, 6),
                    "log_duration_ratio_abs": round(log_ratio_abs, 6),
                    "slc_hits": slc_hits,
                }
            )
            sample_records.append(record)

        num_evaluated = len(valid_ratios)
        for threshold in self.thresholds:
            suffix = self._threshold_suffix(threshold)
            score = float(hit_counts[threshold] / num_evaluated) if num_evaluated > 0 else 0.0
            metrics[f"Duration_Consistency_SLC_{suffix}"] = round(score, 4)

        if self.use_relative_error:
            relative_mean = sum(relative_errors) / num_evaluated if num_evaluated > 0 else 0.0
            metrics["Relative_Duration_Error_Mean"] = round(float(relative_mean), 4)

        if self.use_log_ratio_error:
            log_ratio_mean = sum(log_ratio_errors) / num_evaluated if num_evaluated > 0 else 0.0
            metrics["Log_Duration_Ratio_MAE"] = round(float(log_ratio_mean), 4)

        diagnostics: Dict[str, Any] = {
            "num_samples": len(source_durations_ms),
            "num_evaluated": num_evaluated,
            "num_skipped": skipped,
            "thresholds": [round(float(value), 4) for value in self.thresholds],
            "samples": sample_records,
        }
        return metrics, diagnostics

    def evaluate_all(
        self,
        source_audio: Union[List[str], str],
        target_audio: Union[List[str], str],
        *,
        sample_ids: Optional[Sequence[str]] = None,
        verbose: bool = True,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, float]]:
        source_audio_paths = self._load_audio_list(source_audio, "source_audio")
        target_audio_paths = self._load_audio_list(target_audio, "target_audio")

        if len(source_audio_paths) != len(target_audio_paths):
            raise ValueError(
                f"Source and target size mismatch: {len(source_audio_paths)} vs {len(target_audio_paths)}"
            )
        if not source_audio_paths:
            raise ValueError("No samples found for temporal consistency evaluation.")
        if sample_ids is not None and len(sample_ids) != len(source_audio_paths):
            raise ValueError(f"sample_ids size mismatch: {len(sample_ids)} vs {len(source_audio_paths)}")

        source_durations_ms = [self._get_audio_duration_ms(path) for path in source_audio_paths]
        target_durations_ms = [self._get_audio_duration_ms(path) for path in target_audio_paths]
        metrics, diagnostics = self._compute_metrics(
            source_durations_ms,
            target_durations_ms,
            sample_ids=sample_ids,
        )

        if verbose:
            print("\n[TemporalConsistencyEvaluator] Summary")
            print(f"  Samples: {diagnostics['num_samples']}")
            print(f"  Evaluated: {diagnostics['num_evaluated']}")
            print(f"  Skipped: {diagnostics['num_skipped']}")
            for metric_name, score in metrics.items():
                print(f"  {metric_name}: {score}")

        if return_diagnostics:
            return metrics, diagnostics
        return metrics
