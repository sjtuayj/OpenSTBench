import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import sacrebleu
import torch
from tqdm import tqdm

from ._model_loading import resolve_pretrained_source
from .language_policy import whisper_language_code
from .metricx_evaluator import (
    DEFAULT_METRICX_TOKENIZER,
    DEFAULT_METRICX_VERSION,
    METRICX_METRIC_NAME,
    METRICX_QE_METRIC_NAME,
    MetricXScorer,
)


# ==================== Configuration ====================

CACHE_PATHS = {
    "huggingface": os.path.expanduser("~/.cache/huggingface/hub"),
}


# ==================== Optional Dependencies ====================

try:
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

    HAS_BLEURT = True
except ImportError:
    HAS_BLEURT = False

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    download_model = None
    load_from_checkpoint = None


# ==================== Input Loaders ====================

def load_text_from_file_or_list(input_data: Union[str, List[str]], name: str = "text") -> List[str]:

    if isinstance(input_data, list):
        return input_data

    if not isinstance(input_data, str):
        raise ValueError(f"{name} must be a file path (str) or a list of strings (List[str])")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {input_data}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            if len(data) == 0:
                return []
            if isinstance(data[0], str):
                return data
            if isinstance(data[0], dict):
                for key in ["target_text", "hypothesis", "text", "ref", "reference", "src", "source"]:
                    if key in data[0]:
                        return [item[key] for item in data]
                raise ValueError("JSON list items do not contain common text fields")

        if isinstance(data, dict):
            for key in ["target_text", "hypothesis", "text", "ref", "reference", "src", "source"]:
                if key in data:
                    return data[key]
            raise ValueError("JSON dictionary does not contain common text fields")

        raise ValueError("Unsupported JSON format")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    audio_files = sorted(audio_files, key=lambda x: x.stem)
    if not audio_files:
        raise ValueError(f"Folder contains no audio files: {folder_path}")
    return [str(f) for f in audio_files]


def load_asr_text_from_file_or_list(input_data: Union[str, List[str]]) -> List[str]:
    """Load ASR text while preserving empty entries used for sample alignment."""

    if isinstance(input_data, list):
        if not all(isinstance(item, str) for item in input_data):
            raise ValueError("ASR Text list items must all be strings")
        return input_data

    if not isinstance(input_data, str):
        raise ValueError("ASR Text must be a file path (str) or a list of strings (List[str])")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"ASR Text file not found: {input_data}")

    if path.suffix.lower() == ".json":
        loaded = load_text_from_file_or_list(input_data, "ASR Text")
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("ASR Text JSON must contain a list of strings")
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines()]


# ==================== Evaluator Core Class ====================

DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"
DEFAULT_COMET_QE_MODEL = "Unbabel/wmt22-cometkiwi-da"
DEFAULT_BLEURT_MODEL = "lucadiliello/BLEURT-20"
DEFAULT_WHISPER_MODEL = "medium"
COMET_METRIC_NAME = "COMET"
COMET_QE_METRIC_NAME = "COMETKiwi"


class TranslationEvaluator:
    """
    Text-side Translation Quality Evaluator: Supports text translation metrics
    such as BLEU, chrF++, COMET, COMETKiwi, BLEURT, and MetricX.
    """

    def __init__(
        self,
        use_bleu: bool = True,
        use_chrf: bool = True,
        use_comet: bool = True,
        use_bleurt: bool = False,
        use_metricx: bool = True,
        comet_model: str = DEFAULT_COMET_MODEL,
        comet_qe_model: str = DEFAULT_COMET_QE_MODEL,
        bleurt_path: Optional[str] = None,
        bleurt_model: Optional[str] = None,
        metricx_version: str = DEFAULT_METRICX_VERSION,
        metricx_model: Optional[str] = None,
        metricx_qe_model: Optional[str] = None,
        metricx_tokenizer: str = DEFAULT_METRICX_TOKENIZER,
        metricx_batch_size: int = 1,
        metricx_max_input_length: Optional[int] = None,
        device: Optional[str] = None,
        use_asr: bool = True,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        whisper_language: Optional[str] = None,
        asr_cache_path: Optional[str] = None,
    ):

        self.use_bleu = use_bleu
        self.use_chrf = use_chrf
        self.use_comet = use_comet
        self.use_bleurt = use_bleurt
        self.use_metricx = use_metricx
        self.use_asr = use_asr

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Initializing Translation Evaluator (Text-only) on {self.device}...")

        self.comet_model = comet_model
        self.comet_qe_model = comet_qe_model
        self.comet = None
        self.comet_qe = None

        self.metricx_version = str(metricx_version)
        self.metricx_model = metricx_model
        self.metricx_qe_model = metricx_qe_model
        self.metricx_tokenizer = metricx_tokenizer
        self.metricx_batch_size = metricx_batch_size
        self.metricx_max_input_length = metricx_max_input_length
        self.metricx_scorer = None

        self.whisper_model_name = whisper_model
        self.whisper_language = whisper_language_code(whisper_language)
        self.whisper_model = None
        self._whisper_load_failed = False
        self.asr_cache_path = asr_cache_path

        self.bleurt_model = None
        self.bleurt_tokenizer = None
        if self.use_bleurt:
            self._load_bleurt(bleurt_path, bleurt_model)

        print("Translation text evaluation metrics system is ready!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        metricx_scorer = getattr(self, "metricx_scorer", None)
        if metricx_scorer is not None:
            metricx_scorer.cleanup()
        for attr in [
            "comet",
            "comet_qe",
            "bleurt_model",
            "bleurt_tokenizer",
            "metricx_scorer",
            "whisper_model",
        ]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _resolve_local_comet_checkpoint(self, model_name: str) -> Optional[str]:
        candidate = Path(model_name).expanduser()
        if not candidate.exists():
            return None

        if candidate.is_file():
            return str(candidate.resolve())

        common_ckpts = [
            candidate / "checkpoints" / "model.ckpt",
            candidate / "model.ckpt",
        ]
        for ckpt in common_ckpts:
            if ckpt.exists() and ckpt.is_file():
                return str(ckpt.resolve())
        return None

    def _load_comet(self, model_name: str, *, fallback_model: str, metric_name: str):
        if not download_model:
            print("COMET is not installed, skipping...")
            return None
        try:
            model_source, source_kind = resolve_pretrained_source(
                model_name,
                fallback_source=fallback_model,
            )
            local_ckpt = self._resolve_local_comet_checkpoint(model_source)
            if local_ckpt is not None:
                print(f"[Local] Loading {metric_name}: {local_ckpt}")
                model = load_from_checkpoint(local_ckpt)
            else:
                remote_source = model_source if source_kind == "remote" else fallback_model
                cache = os.path.join(CACHE_PATHS["huggingface"], f"models--{remote_source.replace('/', '--')}")
                status = "[Local]" if os.path.exists(cache) else "[Online]"
                print(f"{status} Loading {metric_name}: {model_name}")
                print(f"Loading {metric_name} ({status}) from {remote_source}")
                model = load_from_checkpoint(download_model(remote_source))
            if self.device.startswith("cuda"):
                model = model.to(self.device)
            return model
        except Exception as e:
            print(f"{metric_name} loading failed: {e}")
            return None

    def _load_bleurt(self, path: Optional[str], model_name: Optional[str]):
        if not HAS_BLEURT:
            print("bleurt-pytorch is not installed, skipping...")
            return

        if path:
            model_source, _source_kind = resolve_pretrained_source(
                path,
                fallback_source=model_name or DEFAULT_BLEURT_MODEL,
            )
        else:
            model_source, _source_kind = resolve_pretrained_source(model_name or DEFAULT_BLEURT_MODEL)
        print(f"Loading BLEURT: {model_source}")

        try:
            self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(model_source)
            self.bleurt_model = BleurtForSequenceClassification.from_pretrained(model_source)
            self.bleurt_model = self.bleurt_model.to(self.device).eval()
        except Exception as e:
            print(f"BLEURT loading failed: {e}")

    def _load_metricx(self):
        if self.metricx_scorer is None:
            print(f"Loading MetricX-{self.metricx_version}")
            self.metricx_scorer = MetricXScorer(
                version=self.metricx_version,
                model=self.metricx_model,
                qe_model=self.metricx_qe_model,
                tokenizer=self.metricx_tokenizer,
                max_input_length=self.metricx_max_input_length,
                batch_size=self.metricx_batch_size,
                device=self.device,
            )
        return self.metricx_scorer

    def _load_whisper(self):
        if self.whisper_model is not None:
            return self.whisper_model
        if self._whisper_load_failed:
            return None

        try:
            whisper = importlib.import_module("whisper")
            model_source, source_kind = resolve_pretrained_source(
                self.whisper_model_name,
                fallback_source=DEFAULT_WHISPER_MODEL,
            )
            print(f"Loading Whisper ({source_kind}) from {model_source}...")
            self.whisper_model = whisper.load_model(model_source, device=self.device)
        except Exception as e:
            self._whisper_load_failed = True
            print(f"Whisper loading failed: {e}")
            return None
        return self.whisper_model

    def _transcribe_audio(self, audio_paths: List[str], target_lang: str) -> List[str]:
        model = self._load_whisper()
        if model is None:
            return [""] * len(audio_paths)

        transcribe_kwargs = {
            "fp16": torch.cuda.is_available() and self.device.startswith("cuda"),
            "task": "transcribe",
        }
        language = self.whisper_language or whisper_language_code(target_lang)
        if language:
            transcribe_kwargs["language"] = language

        transcripts = []
        for audio_path in tqdm(audio_paths, desc=f"Whisper ASR ({self.whisper_model_name})"):
            path = Path(audio_path)
            if not path.exists() or not path.is_file():
                print(f"ASR skipped missing audio file: {audio_path}")
                transcripts.append("")
                continue
            try:
                result = model.transcribe(str(path), **transcribe_kwargs)
                transcripts.append(str(result.get("text") or "").strip())
            except Exception as e:
                print(f"ASR transcription failed for {audio_path}: {e}")
                transcripts.append("")
        return transcripts

    @staticmethod
    def _audio_sha256(audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists() or not path.is_file():
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _asr_cache_key(record: dict) -> tuple:
        return (
            str(record.get("sample_id") or ""),
            str(record.get("predicted_audio") or "").replace("\\", "/"),
            str(record.get("predicted_audio_sha256") or ""),
            str(record.get("whisper_model") or ""),
            str(record.get("whisper_language") or ""),
        )

    def _read_asr_cache(self) -> List[dict]:
        if not self.asr_cache_path:
            return []
        path = Path(self.asr_cache_path)
        if not path.exists():
            return []

        records = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(
                            f"Skipping invalid ASR cache line {line_number} "
                            f"in {path}: {e}"
                        )
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError as e:
            print(f"ASR cache read failed for {path}: {e}")
            return []
        return records

    def _write_asr_cache(self, records: List[dict]) -> None:
        if not self.asr_cache_path:
            return
        path = Path(self.asr_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temporary_path, path)
        except OSError as e:
            print(f"ASR cache write failed for {path}: {e}")
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _transcribe_audio_with_cache(
        self,
        audio_paths: List[str],
        target_lang: str,
        sample_ids: Optional[List[str]] = None,
    ) -> List[str]:
        if sample_ids is not None and len(sample_ids) != len(audio_paths):
            raise ValueError("ASR sample IDs and Target Audio have different lengths")
        if not self.asr_cache_path:
            return self._transcribe_audio(audio_paths, target_lang)

        effective_language = self.whisper_language or whisper_language_code(target_lang) or ""
        normalized_sample_ids = sample_ids or [""] * len(audio_paths)
        input_records = [
            {
                "sample_id": str(sample_id),
                "predicted_audio": str(audio_path).replace("\\", "/"),
                "predicted_audio_sha256": self._audio_sha256(audio_path),
                "whisper_model": self.whisper_model_name,
                "whisper_language": effective_language,
            }
            for sample_id, audio_path in zip(normalized_sample_ids, audio_paths)
        ]

        existing_records = self._read_asr_cache()
        records_by_key = {
            self._asr_cache_key(record): record for record in existing_records
        }
        transcripts = [""] * len(audio_paths)
        pending_indices = []
        for index, record in enumerate(input_records):
            cached = records_by_key.get(self._asr_cache_key(record))
            cached_text = str((cached or {}).get("asr_text") or "").strip()
            if cached_text:
                transcripts[index] = cached_text
            else:
                pending_indices.append(index)

        if not pending_indices:
            print(f"Reused {len(audio_paths)} ASR transcripts from {self.asr_cache_path}")
            return transcripts

        pending_paths = [audio_paths[index] for index in pending_indices]
        pending_transcripts = self._transcribe_audio(pending_paths, target_lang)
        for index, transcript in zip(pending_indices, pending_transcripts):
            text = str(transcript or "").strip()
            transcripts[index] = text
            record = {**input_records[index], "asr_text": text}
            if not text:
                record["failure_reason"] = (
                    "predicted_audio_missing_file"
                    if not Path(audio_paths[index]).is_file()
                    else "asr_empty_or_failed"
                )
            records_by_key[self._asr_cache_key(record)] = record

        self._write_asr_cache(list(records_by_key.values()))
        reused = len(audio_paths) - len(pending_indices)
        print(
            f"ASR cache: reused {reused}, transcribed {len(pending_indices)}, "
            f"path={self.asr_cache_path}"
        )
        return transcripts

    def get_asr_transcripts(
        self,
        target_audio: Union[List[str], str],
        target_lang: str = "en",
        asr_sample_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Return aligned ASR transcripts, reusing the configured cache when possible."""

        audio_paths = self._load_audio_input(target_audio)
        return self._transcribe_audio_with_cache(
            audio_paths,
            target_lang,
            sample_ids=asr_sample_ids,
        )

    @staticmethod
    def _load_audio_input(target_audio: Union[List[str], str]) -> List[str]:
        if isinstance(target_audio, list):
            if not all(isinstance(item, str) for item in target_audio):
                raise ValueError("Target Audio list items must all be file paths")
            return target_audio
        if not isinstance(target_audio, str):
            raise ValueError("Target Audio must be a file path, directory, or list of file paths")
        if os.path.isdir(target_audio):
            return load_audio_from_folder(target_audio)
        return [target_audio]

    def _get_bleu_tokenizer_name(self, lang: str) -> str:
        if lang == "zh":
            return "zh"
        elif lang == "ja":
            return "ja-mecab"
        elif lang == "ko":
            return "ko-mecab"
        else:
            return "13a"

    def _compute_bleurt_score(self, references: List[str], candidates: List[str]) -> float:
        all_scores = []
        batch_size = 32
        for i in range(0, len(references), batch_size):
            br = references[i : i + batch_size]
            bc = candidates[i : i + batch_size]
            with torch.no_grad():
                inputs = self.bleurt_tokenizer(
                    br,
                    bc,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.bleurt_model(**inputs).logits.flatten().tolist()
                all_scores.extend(scores)
        return float(np.mean(all_scores))

    def _mark_asr_metrics_unavailable(
        self,
        results: Dict[str, Union[int, float]],
        reference: Optional[List[str]],
        source: Optional[List[str]],
    ) -> None:
        if self.use_bleu and reference is not None:
            results["ASR_sacreBLEU"] = -1.0
        if self.use_chrf and reference is not None:
            results["ASR_chrF++"] = -1.0
        if self.use_bleurt and reference is not None:
            results["ASR_BLEURT"] = -1.0
        if self.use_comet:
            if reference is not None:
                results["ASR_COMET"] = -1.0
            results["ASR_COMETKiwi"] = -1.0
        if self.use_metricx:
            if reference is not None:
                results["ASR_MetricX"] = -1.0
            if source is not None:
                results["ASR_MetricX_QE"] = -1.0

    def _score_asr_candidates(
        self,
        candidates: List[str],
        reference: Optional[List[str]],
        source: Optional[List[str]],
        target_lang: str,
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}

        if self.use_bleu and reference is not None:
            tokenizer_name = self._get_bleu_tokenizer_name(target_lang)
            try:
                results["ASR_sacreBLEU"] = sacrebleu.corpus_bleu(
                    candidates,
                    [reference],
                    tokenize=tokenizer_name,
                ).score
            except Exception:
                results["ASR_sacreBLEU"] = -1.0

        if self.use_chrf and reference is not None:
            try:
                results["ASR_chrF++"] = sacrebleu.corpus_chrf(
                    candidates,
                    [reference],
                    word_order=2,
                ).score
            except Exception:
                results["ASR_chrF++"] = -1.0

        if self.use_bleurt and self.bleurt_model and reference is not None:
            try:
                results["ASR_BLEURT"] = self._compute_bleurt_score(reference, candidates)
            except Exception:
                results["ASR_BLEURT"] = -1.0

        if self.use_comet:
            if not source:
                if reference is not None:
                    results["ASR_COMET"] = -1.0
                results["ASR_COMETKiwi"] = -1.0
            else:
                gpus = 1 if self.device.startswith("cuda") else 0

                if reference is not None:
                    if self.comet is None:
                        self.comet = self._load_comet(
                            self.comet_model,
                            fallback_model=DEFAULT_COMET_MODEL,
                            metric_name=COMET_METRIC_NAME,
                        )
                    if self.comet:
                        try:
                            data = [
                                {"src": s, "mt": t, "ref": r}
                                for s, t, r in zip(source, candidates, reference)
                            ]
                            results["ASR_COMET"] = self.comet.predict(
                                data,
                                batch_size=8,
                                gpus=gpus,
                            ).system_score
                        except Exception:
                            results["ASR_COMET"] = -1.0

                if self.comet_qe is None:
                    self.comet_qe = self._load_comet(
                        self.comet_qe_model,
                        fallback_model=DEFAULT_COMET_QE_MODEL,
                        metric_name=COMET_QE_METRIC_NAME,
                    )
                if self.comet_qe:
                    try:
                        data = [{"src": s, "mt": t} for s, t in zip(source, candidates)]
                        results["ASR_COMETKiwi"] = self.comet_qe.predict(
                            data,
                            batch_size=8,
                            gpus=gpus,
                        ).system_score
                    except Exception:
                        results["ASR_COMETKiwi"] = -1.0

        if self.use_metricx:
            if reference is not None:
                try:
                    metricx = self._load_metricx()
                    results["ASR_MetricX"] = metricx.score_reference(
                        candidates=candidates,
                        references=reference,
                        sources=source,
                    )
                except Exception as e:
                    print(f"ASR_MetricX scoring failed: {e}")
                    results["ASR_MetricX"] = -1.0

            if source is not None:
                try:
                    metricx = self._load_metricx()
                    results["ASR_MetricX_QE"] = metricx.score_qe(
                        candidates=candidates,
                        sources=source,
                    )
                except Exception as e:
                    print(f"ASR_MetricX_QE scoring failed: {e}")
                    results["ASR_MetricX_QE"] = -1.0

        return results

    def _evaluate_asr(
        self,
        *,
        reference: Optional[Union[List[str], str]],
        source: Optional[Union[List[str], str]],
        target_audio: Optional[Union[List[str], str]],
        asr_text: Optional[Union[List[str], str]],
        asr_sample_ids: Optional[List[str]],
        target_lang: str,
        expected_count: Optional[int],
    ) -> Dict[str, Union[int, float]]:
        if target_audio is not None and asr_text is not None:
            raise ValueError("target_audio and asr_text are mutually exclusive")

        if asr_text is not None:
            raw_asr_text = load_asr_text_from_file_or_list(asr_text)
        elif target_audio is not None:
            raw_asr_text = self.get_asr_transcripts(
                target_audio,
                target_lang,
                asr_sample_ids=asr_sample_ids,
            )
        else:
            return {}

        total = len(raw_asr_text)
        if expected_count is not None and total != expected_count:
            raise ValueError("ASR input and Target Text have different lengths")

        final_ref = None
        if reference is not None:
            loaded_ref = load_text_from_file_or_list(reference, "Reference")
            final_ref = loaded_ref if loaded_ref else None
            if final_ref is not None and len(final_ref) != total:
                raise ValueError("ASR input and Reference have different lengths")

        final_src = None
        if source:
            final_src = load_text_from_file_or_list(source, "Source")
            if len(final_src) != total:
                raise ValueError("ASR input and Source have different lengths")

        successful_indices = [
            index for index, text in enumerate(raw_asr_text) if str(text).strip()
        ]
        candidates = [str(raw_asr_text[index]).strip() for index in successful_indices]
        filtered_ref = (
            [final_ref[index] for index in successful_indices]
            if final_ref is not None
            else None
        )
        filtered_src = (
            [final_src[index] for index in successful_indices]
            if final_src is not None
            else None
        )

        scored = len(candidates)
        results: Dict[str, Union[int, float]] = {
            "ASR_samples_total": total,
            "ASR_samples_scored": scored,
            "ASR_samples_failed": total - scored,
            "ASR_coverage": scored / total if total else 0.0,
        }
        if not candidates:
            self._mark_asr_metrics_unavailable(results, final_ref, final_src)
            return results

        results.update(
            self._score_asr_candidates(
                candidates,
                filtered_ref,
                filtered_src,
                target_lang,
            )
        )
        return results

    def evaluate_all(
        self,
        reference: Optional[Union[List[str], str]] = None,
        target_text: Optional[Union[List[str], str]] = None,
        source: Optional[Union[List[str], str]] = None,
        target_lang: str = "en",
        target_audio: Optional[Union[List[str], str]] = None,
        asr_text: Optional[Union[List[str], str]] = None,
        asr_sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Args:
            reference: Optional benchmark translation reference from the dataset.
            target_text: Direct text translation output from the translation model.
            source: Optional original source text, mandatory for COMET, COMETKiwi, and MetricX_QE.
            target_lang: Target language, affects BLEU tokenizer selection.
            target_audio: Optional generated target audio for ASR-based metrics.
            asr_text: Optional precomputed ASR text. Mutually exclusive with target_audio.
            asr_sample_ids: Optional sample IDs used to validate and key ASR cache records.
        """
        results = {}
        print(f"\n--- Starting Text Translation Quality Evaluation (Target Lang: {target_lang}) ---")

        has_asr_input = target_audio is not None or asr_text is not None
        if self.use_asr and target_audio is not None and asr_text is not None:
            raise ValueError("target_audio and asr_text are mutually exclusive")

        if target_text is None:
            if not (self.use_asr and has_asr_input):
                raise ValueError("target_text is required when no enabled ASR input is provided")
            results.update(
                self._evaluate_asr(
                    reference=reference,
                    source=source,
                    target_audio=target_audio,
                    asr_text=asr_text,
                    asr_sample_ids=asr_sample_ids,
                    target_lang=target_lang,
                    expected_count=None,
                )
            )
            return {k: round(v, 4) if v >= 0 else v for k, v in results.items()}

        final_text = load_text_from_file_or_list(target_text, "Target Text")

        final_ref = None
        if reference is not None:
            loaded_ref = load_text_from_file_or_list(reference, "Reference")
            final_ref = loaded_ref if loaded_ref else None

        final_src = None
        if source:
            final_src = load_text_from_file_or_list(source, "Source")
            if len(final_src) != len(final_text):
                raise ValueError("Source and Target Text have different lengths")

        if final_ref is not None and len(final_text) != len(final_ref):
            raise ValueError("Target Text and Reference have different lengths")

        # 1. sacreBLEU
        if self.use_bleu and final_ref is not None:
            tokenizer_name = self._get_bleu_tokenizer_name(target_lang)
            try:
                results["sacreBLEU"] = sacrebleu.corpus_bleu(final_text, [final_ref], tokenize=tokenizer_name).score
            except Exception:
                results["sacreBLEU"] = -1.0

        # 2. chrF++
        if self.use_chrf and final_ref is not None:
            try:
                results["chrF++"] = sacrebleu.corpus_chrf(final_text, [final_ref], word_order=2).score
            except Exception:
                results["chrF++"] = -1.0

        # 3. BLEURT
        if self.use_bleurt and self.bleurt_model and final_ref is not None:
            try:
                results["BLEURT"] = self._compute_bleurt_score(final_ref, final_text)
            except Exception:
                results["BLEURT"] = -1.0

        # 4. COMET and COMETKiwi.
        if self.use_comet:
            if not final_src:
                if final_ref is not None:
                    results[COMET_METRIC_NAME] = -1.0
                results[COMET_QE_METRIC_NAME] = -1.0
            else:
                gpus = 1 if self.device.startswith("cuda") else 0

                if final_ref is not None:
                    if self.comet is None:
                        self.comet = self._load_comet(
                            self.comet_model,
                            fallback_model=DEFAULT_COMET_MODEL,
                            metric_name=COMET_METRIC_NAME,
                        )
                    if self.comet:
                        try:
                            data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(final_src, final_text, final_ref)]
                            results[COMET_METRIC_NAME] = self.comet.predict(data, batch_size=8, gpus=gpus).system_score
                        except Exception:
                            results[COMET_METRIC_NAME] = -1.0

                if self.comet_qe is None:
                    self.comet_qe = self._load_comet(
                        self.comet_qe_model,
                        fallback_model=DEFAULT_COMET_QE_MODEL,
                        metric_name=COMET_QE_METRIC_NAME,
                    )
                if self.comet_qe:
                    try:
                        data = [{"src": s, "mt": t} for s, t in zip(final_src, final_text)]
                        results[COMET_QE_METRIC_NAME] = self.comet_qe.predict(data, batch_size=8, gpus=gpus).system_score
                    except Exception:
                        results[COMET_QE_METRIC_NAME] = -1.0

        # 5. MetricX. MetricX is isolated from the other text metrics.
        if self.use_metricx:
            if final_ref is not None:
                try:
                    metricx = self._load_metricx()
                    sources_for_metricx = final_src if final_src is not None else None
                    results[METRICX_METRIC_NAME] = metricx.score_reference(
                        candidates=final_text,
                        references=final_ref,
                        sources=sources_for_metricx,
                    )
                except Exception as e:
                    print(f"{METRICX_METRIC_NAME} scoring failed: {e}")
                    results[METRICX_METRIC_NAME] = -1.0

            if final_src is not None:
                try:
                    metricx = self._load_metricx()
                    results[METRICX_QE_METRIC_NAME] = metricx.score_qe(
                        candidates=final_text,
                        sources=final_src,
                    )
                except Exception as e:
                    print(f"{METRICX_QE_METRIC_NAME} scoring failed: {e}")
                    results[METRICX_QE_METRIC_NAME] = -1.0

        if self.use_asr and has_asr_input:
            results.update(
                self._evaluate_asr(
                    reference=reference,
                    source=source,
                    target_audio=target_audio,
                    asr_text=asr_text,
                    asr_sample_ids=asr_sample_ids,
                    target_lang=target_lang,
                    expected_count=len(final_text),
                )
            )

        return {k: round(v, 4) if v >= 0 else v for k, v in results.items()}
