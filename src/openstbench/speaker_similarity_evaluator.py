import os
from typing import Dict, List, Sequence

import librosa
import numpy as np
import torch
from tqdm import tqdm

from ._model_loading import resolve_pretrained_source

DEFAULT_WAVLM_MODEL_SOURCE = "microsoft/wavlm-base-plus-sv"
DEFAULT_WAVLM_SAMPLE_RATE = 16000
VALID_MODEL_TYPES = {"wavlm", "resemblyzer", "both"}


class SpeakerSimilarityEvaluator:
    """Compute speaker-similarity scores between reference and synthesized audio."""

    def __init__(
        self,
        model_type: str = "wavlm",
        device: str = None,
        wavlm_model_path: str = DEFAULT_WAVLM_MODEL_SOURCE,
        resemblyzer_weights_path: str = "pretrained.pt",
    ):
        self.device = self._resolve_device(device)
        self.model_type = str(model_type).strip().lower()
        if self.model_type not in VALID_MODEL_TYPES:
            valid = ", ".join(sorted(VALID_MODEL_TYPES))
            raise ValueError(f"Unsupported model_type `{model_type}`. Expected one of: {valid}.")

        print(f"Initializing SpeakerSimilarityEvaluator with model(s): {self.model_type} on {self.device}")

        self.wavlm_feature_extractor = None
        self.wavlm_model = None
        self.resemblyzer_encoder = None
        self.resemblyzer_preprocess_wav = None

        if self.model_type in {"wavlm", "both"}:
            self._load_wavlm(wavlm_model_path)

        if self.model_type in {"resemblyzer", "both"}:
            self._load_resemblyzer(resemblyzer_weights_path)

    @staticmethod
    def _resolve_device(device: str = None) -> str:
        if device:
            return str(device)
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_audio_16k_mono(audio_path: str) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=DEFAULT_WAVLM_SAMPLE_RATE, mono=True)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError(f"Loaded empty audio from `{audio_path}`.")
        return audio

    @staticmethod
    def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    @staticmethod
    def _import_transformers_wavlm():
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("USE_TORCH", "1")
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        except Exception as exc:
            raise ImportError(
                "Please ensure the WavLM dependencies are importable before using speaker similarity. "
                f"Root cause: {exc}"
            ) from exc
        return Wav2Vec2FeatureExtractor, WavLMForXVector

    @staticmethod
    def _import_resemblyzer():
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
        except Exception as exc:
            raise ImportError(
                "Please ensure the Resemblyzer dependencies are importable before using speaker similarity. "
                f"Root cause: {exc}"
            ) from exc
        return VoiceEncoder, preprocess_wav

    def _load_wavlm(self, wavlm_model_path: str) -> None:
        Wav2Vec2FeatureExtractor, WavLMForXVector = self._import_transformers_wavlm()
        model_source, source_kind = resolve_pretrained_source(
            wavlm_model_path,
            fallback_source=DEFAULT_WAVLM_MODEL_SOURCE,
        )
        print(f"Loading WavLM ({source_kind}) from {model_source}...")
        try:
            self.wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_source)
            self.wavlm_model = WavLMForXVector.from_pretrained(model_source).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load WavLM speaker model from `{model_source}` ({source_kind}). Root cause: {exc}"
            ) from exc

    def _load_resemblyzer(self, resemblyzer_weights_path: str) -> None:
        VoiceEncoder, preprocess_wav = self._import_resemblyzer()
        self.resemblyzer_preprocess_wav = preprocess_wav
        print("Loading Resemblyzer VoiceEncoder...")
        try:
            if resemblyzer_weights_path and os.path.exists(resemblyzer_weights_path):
                self.resemblyzer_encoder = VoiceEncoder(weights_fpath=resemblyzer_weights_path)
            else:
                self.resemblyzer_encoder = VoiceEncoder()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Resemblyzer VoiceEncoder. "
                f"weights_path=`{resemblyzer_weights_path}`. Root cause: {exc}"
            ) from exc

    @torch.no_grad()
    def _evaluate_wavlm_pair(self, ref_wav_path: str, synth_wav_path: str) -> float:
        ref_audio = self._load_audio_16k_mono(ref_wav_path)
        synth_audio = self._load_audio_16k_mono(synth_wav_path)

        ref_inputs = self.wavlm_feature_extractor(
            ref_audio,
            return_tensors="pt",
            sampling_rate=DEFAULT_WAVLM_SAMPLE_RATE,
        )
        synth_inputs = self.wavlm_feature_extractor(
            synth_audio,
            return_tensors="pt",
            sampling_rate=DEFAULT_WAVLM_SAMPLE_RATE,
        )
        ref_inputs = self._move_batch_to_device(ref_inputs, self.device)
        synth_inputs = self._move_batch_to_device(synth_inputs, self.device)

        ref_embeddings = self.wavlm_model(**ref_inputs).embeddings
        synth_embeddings = self.wavlm_model(**synth_inputs).embeddings
        ref_embeddings = torch.nn.functional.normalize(ref_embeddings, dim=-1)
        synth_embeddings = torch.nn.functional.normalize(synth_embeddings, dim=-1)
        similarity = torch.nn.functional.cosine_similarity(ref_embeddings, synth_embeddings, dim=-1)
        return float(similarity.item())

    def _evaluate_resemblyzer_pair(self, ref_wav_path: str, synth_wav_path: str) -> float:
        ref_wav = self.resemblyzer_preprocess_wav(ref_wav_path)
        synth_wav = self.resemblyzer_preprocess_wav(synth_wav_path)
        ref_embedding = self.resemblyzer_encoder.embed_utterance(ref_wav)
        synth_embedding = self.resemblyzer_encoder.embed_utterance(synth_wav)
        return float(np.inner(ref_embedding, synth_embedding))

    @torch.no_grad()
    def evaluate(self, ref_wav_path: str, synth_wav_path: str) -> Dict[str, float]:
        results: Dict[str, float] = {}

        if self.model_type in {"wavlm", "both"}:
            results["wavlm_similarity"] = self._evaluate_wavlm_pair(ref_wav_path, synth_wav_path)

        if self.model_type in {"resemblyzer", "both"}:
            results["resemblyzer_similarity"] = self._evaluate_resemblyzer_pair(ref_wav_path, synth_wav_path)

        return results

    def evaluate_batch(self, ref_wav_paths: Sequence[str], synth_wav_paths: Sequence[str]) -> Dict[str, object]:
        if len(ref_wav_paths) != len(synth_wav_paths):
            raise ValueError(
                "Reference and synthesized audio lists must have the same length. "
                f"Got {len(ref_wav_paths)} and {len(synth_wav_paths)}."
            )
        if not ref_wav_paths:
            raise ValueError("Speaker-similarity evaluation received an empty audio list.")

        batch_results: Dict[str, object] = {"details": []}
        wavlm_scores: List[float] = []
        res_scores: List[float] = []

        print("Evaluating speaker similarity...")
        for index, (ref_path, synth_path) in enumerate(
            tqdm(zip(ref_wav_paths, synth_wav_paths), total=len(ref_wav_paths))
        ):
            try:
                result = self.evaluate(ref_path, synth_path)
            except Exception as exc:
                raise RuntimeError(
                    "Speaker-similarity evaluation failed for one sample. "
                    f"index={index}, ref=`{ref_path}`, synth=`{synth_path}`. Root cause: {exc}"
                ) from exc

            batch_results["details"].append({"ref": ref_path, "synth": synth_path, "score": result})
            if "wavlm_similarity" in result:
                wavlm_scores.append(result["wavlm_similarity"])
            if "resemblyzer_similarity" in result:
                res_scores.append(result["resemblyzer_similarity"])

        if wavlm_scores:
            batch_results["average_wavlm_similarity"] = float(np.mean(wavlm_scores))
        if res_scores:
            batch_results["average_resemblyzer_similarity"] = float(np.mean(res_scores))

        return batch_results
