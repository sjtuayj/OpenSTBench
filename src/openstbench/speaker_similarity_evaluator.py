import os
from typing import Dict, List, Sequence

import librosa
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_WAVLM_SAMPLE_RATE = 16000


class SpeakerSimilarityEvaluator:
    """Compute speaker-similarity scores with a supplied WavLM-like model and Resemblyzer."""

    def __init__(
        self,
        wavlm_model,
        device: str = None,
        resemblyzer_weights_path: str = None,
    ):
        if wavlm_model is None:
            raise ValueError("`wavlm_model` is required.")
        self.device = self._resolve_device(device)
        self.resemblyzer_weights_path = self._require_existing_file(
            resemblyzer_weights_path,
            "resemblyzer_weights_path",
        )

        print(
            "Initializing SpeakerSimilarityEvaluator "
            f"(supplied WavLM-like model + Resemblyzer) on {self.device}"
        )

        self.wavlm_model = wavlm_model
        self.resemblyzer_encoder = None
        self.resemblyzer_preprocess_wav = None

        self._prepare_wavlm_model()
        self._load_resemblyzer()

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
    def _require_existing_file(path: str, argument_name: str) -> str:
        if not path:
            raise ValueError(f"`{argument_name}` is required.")
        normalized = os.path.expanduser(str(path))
        if not os.path.isfile(normalized):
            raise FileNotFoundError(f"`{argument_name}` does not exist: {normalized}")
        return normalized

    @staticmethod
    def _load_audio_16k_mono(audio_path: str) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=DEFAULT_WAVLM_SAMPLE_RATE, mono=True)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError(f"Loaded empty audio from `{audio_path}`.")
        return audio

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

    def _prepare_wavlm_model(self) -> None:
        try:
            if hasattr(self.wavlm_model, "to"):
                self.wavlm_model.to(self.device)
            if hasattr(self.wavlm_model, "eval"):
                self.wavlm_model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to prepare the supplied WavLM-like speaker model. "
                f"Root cause: {exc}"
            ) from exc

    def _load_resemblyzer(self) -> None:
        VoiceEncoder, preprocess_wav = self._import_resemblyzer()
        self.resemblyzer_preprocess_wav = preprocess_wav
        print(f"Loading Resemblyzer VoiceEncoder from {self.resemblyzer_weights_path}...")
        try:
            self.resemblyzer_encoder = VoiceEncoder(weights_fpath=self.resemblyzer_weights_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Resemblyzer VoiceEncoder. "
                f"weights_path=`{self.resemblyzer_weights_path}`. Root cause: {exc}"
            ) from exc

    @torch.no_grad()
    def _evaluate_wavlm_large_pair(self, ref_wav_path: str, synth_wav_path: str) -> float:
        ref_audio = self._load_audio_16k_mono(ref_wav_path)
        synth_audio = self._load_audio_16k_mono(synth_wav_path)
        ref_tensor = torch.tensor(ref_audio).unsqueeze(0).float().to(self.device)
        synth_tensor = torch.tensor(synth_audio).unsqueeze(0).float().to(self.device)

        ref_embedding = self.wavlm_model(ref_tensor).cpu()
        synth_embedding = self.wavlm_model(synth_tensor).cpu()
        similarity = torch.nn.functional.cosine_similarity(ref_embedding, synth_embedding, dim=-1)
        return float(similarity.item())

    def _evaluate_resemblyzer_pair(self, ref_wav_path: str, synth_wav_path: str) -> float:
        ref_wav = self.resemblyzer_preprocess_wav(ref_wav_path)
        synth_wav = self.resemblyzer_preprocess_wav(synth_wav_path)
        ref_embedding = self.resemblyzer_encoder.embed_utterance(ref_wav)
        synth_embedding = self.resemblyzer_encoder.embed_utterance(synth_wav)
        return float(np.inner(ref_embedding, synth_embedding))

    @torch.no_grad()
    def evaluate(self, ref_wav_path: str, synth_wav_path: str) -> Dict[str, float]:
        return {
            "wavlm_large_similarity": self._evaluate_wavlm_large_pair(ref_wav_path, synth_wav_path),
            "resemblyzer_similarity": self._evaluate_resemblyzer_pair(ref_wav_path, synth_wav_path),
        }

    def evaluate_batch(self, ref_wav_paths: Sequence[str], synth_wav_paths: Sequence[str]) -> Dict[str, object]:
        if len(ref_wav_paths) != len(synth_wav_paths):
            raise ValueError(
                "Reference and synthesized audio lists must have the same length. "
                f"Got {len(ref_wav_paths)} and {len(synth_wav_paths)}."
            )
        if not ref_wav_paths:
            raise ValueError("Speaker-similarity evaluation received an empty audio list.")

        details: List[Dict[str, object]] = []
        wavlm_scores: List[float] = []
        resemblyzer_scores: List[float] = []

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

            details.append({"ref": ref_path, "synth": synth_path, "score": result})
            wavlm_scores.append(float(result["wavlm_large_similarity"]))
            resemblyzer_scores.append(float(result["resemblyzer_similarity"]))

        return {
            "details": details,
            "average_wavlm_large_similarity": float(np.mean(wavlm_scores)),
            "average_resemblyzer_similarity": float(np.mean(resemblyzer_scores)),
        }
