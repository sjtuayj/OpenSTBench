from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from ._model_loading import resolve_pretrained_source
from .language_policy import normalize_language_code, whisper_language_code


class ASRBackend(ABC):
    """Minimal interface for language-routable ASR implementations."""

    backend_id = "custom"

    @abstractmethod
    def transcribe(
        self,
        audio_paths: Sequence[str],
        language: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError

    @property
    def cache_identity(self) -> str:
        return self.backend_id

    def cleanup(self) -> None:
        return None


class WhisperASRBackend(ASRBackend):
    """Lazy OpenAI Whisper backend with checkpoint-level language checks."""

    backend_id = "whisper"

    def __init__(
        self,
        model: str = "medium",
        *,
        language: Optional[str] = None,
        device: Optional[str] = None,
        fallback_model: str = "medium",
        fp16: Optional[bool] = None,
    ) -> None:
        self.model_name = model
        self.language = whisper_language_code(language)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback_model = fallback_model
        self.fp16 = fp16
        self.model = None
        self.load_failed = False
        self._whisper_module = None

    @property
    def cache_identity(self) -> str:
        return f"{self.backend_id}:{self.model_name}"

    def load_model(self):
        if self.model is not None:
            return self.model
        if self.load_failed:
            return None
        try:
            self._whisper_module = importlib.import_module("whisper")
            model_source, source_kind = resolve_pretrained_source(
                self.model_name,
                fallback_source=self.fallback_model,
            )
            print(f"Loading Whisper ({source_kind}) from {model_source}...")
            self.model = self._whisper_module.load_model(model_source, device=self.device)
        except Exception as exc:
            self.load_failed = True
            print(f"Whisper loading failed: {exc}")
            return None
        return self.model

    def supported_languages(self) -> Optional[set]:
        model = self.load_model()
        if model is None:
            return set()
        try:
            tokenizer_module = importlib.import_module("whisper.tokenizer")
            languages = list(tokenizer_module.LANGUAGES.keys())
        except Exception:
            return None
        num_languages = getattr(model, "num_languages", None)
        if isinstance(num_languages, int) and num_languages >= 0:
            languages = languages[:num_languages]
        return set(languages)

    def supports_language(self, language: Optional[str]) -> bool:
        normalized = whisper_language_code(language)
        if not normalized:
            return True
        supported = self.supported_languages()
        return supported is None or normalized in supported

    def transcribe(
        self,
        audio_paths: Sequence[str],
        language: Optional[str] = None,
    ) -> List[str]:
        model = self.load_model()
        paths = list(audio_paths)
        if model is None:
            return [""] * len(paths)

        effective_language = self.language or whisper_language_code(language)
        if effective_language and not self.supports_language(effective_language):
            print(
                f"Whisper checkpoint '{self.model_name}' does not support "
                f"language '{effective_language}'; ASR is unavailable."
            )
            return [""] * len(paths)

        kwargs = {
            "fp16": (
                self.fp16
                if self.fp16 is not None
                else torch.cuda.is_available() and self.device.startswith("cuda")
            ),
            "task": "transcribe",
        }
        if effective_language:
            kwargs["language"] = effective_language

        transcripts = []
        for audio_path in tqdm(paths, desc=f"Whisper ASR ({self.model_name})"):
            try:
                result = model.transcribe(str(audio_path), **kwargs)
                transcripts.append(str(result.get("text") or "").strip())
            except Exception as exc:
                print(f"ASR transcription failed for {audio_path}: {exc}")
                transcripts.append("")
        return transcripts

    def cleanup(self) -> None:
        self.model = None


class ASRRouter:
    """Resolve an ASR backend by normalized target-language code."""

    def __init__(self, routes: Mapping[str, ASRBackend]) -> None:
        self._routes: Dict[str, ASRBackend] = {}
        self._default = None
        for language, backend in routes.items():
            if not hasattr(backend, "transcribe"):
                raise TypeError("ASR backends must provide transcribe(audio_paths, language)")
            if str(language).strip().lower() == "default":
                self._default = backend
            else:
                normalized = normalize_language_code(language)
                if not normalized:
                    raise ValueError("ASR route language must not be empty")
                self._routes[normalized] = backend

    def resolve(self, language: Optional[str]) -> Optional[ASRBackend]:
        normalized = normalize_language_code(language)
        return self._routes.get(normalized, self._default)

    def transcribe(
        self,
        audio_paths: Sequence[str],
        language: Optional[str] = None,
    ) -> List[str]:
        backend = self.resolve(language)
        if backend is None:
            print(f"No ASR backend is configured for language '{language or ''}'.")
            return [""] * len(list(audio_paths))
        return backend.transcribe(audio_paths, language=language)

    def cleanup(self) -> None:
        seen = set()
        for backend in list(self._routes.values()) + [self._default]:
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            cleanup = getattr(backend, "cleanup", None)
            if callable(cleanup):
                cleanup()
