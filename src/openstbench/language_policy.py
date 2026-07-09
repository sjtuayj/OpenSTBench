from __future__ import annotations

from typing import Optional


LANGUAGE_POLICY_VERSION = "2026-07-fleurs-wer-cer-v1"

_LANGUAGE_ALIASES = {
    "cmn": "zh",
    "zho": "zh",
    "chi": "zh",
    "zh": "zh",
    "yue": "zh",
    "eng": "en",
    "en": "en",
    "fra": "fr",
    "fre": "fr",
    "fr": "fr",
    "deu": "de",
    "ger": "de",
    "de": "de",
    "spa": "es",
    "es": "es",
    "jpn": "ja",
    "jp": "ja",
    "ja": "ja",
    "kor": "ko",
    "ko": "ko",
    "tha": "th",
    "th": "th",
    "lao": "lo",
    "lo": "lo",
    "khm": "km",
    "km": "km",
    "mya": "my",
    "bur": "my",
    "my": "my",
    "bod": "bo",
    "tib": "bo",
    "bo": "bo",
    "dzo": "dz",
    "dz": "dz",
}

_CER_LANGUAGES = {
    "zh",
    "ja",
    "ko",
    "th",
    "lo",
    "km",
    "my",
    "bo",
    "dz",
}


def normalize_language_code(language: Optional[str]) -> str:
    text = str(language or "").strip().lower()
    if not text:
        return ""
    if text.startswith("<|") and text.endswith("|>"):
        text = text[2:-2]
    text = text.replace("-", "_")
    base = text.split("_", 1)[0]
    return _LANGUAGE_ALIASES.get(base, base)


def speech_consistency_unit(language: Optional[str]) -> str:
    normalized = normalize_language_code(language)
    return "cer" if normalized in _CER_LANGUAGES else "wer"


def whisper_language_code(language: Optional[str]) -> Optional[str]:
    normalized = normalize_language_code(language)
    return normalized or None


def tokenization_metadata(language: Optional[str]) -> dict:
    normalized = normalize_language_code(language)
    warnings = []
    if not normalized:
        warnings.append("empty_language_code_defaulted_to_wer")
    return {
        "language_policy_version": LANGUAGE_POLICY_VERSION,
        "target_language": str(language or ""),
        "normalized_language": normalized,
        "speech_consistency_unit": speech_consistency_unit(language),
        "whisper_language": whisper_language_code(language),
        "warnings": warnings,
    }
