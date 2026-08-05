from __future__ import annotations

from typing import Optional


LANGUAGE_POLICY_VERSION = "2026-08-multilingual-routing-v1"

_LANGUAGE_ALIASES = {
    "cmn": "zh",
    "zho": "zh",
    "chi": "zh",
    "zh": "zh",
    "yue": "yue",
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
    "por": "pt",
    "pt": "pt",
    "ita": "it",
    "it": "it",
    "rus": "ru",
    "ru": "ru",
    "ukr": "uk",
    "uk": "uk",
    "nld": "nl",
    "dut": "nl",
    "nl": "nl",
    "pol": "pl",
    "pl": "pl",
    "tur": "tr",
    "tr": "tr",
    "ara": "ar",
    "ar": "ar",
    "heb": "he",
    "he": "he",
    "fas": "fa",
    "per": "fa",
    "fa": "fa",
    "hin": "hi",
    "hi": "hi",
    "urd": "ur",
    "ur": "ur",
    "ben": "bn",
    "bn": "bn",
    "tam": "ta",
    "ta": "ta",
    "tel": "te",
    "te": "te",
    "vie": "vi",
    "vi": "vi",
    "ind": "id",
    "id": "id",
    "msa": "ms",
    "may": "ms",
    "ms": "ms",
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
    "yue",
    "ja",
    "ko",
    "th",
    "lo",
    "km",
    "my",
    "bo",
    "dz",
}

_CHAR_BLEU_LANGUAGES = {
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


def bleu_tokenizer_name(language: Optional[str]) -> str:
    """Return the SacreBLEU tokenizer for a target language.

    The existing English and Chinese mappings intentionally remain ``13a``
    and ``zh``. Languages without a dedicated SacreBLEU tokenizer use the
    library's standard ``13a`` tokenizer.
    """

    normalized = normalize_language_code(language)
    if normalized in {"zh", "yue"}:
        return "zh"
    if normalized == "ja":
        return "ja-mecab"
    if normalized == "ko":
        return "ko-mecab"
    if normalized in _CHAR_BLEU_LANGUAGES:
        return "char"
    return "13a"


def bleu_tokenizer_install_hint(tokenizer_name: str) -> Optional[str]:
    if tokenizer_name == "ja-mecab":
        return 'pip install "OpenSTBench[tokenizer-ja]"'
    if tokenizer_name == "ko-mecab":
        return 'pip install "OpenSTBench[tokenizer-ko]"'
    return None


def latency_unit(language: Optional[str]) -> str:
    """Resolve ``auto`` latency tokenization without changing explicit units."""

    return "char" if speech_consistency_unit(language) == "cer" else "word"


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
        "bleu_tokenizer": bleu_tokenizer_name(language),
        "latency_unit": latency_unit(language),
        "whisper_language": whisper_language_code(language),
        "warnings": warnings,
    }
