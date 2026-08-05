from openstbench import ASRBackend, ASRRouter, TranslationEvaluator, WhisperASRBackend


"""
Translation quality example.

Candidate inputs (at least one evaluation path is required):
- target_text: system text outputs as list[str], .txt, or .json.
- target_audio: generated speech as one file, list[str], or a directory. Whisper
  transcribes it before the ASR metrics are computed.
- asr_text: precomputed ASR output as list[str], .txt, or .json. It is mutually
  exclusive with target_audio and preserves empty rows as failed ASR samples.

Optional evaluation inputs:
- reference: reference translations as list[str], .txt, or .json.
- source: source text, required for COMET, COMETKiwi, and MetricX_QE.
- target_lang: target language code used to choose the BLEU tokenizer and the
  ASR route. English/other space-delimited languages use 13a, Chinese and
  Cantonese use zh, Japanese uses ja-mecab, Korean uses ko-mecab, and the
  configured non-space languages use char.

Configurable evaluator parameters:
- use_bleu: compute sacreBLEU.
- use_chrf: compute chrF++.
- use_comet: compute COMET; requires the comet extra.
- use_bleurt: compute BLEURT; requires bleurt-pytorch.
- use_metricx: compute MetricX; enabled by default and requires the metricx extra.
- use_asr: compute ASR variants when target_audio or asr_text is supplied;
  enabled by default.
- whisper_model: Whisper model name or local path; defaults to "medium".
- whisper_language: optional language override; defaults to target_lang.
- asr_backend: one custom ASR backend used for every language.
- asr_router: route normalized language codes to different ASR backends.
- asr_cache_path: optional JSONL cache for successful ASR transcripts.
- comet_model: local path or remote model id.
- bleurt_path: local BLEURT checkpoint path.
- bleurt_model: local path or remote model id for BLEURT loading.
- metricx_version: "24" or "23"; defaults to "24".
- metricx_model: local path or remote model id for reference-based MetricX.
- metricx_qe_model: local path or remote model id for QE MetricX.
- metricx_tokenizer: local path or remote tokenizer id; defaults to google/mt5-xl.
- device: "cuda", "cpu", or another torch device string.

Output metrics:
- sacreBLEU
- chrF++
- COMET
- BLEURT
- MetricX: reference-based score, lower is better.
- MetricX_QE: reference-free score, lower is better.
- ASR_sacreBLEU, ASR_chrF++, ASR_COMET, ASR_COMETKiwi, ASR_BLEURT,
  ASR_MetricX, and ASR_MetricX_QE.
- ASR_samples_total, ASR_samples_scored, ASR_samples_failed, and ASR_coverage.

MetricX follows the official google-research/metricx README and scores text.
ASR_MetricX first converts target audio to text, then applies the same MetricX
scorer; MetricX itself never consumes raw audio.

Install Japanese/Korean BLEU tokenizers only when needed:
  pip install "OpenSTBench[tokenizer-ja]"
  pip install "OpenSTBench[tokenizer-ko]"

Providing asr_text bypasses Whisper completely. A custom backend must implement
ASRBackend.transcribe(audio_paths, language=None). Keep Cantonese on the
independent "yue" route instead of mapping it to "zh".
""" 


def build_asr_router(cantonese_backend: ASRBackend = None) -> ASRRouter:
    routes = {
        "default": WhisperASRBackend(model="medium"),
        "ja": WhisperASRBackend(model="large-v3"),
    }
    if cantonese_backend is not None:
        # Example: build_asr_router(YourCantoneseASRBackend(...))
        routes["yue"] = cantonese_backend
    return ASRRouter(routes)


def main():
    evaluator = TranslationEvaluator(
        use_bleu=True,
        use_chrf=True,
        use_comet=False,
        use_bleurt=False,
        use_metricx=True,
        comet_model="./model/Unbabel/wmt22-comet-da",
        bleurt_path="./model/lucadiliello/BLEURT-20",
        bleurt_model=None,
        metricx_version="24",
        metricx_model="google/metricx-24-hybrid-large-v2p6",
        metricx_qe_model=None,
        metricx_tokenizer="google/mt5-xl",
        metricx_batch_size=1,
        device="cuda",
        use_asr=True,
        whisper_model="medium",
        whisper_language=None,
        asr_cache_path="./asr_transcripts_whisper_medium.jsonl",
        asr_router=build_asr_router(),
    )

    results = evaluator.evaluate_all(
        reference=["我喜欢看电影。", "今天天气很好。"],
        target_text=["我喜欢看电影。", "今天天气很好。"],
        source=["I like watching movies.", "The weather is nice today."],
        target_lang="zh",
        # Reuse cached transcripts. Replace this with
        # target_audio="./generated_wavs" to run Whisper directly.
        asr_text=["我喜欢看电影。", "今天天气很好。"],
    )

    print(results)

    # Japanese uses ja-mecab for BLEU and CER for speech consistency. Because
    # this call supplies asr_text, neither the default nor Japanese Whisper
    # checkpoint is loaded.
    japanese_results = evaluator.evaluate_all(
        reference=["今日は天気がとても良いです。"],
        target_text=["今日は天気がとても良いです。"],
        source=["The weather is very nice today."],
        target_lang="ja",
        asr_text=["今日は天気がとても良いです。"],
    )
    print(japanese_results)


if __name__ == "__main__":
    main()
