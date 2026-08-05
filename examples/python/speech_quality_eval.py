from openstbench import ASRBackend, ASRRouter, SpeechQualityEvaluator, WhisperASRBackend


"""
Speech quality and text-speech consistency example.

Required evaluation inputs:
- target_audio: generated speech as a folder path, single path, or list[str].

Optional evaluation inputs:
- target_text: generated text used as the WER/CER reference.
- target_lang: language code. zh/yue/ja/ko/th/lo/km/my/bo/dz report
  CER_Consistency; space-delimited languages report WER_Consistency.
- asr_text: optional precomputed transcripts. When supplied, no ASR backend is
  loaded. This is also the fallback for languages unsupported by Whisper.

Configurable evaluator parameters:
- use_wer: compute ASR-based text-speech consistency.
- use_utmos: compute UTMOS speech naturalness.
- whisper_model: local Whisper path or remote/default Whisper model name.
- whisper_language: optional Whisper language hint.
- asr_backend: one custom ASR backend used for every language.
- asr_router: route normalized language codes to different ASR backends.
- utmos_model_path: local path to the SpeechMOS/UTMOS package or model code.
- utmos_ckpt_path: local UTMOS checkpoint path.
- device: "cuda", "cpu", or another torch device string.

Output metrics:
- UTMOS
- WER_Consistency or CER_Consistency
"""


def build_asr_router(cantonese_backend: ASRBackend = None) -> ASRRouter:
    routes = {
        "default": WhisperASRBackend(model="medium"),
        "ja": WhisperASRBackend(model="large-v3"),
    }
    if cantonese_backend is not None:
        # Keep Cantonese as "yue". Supply a real Cantonese backend here when
        # the selected Whisper checkpoint does not expose that language.
        routes["yue"] = cantonese_backend
    return ASRRouter(routes)


def main():
    evaluator = SpeechQualityEvaluator(
        use_wer=True,
        use_utmos=True,
        whisper_model="medium",
        whisper_language=None,
        utmos_model_path=None,
        utmos_ckpt_path=None,
        device="cuda",
        asr_router=build_asr_router(),
    )

    results = evaluator.evaluate_all(
        target_audio="./generated_wavs",
        target_text=["你好世界", "这是一个测试"],
        target_lang="zh",
        # Remove asr_text to transcribe through the configured ASRRouter.
        asr_text=["你好世界", "这是一个测试"],
    )

    print(results)


if __name__ == "__main__":
    main()
