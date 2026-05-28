from openstbench import SpeechQualityEvaluator


"""
Speech quality and text-speech consistency example.

Required evaluation inputs:
- target_audio: generated speech as a folder path, single path, or list[str].

Optional evaluation inputs:
- target_text: generated text used as the WER/CER reference.
- target_lang: language code; zh/ja/ko report CER_Consistency, others report
  WER_Consistency.

Configurable evaluator parameters:
- use_wer: compute ASR-based text-speech consistency.
- use_utmos: compute UTMOS speech naturalness.
- whisper_model: local Whisper path or remote/default Whisper model name.
- whisper_language: optional Whisper language hint.
- utmos_model_path: local path to the SpeechMOS/UTMOS package or model code.
- utmos_ckpt_path: local UTMOS checkpoint path.
- device: "cuda", "cpu", or another torch device string.

Output metrics:
- UTMOS
- WER_Consistency or CER_Consistency
"""


def main():
    evaluator = SpeechQualityEvaluator(
        use_wer=True,
        use_utmos=True,
        whisper_model="medium",
        whisper_language=None,
        utmos_model_path=None,
        utmos_ckpt_path=None,
        device="cuda",
    )

    results = evaluator.evaluate_all(
        target_audio="./generated_wavs",
        target_text=["你好世界", "这是一个测试"],
        target_lang="zh",
    )

    print(results)


if __name__ == "__main__":
    main()
