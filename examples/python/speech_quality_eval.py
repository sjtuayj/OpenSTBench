from openstbench import SpeechQualityEvaluator


def main():
    evaluator = SpeechQualityEvaluator(
        use_wer=True,
        use_utmos=True,
        # You can pass a local Whisper checkpoint path here. If it does not
        # exist, the evaluator falls back to the default remote model id.
        whisper_model="medium",
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
