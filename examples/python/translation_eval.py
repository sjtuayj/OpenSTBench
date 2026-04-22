from openstbench import TranslationEvaluator


def main():
    evaluator = TranslationEvaluator(
        use_bleu=True,
        use_chrf=True,
        use_comet=False,
        use_bleurt=False,
        # When enabled, comet_model and bleurt_path may point to local files or
        # directories. Missing local paths fall back to the default remote ids.
        # comet_model="./model/Unbabel/wmt22-comet-da",
        # bleurt_path="./model/lucadiliello/BLEURT-20",
        device="cuda",
    )

    results = evaluator.evaluate_all(
        reference=["我喜欢看电影。", "今天天气很好。"],
        target_text=["我喜欢看电影。", "今天天气很好。"],
        source=["I like watching movies.", "The weather is nice today."],
        target_lang="zh",
    )

    print(results)


if __name__ == "__main__":
    main()
