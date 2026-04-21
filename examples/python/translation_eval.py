from openstbench import TranslationEvaluator


def main():
    evaluator = TranslationEvaluator(
        use_bleu=True,
        use_chrf=True,
        use_comet=False,
        use_bleurt=False,
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
