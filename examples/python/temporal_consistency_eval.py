from openstbench import TemporalConsistencyEvaluator


def main():
    evaluator = TemporalConsistencyEvaluator(
        thresholds=(0.2, 0.4),
        use_relative_error=True,
        use_log_ratio_error=True,
    )

    results, diagnostics = evaluator.evaluate_all(
        source_audio="./source_wavs",
        target_audio="./generated_wavs",
        sample_ids=["sample_1", "sample_2"],
        verbose=True,
        return_diagnostics=True,
    )

    print(results)
    print(diagnostics["samples"][:2])


if __name__ == "__main__":
    main()
