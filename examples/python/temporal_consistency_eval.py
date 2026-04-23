from openstbench import TemporalConsistencyEvaluator


def main():
    # The evaluator accepts folders, .txt/.json path lists, or Python lists
    # for both source_audio and target_audio.
    evaluator = TemporalConsistencyEvaluator(
        thresholds=(0.2, 0.4),
    )

    results, diagnostics = evaluator.evaluate_all(
        source_audio="./source_wavs",
        target_audio="./generated_wavs",
        sample_ids=["sample_1", "sample_2"],
        verbose=True,
        return_diagnostics=True,
    )

    print(results)
    # Per-sample diagnostics include source/target duration pairs and
    # thresholded consistency flags.
    print(diagnostics["samples"][:2])


if __name__ == "__main__":
    main()
