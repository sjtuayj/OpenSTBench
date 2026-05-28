from openstbench import TemporalConsistencyEvaluator


"""
Temporal consistency example.

Required evaluation inputs:
- source_audio: source speech folder, path list, .txt, or .json.
- target_audio: generated speech folder, path list, .txt, or .json.

Configurable evaluator parameters:
- thresholds: allowed relative duration-ratio deviations for SLC metrics.

Configurable evaluate_all parameters:
- sample_ids: optional IDs with the same length as the audio lists.
- verbose: print a summary report.
- return_diagnostics: return per-sample duration ratios and SLC hits.

Output metrics:
- Duration_Consistency_SLC_0.2
- Duration_Consistency_SLC_0.4
- Additional thresholds produce matching Duration_Consistency_SLC_* keys.
"""


def main():
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
