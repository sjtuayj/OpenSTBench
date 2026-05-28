from openstbench import ParalinguisticEvaluator


"""
Paralinguistic event-preservation example.

Required evaluation inputs:
- source_audio: source speech folder, path list, .txt, or .json.
- target_audio: generated speech folder, path list, .txt, or .json.
- source_events: one list of event dicts per source sample. Each event may
  include label, onset_ms, offset_ms, and score.

Optional evaluation inputs:
- target_events: precomputed target events. If omitted, CLAP localization is
  run over target_audio using candidate_labels.
- candidate_labels: labels used by CLAP target-event localization.
- label_normalizer: callable that maps label variants to canonical labels.
- sample_ids: optional sample IDs.
- verbose: print a summary report.
- return_diagnostics: return count/localization diagnostics.

Configurable evaluator parameters:
- clap_model_path: local path or remote id for CLAP.
- event_localizer: custom BaseAudioEventLocalizer implementation.
- event_prediction_config: score_threshold.
- event_localization_config: window_ms, hop_ms, merge_gap_ms, min_duration_ms.
- event_matching_config: relative_onset_tolerance.
- device: "cuda", "cpu", or another torch device string.

Output metrics:
- Acoustic_Event_Count_F1
- Acoustic_Event_Localization_F1
- Acoustic_Event_Onset_Error
"""


LABEL_MAP = {
    "laugh": "laughter",
    "laughing": "laughter",
    "laughter": "laughter",
    "sigh": "sighing",
    "sighing": "sighing",
    "clear throat": "throat clearing",
    "clearing throat": "throat clearing",
    "throat clearing": "throat clearing",
}

CANDIDATE_LABELS = [
    "laughter",
    "sighing",
    "throat clearing",
]


def normalize_label(label: str):
    normalized = " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
    return LABEL_MAP.get(normalized, normalized)


def main():
    source_events = [
        [{"label": "laugh", "onset_ms": 1100.0}],
        [{"label": "throat clearing", "onset_ms": 2350.0}],
    ]

    evaluator = ParalinguisticEvaluator(
        # If this local path exists it is used first; otherwise the evaluator
        # falls back to the default remote model id.
        clap_model_path="./model/clap-htsat-fused",
        event_localizer=None,
        event_prediction_config={
            "score_threshold": 0.2,
        },
        event_localization_config={
            "window_ms": 320.0,
            "hop_ms": 40.0,
            "merge_gap_ms": 80.0,
            "min_duration_ms": 0.0,
        },
        event_matching_config={
            "relative_onset_tolerance": 0.15,
        },
        device="cuda",
    )

    scores, diagnostics = evaluator.evaluate_all(
        source_audio=["./src_wavs/sample_001.wav", "./src_wavs/sample_002.wav"],
        target_audio=["./tgt_wavs/sample_001.wav", "./tgt_wavs/sample_002.wav"],
        source_events=source_events,
        target_events=None,
        candidate_labels=CANDIDATE_LABELS,
        label_normalizer=normalize_label,
        sample_ids=["sample_001", "sample_002"],
        verbose=True,
        return_diagnostics=True,
    )

    print(scores)
    print(diagnostics["count_metrics"]["per_label"])
    print(diagnostics["localization_metrics"]["samples"])


if __name__ == "__main__":
    main()
