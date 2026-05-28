from openstbench import evaluate_paralinguistic_dataset, load_paralinguistic_samples


"""
Paralinguistic dataset-helper example.

Manifest format:
- JSON list of samples.
- Each sample must contain source_audio and source_events.
- Each source_events item may include label, onset_ms, offset_ms, and score.

Configurable load_paralinguistic_samples parameters:
- path: manifest path.
- max_samples: optional cap.
- label_normalizer: optional label canonicalizer.

Configurable evaluate_paralinguistic_dataset parameters:
- target_audio: generated audio paths with the same length as samples.
- samples or manifest_path: provide loaded samples or a manifest path.
- max_samples: optional cap when loading inside the helper.
- evaluator: optional prebuilt ParalinguisticEvaluator.
- evaluator_kwargs: kwargs passed to ParalinguisticEvaluator.
- device: device used when the helper creates an evaluator.
- return_diagnostics: return count/localization diagnostics.
- sample_transform: optional callable that transforms loaded samples.
- target_events: optional precomputed target events.
- candidate_labels: labels used when target_events is omitted.
- label_normalizer: label canonicalizer used during evaluation.

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
    "gasp": "gasping",
    "gasping": "gasping",
    "pause": "pause",
    "tsk": "tsk",
}

CANDIDATE_LABELS = [
    "laughter",
    "sighing",
    "throat clearing",
    "gasping",
    "pause",
    "tsk",
]


def normalize_label(label: str):
    normalized = " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
    return LABEL_MAP.get(normalized, normalized)


def main():
    manifest_path = "./prepared/synparaspeech_manifest.json"
    samples = load_paralinguistic_samples(
        manifest_path,
        max_samples=None,
        label_normalizer=normalize_label,
    )
    source_audio_paths = [sample.source_audio for sample in samples]

    # Localization metrics use onset_ms values from each sample's source_events.
    scores, diagnostics = evaluate_paralinguistic_dataset(
        target_audio=source_audio_paths,
        samples=samples,
        manifest_path=None,
        max_samples=None,
        evaluator=None,
        evaluator_kwargs={
            # If this local path exists it is used first; otherwise the evaluator
            # falls back to the default remote model id.
            "clap_model_path": "./model/clap-htsat-fused",
            "event_prediction_config": {
                "score_threshold": 0.2,
            },
        },
        device="cuda",
        candidate_labels=CANDIDATE_LABELS,
        label_normalizer=normalize_label,
        sample_transform=None,
        target_events=None,
        return_diagnostics=True,
    )

    print(scores)
    print(diagnostics["count_metrics"]["per_label"])


if __name__ == "__main__":
    main()
