from openst import evaluate_paralinguistic_dataset, load_paralinguistic_samples


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
    samples = load_paralinguistic_samples(manifest_path)
    source_audio_paths = [sample.source_audio for sample in samples]

    # This baseline only demonstrates preservation metrics. Alignment metrics
    # are returned only if the loaded samples already contain source_onset_ms.
    scores, diagnostics = evaluate_paralinguistic_dataset(
        target_audio=source_audio_paths,
        samples=samples,
        evaluator_kwargs={
            "use_continuous_fidelity": True,
            "use_event_preservation": True,
            "clap_model_path": "./model/clap-htsat-fused",  # Or a local snapshot path
            "event_prediction_config": {
                "score_threshold": 0.2,
                "fallback_top1": False,
            },
        },
        candidate_labels=CANDIDATE_LABELS,
        label_normalizer=normalize_label,
        return_diagnostics=True,
    )

    print(scores)
    print(diagnostics["event_preservation"]["confusion_matrix"])


if __name__ == "__main__":
    main()
