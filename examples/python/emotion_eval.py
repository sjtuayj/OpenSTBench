from openstbench import EmotionEvaluator


"""
Emotion preservation and emotion classification example.

Evaluation modes:
- source_audio + target_audio: Emotion2Vec embedding similarity.
- target_audio + reference_labels: discrete emotion classification accuracy.
- source_audio + target_audio + reference_labels: compute both in one call.

Configurable evaluator parameters:
- e2v_model_path: local path or remote id for Emotion2Vec+.
- custom_label_map: optional normalization map for predicted/reference labels.
- device: "cuda", "cpu", or another torch device string.

Configurable evaluate_all parameters:
- source_audio: source speech folder, path list, .txt, or .json.
- target_audio: generated speech folder, path list, .txt, or .json.
- reference_labels: label list, .txt, or .json.
- verbose: print a summary report.

Output metrics:
- Emotion2Vec_Cosine_Similarity
- Audio_Emotion_Accuracy
"""


def main():
    evaluator = EmotionEvaluator(
        e2v_model_path="./model/emotion2vec_plus_large",
        custom_label_map={
            "angry": "angry",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
        },
        device="cuda",
    )

    preservation_results = evaluator.evaluate_all(
        source_audio="./src_wavs",
        target_audio="./tgt_wavs",
        verbose=True,
    )
    print("Emotion preservation:", preservation_results)

    classification_results = evaluator.evaluate_all(
        target_audio="./emotion_wavs",
        reference_labels=["happy", "sad", "neutral"],
        verbose=True,
    )
    print("Emotion classification:", classification_results)


if __name__ == "__main__":
    main()
