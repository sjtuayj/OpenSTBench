from openstbench import EmotionEvaluator


def main():
    evaluator = EmotionEvaluator(
        # You can pass e2v_model_path="./model/emotion2vec_plus_large".
        # If that local path does not exist, the evaluator falls back to the
        # default remote model id.
        device="cuda",
    )

    preservation_results = evaluator.evaluate_all(
        source_audio="./src_wavs",
        target_audio="./tgt_wavs",
    )
    print("Emotion preservation:", preservation_results)

    classification_results = evaluator.evaluate_all(
        target_audio="./emotion_wavs",
        reference_labels=["happy", "sad", "neutral"],
    )
    print("Emotion classification:", classification_results)


if __name__ == "__main__":
    main()
