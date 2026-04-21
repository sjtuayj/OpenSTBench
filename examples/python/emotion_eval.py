from openst import EmotionEvaluator


def main():
    evaluator = EmotionEvaluator(device="cuda")

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
