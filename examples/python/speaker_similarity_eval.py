from openstbench import SpeakerSimilarityEvaluator


def main():
    evaluator = SpeakerSimilarityEvaluator(
        model_type="both",
        device="cuda",
    )

    results = evaluator.evaluate_batch(
        ref_wav_paths=["./ref/1.wav", "./ref/2.wav"],
        synth_wav_paths=["./gen/1.wav", "./gen/2.wav"],
    )

    print(results)


if __name__ == "__main__":
    main()
