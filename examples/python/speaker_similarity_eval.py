from openstbench import SpeakerSimilarityEvaluator


def main():
    evaluator = SpeakerSimilarityEvaluator(
        model_type="both",
        # You can also pass wavlm_model_path="./model/wavlm-base-plus-sv".
        # If that local path does not exist, the evaluator falls back to the
        # default remote model id.
        device="cuda",
    )

    results = evaluator.evaluate_batch(
        ref_wav_paths=["./ref/1.wav", "./ref/2.wav"],
        synth_wav_paths=["./gen/1.wav", "./gen/2.wav"],
    )

    print(results)


if __name__ == "__main__":
    main()
