from openstbench import SpeakerSimilarityEvaluator


"""
Speaker similarity example.

Required evaluator parameters:
- wavlm_model: an already loaded WavLM-like speaker embedding model.
- resemblyzer_weights_path: local Resemblyzer VoiceEncoder weights.

Optional evaluator parameters:
- device: "cuda", "cpu", or another torch device string.

Required evaluation inputs:
- ref_wav_paths: reference speaker audio paths.
- synth_wav_paths: generated speaker audio paths.

Output metrics from evaluate_batch:
- average_wavlm_large_similarity
- average_resemblyzer_similarity
- details with per-pair wavlm_large_similarity and resemblyzer_similarity
"""


def load_wavlm_speaker_model():
    # Replace this with your project-specific WavLM speaker model loader. The
    # evaluator expects a callable model that accepts a 16 kHz mono waveform
    # tensor shaped [1, num_samples] and returns a speaker embedding tensor.
    raise NotImplementedError("Load and return a WavLM-like speaker embedding model.")


def main():
    wavlm_model = load_wavlm_speaker_model()

    evaluator = SpeakerSimilarityEvaluator(
        wavlm_model=wavlm_model,
        resemblyzer_weights_path="./model/resemblyzer/pretrained.pt",
        device="cuda",
    )

    results = evaluator.evaluate_batch(
        ref_wav_paths=["./ref/1.wav", "./ref/2.wav"],
        synth_wav_paths=["./gen/1.wav", "./gen/2.wav"],
    )

    print(results)


if __name__ == "__main__":
    main()
