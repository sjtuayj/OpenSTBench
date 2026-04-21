# OpenSTBench

English | [Chinese](./README_zh.md)

[![PyPI version](https://badge.fury.io/py/OpenSTBench.svg)](https://pypi.org/project/OpenSTBench/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenSTBench is an evaluation toolkit centered on translation and speech translation. It provides a unified way to score text translation quality, speech output quality, preservation-related properties, and streaming latency.

## What It Can Be Used For

This project is best suited for these directions:

- MT or S2TT text-side evaluation with `BLEU`, `chrF++`, `COMET`, and `BLEURT`
- S2ST evaluation by combining text quality, speech quality, speaker similarity, and latency
- Streaming or simultaneous speech translation latency evaluation with a custom agent
- Preservation analysis for speech translation outputs, including speaker similarity, emotion, and paralinguistic similarity

## Core Modules

| Module | Main Use | Typical Metrics |
| :--- | :--- | :--- |
| `TranslationEvaluator` | Text-side translation quality | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | Naturalness and text-speech consistency | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | Speaker preservation | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | Emotion preservation or classification accuracy | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | Non-verbal and paralinguistic preservation | `Paralinguistic_Fidelity_Cosine`, `Acoustic_Event_Preservation_Rate`, `Acoustic_Event_Preservation_Macro_F1`, `Acoustic_Event_Preservation_Macro_Recall`, `Event_Aligned_Preservation_Rate`, `Conditional_Relative_Onset_Error` |
| `LatencyEvaluator` | Streaming / simultaneous translation latency | `StartOffset`, `ATD`, `CustomATD`, `RTF`, `Model_Generate_RTF` |

## Installation

Basic install:

```bash
pip install OpenSTBench
```

Optional extras:

```bash
pip install "OpenSTBench[comet]"
pip install "OpenSTBench[whisper]"
pip install "OpenSTBench[speech_quality]"
pip install "OpenSTBench[emotion]"
pip install "OpenSTBench[paralinguistics]"
pip install "OpenSTBench[all]"
```

If you need BLEURT:

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## Import

PyPI package name:

```python
OpenSTBench
```

Python import name:

```python
openstbench
```

Example:

```python
from openstbench import TranslationEvaluator, SpeechQualityEvaluator
```

## Quick Start

Quick-start scripts live under `examples/`.

Python examples:

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/paralinguistic_identity_baseline.py`
- `examples/python/latency_eval.py`

Shell examples:

- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`

Latency output distinguishes two RTF variants:

- `Real_Time_Factor_(RTF)`: system-level RTF. This includes agent policy overhead, pre/post-processing, and other runtime costs around model inference.
- `Model_Generate_RTF`: model-level RTF. This is reported only when the agent explicitly records model inference time via `record_model_inference_time(...)` or returns it in `Segment.config["model_inference_time"]`.

## Input Conventions

Common text inputs support:

- Python `List[str]`
- `.txt` files with one sample per line
- `.json` files

Common audio inputs support:

- folder path
- Python `List[str]`
- `.txt` files
- `.json` files

## Notes

- For `zh` / `ja` / `ko`, the toolkit uses CJK-aware handling for text-side evaluation.
- `SpeechQualityEvaluator` returns `CER_Consistency` for `zh` / `ja` / `ko`, and `WER_Consistency` for most other languages.
- `ParalinguisticEvaluator` always supports `Paralinguistic_Fidelity_Cosine`, a continuous CLAP-based audio similarity score between source and target speech.
- The discrete preservation branch is an utterance-level single-label task. With source-side gold labels, it reports `Acoustic_Event_Preservation_Rate`, `Acoustic_Event_Preservation_Macro_F1`, and `Acoustic_Event_Preservation_Macro_Recall`.
- If `source_onsets_ms` are available, the evaluator can also report alignment-aware metrics: `Event_Aligned_Preservation_Rate` and `Conditional_Relative_Onset_Error`.
- Alignment is computed on relative onset position, not absolute wall-clock time. This makes it suitable for cross-lingual S2ST where source and target utterance durations naturally differ.
- If target-side onset timestamps are not provided, the default localizer estimates them with CLAP sliding-window scoring conditioned on the target event label.
- These alignment metrics should be interpreted as weak, coarse-grained alignment signals rather than timestamp-accurate event localization benchmarks.
- If source-side gold labels are not available, the evaluator can still run in prediction-only mode and reports `Predicted_Event_Consistency_Rate`, `Predicted_Event_Consistency_Macro_F1`, and `Predicted_Event_Consistency_Macro_Recall`.
- The default discrete predictor is a closed-set CLAP classifier over `candidate_labels`. Users may replace it with any custom predictor object that implements `predict(audio_paths, candidate_labels)`.
- The default event localizer is also replaceable. Custom localizers only need to implement `localize(audio_paths, labels, candidate_labels)`.
- Dataset-specific label mapping is intentionally outside the core package. Pass `candidate_labels` and `label_normalizer` at call time so the same evaluator works across datasets without changing core code.
- For offline environments, `clap_model_path` accepts either a Hugging Face repo id or a local model directory or snapshot.
- In S2S latency evaluation, alignment prefers the model's native transcript when available. If the model is audio-only, the evaluator can optionally use ASR fallback to prepare alignment text.
- For S2S forced alignment, pass language-appropriate MFA models through `alignment_acoustic_model` and `alignment_dictionary_model`. The defaults are English.
- Some modules rely on optional dependencies or local model paths in offline environments.

## License

MIT License
