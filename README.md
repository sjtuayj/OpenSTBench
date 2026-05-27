# OpenSTBench

English | [Chinese](./README_zh.md)

[![PyPI version](https://badge.fury.io/py/OpenSTBench.svg)](https://pypi.org/project/OpenSTBench/)
[![Python 3.9-3.10](https://img.shields.io/badge/python-3.9--3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenSTBench is a speech and translation evaluation toolkit. It covers text translation quality, speech quality, speaker and style preservation, temporal consistency, and streaming latency.

## Installation

```bash
pip install OpenSTBench
```

For local development:

```bash
pip install -e .
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

BLEURT is installed separately:

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## Package Names

- PyPI package: `OpenSTBench`
- Python import: `openstbench`

## Evaluators

| Evaluator | Scope | Main outputs |
| :--- | :--- | :--- |
| `TranslationEvaluator` | MT and S2TT text quality | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | Generated speech quality and text consistency | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | Speaker preservation | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | Emotion preservation or emotion classification | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | Non-verbal and paralinguistic preservation | `Paralinguistic_Fidelity_Cosine`, `Acoustic_Event_Count_F1`, `Acoustic_Event_Localization_F1` |
| `TemporalConsistencyEvaluator` | Source-target duration consistency | `Duration_Consistency_SLC_*`, duration diagnostics |
| `LatencyEvaluator` | Streaming and simultaneous ST latency | `StartOffset`, `ATD`, `CustomATD`, `RTF` |

## Examples

Usage examples are kept under `examples/`.

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/paralinguistic_identity_baseline.py`
- `examples/python/temporal_consistency_eval.py`
- `examples/python/latency_eval.py`
- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`

Latency can also be run from the module CLI:

```bash
python -m openstbench.latency.cli --help
```

## Conventions

- Text inputs generally accept `list[str]`, one-sample-per-line `.txt` files, and `.json` files.
- Audio inputs generally accept folders, `list[str]`, `.txt` path lists, and `.json` path lists.
- For `zh`, `ja`, and `ko`, text-side evaluation uses CJK-aware handling; speech consistency reports `CER_Consistency` instead of `WER_Consistency`.
- Model path arguments use a local-first rule. If the supplied local path exists, OpenSTBench uses it; otherwise it falls back to the configured remote model id.
- Optional dependencies are loaded only by the evaluator that needs them.

