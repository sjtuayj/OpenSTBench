# OpenSTBench

English | [中文](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2605.30792-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.30792)
[![PyPI](https://img.shields.io/pypi/v/OpenSTBench?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/OpenSTBench/)
[![Python](https://img.shields.io/badge/Python-3.9--3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-OpenSTBench-181717?style=for-the-badge&logo=github)](https://github.com/sjtuayj/OpenSTBench)
[![X-LANCE](https://img.shields.io/badge/X--LANCE-grey?labelColor=lightgrey&logo=leanpub&style=for-the-badge)](https://x-lance.sjtu.edu.cn/)

OpenSTBench is a multidimensional evaluation toolkit for speech translation. It is designed for heterogeneous systems, including speech-to-text translation (S2TT), speech-to-speech translation (S2ST), offline systems, and streaming systems.

The toolkit organizes evaluation into three dimensions:

- **Translation Quality**: whether the translated text preserves the source meaning.
- **Speech Quality**: whether generated speech is natural, text-consistent, speaker-preserving, emotion-preserving, and faithful to non-verbal or paralinguistic events.
- **Temporal Quality**: whether generated speech preserves duration structure and, for streaming systems, whether output is responsive.

## Installation

```bash
pip install OpenSTBench
```

For local development:

```bash
git clone https://github.com/sjtuayj/OpenSTBench.git
cd OpenSTBench
conda create -n openstbench python=3.10 -y
conda activate openstbench
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

## Evaluation Dimensions

| Dimension | Evaluator | System type | Main outputs |
| :--- | :--- | :--- | :--- |
| Translation Quality | `TranslationEvaluator` | S2TT, S2ST transcripts | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| Speech Quality | `SpeechQualityEvaluator` | S2ST | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| Speech Quality | `SpeakerSimilarityEvaluator` | S2ST | `average_wavlm_large_similarity`, `average_resemblyzer_similarity` |
| Speech Quality | `EmotionEvaluator` | S2ST | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| Speech Quality | `ParalinguisticEvaluator` | S2ST | `Acoustic_Event_Count_F1`, `Acoustic_Event_Localization_F1`, `Acoustic_Event_Onset_Error` |
| Temporal Quality | `TemporalConsistencyEvaluator` | S2ST | `Duration_Consistency_SLC_0.2`, `Duration_Consistency_SLC_0.4` |
| Temporal Quality | `LatencyEvaluator` | Streaming S2TT/S2ST | `First_Audio_Delay_(StartOffset_ms)`, `Overall_Translation_Delay_(ATD_ms)`, `End_Action_Delay_(CustomATD_ms)`, `Real_Time_Factor_(RTF)` |

Offline and streaming are supported system settings, not separate metric dimensions. Use the evaluators that match the available outputs: text, generated speech, source/target audio pairs, event annotations, or streaming traces.

## Experimental Overview

The radar plot below illustrates the multidimensional view produced by OpenSTBench for representative streaming and offline speech translation systems. It summarizes how systems can differ across translation quality, speech quality, and temporal quality: a system with strong translation quality may still show different behavior in speech realization, speaker or emotion preservation, paralinguistic fidelity, temporal consistency, and latency or efficiency.

![OpenSTBench experimental radar overview](./radar.png)

## Datasets

The paper uses the following datasets. Please follow the license and access terms of each original dataset.

| Dataset | Used for | Link |
| :--- | :--- | :--- |
| MSLT dev | Translation quality, speech quality, temporal consistency, latency | [Microsoft Speech Language Translation Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=54689) |
| LibriTTS-based paired speaker set | Speaker preservation | The constructed OpenSTBench paired set will be released through [GitHub Releases](https://github.com/sjtuayj/OpenSTBench/releases); the source corpus is [LibriTTS](https://www.openslr.org/60/) |
| RAVDESS | Emotion preservation | [Audio_Speech_Actors_01-24.zip](https://zenodo.org/records/1188976) from the RAVDESS Zenodo record |
| MCAE-SPPS | Emotion preservation | [MCAE-SPPS on OSF](https://doi.org/10.17605/OSF.IO/9JYZC) |
| NonverbalTTS test | Paralinguistic fidelity | [deepvk/NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS) |
| SynParaSpeech | Paralinguistic fidelity | [shawnpi/SynParaSpeech](https://huggingface.co/datasets/shawnpi/SynParaSpeech) |

## Quick Start

```python
from openstbench import TranslationEvaluator

evaluator = TranslationEvaluator(
    use_bleu=True,
    use_chrf=True,
    use_comet=False,
    use_bleurt=False,
    device="cuda",
)

scores = evaluator.evaluate_all(
    reference=["我喜欢看电影。", "今天天气很好。"],
    target_text=["我喜欢看电影。", "今天天气很好。"],
    source=["I like watching movies.", "The weather is nice today."],
    target_lang="zh",
)

print(scores)
```

## Examples

Complete parameter templates are kept in `examples/`. The README intentionally stays compact; use these files for configurable parameters, input formats, and output fields.

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

- Text inputs generally accept `list[str]`, one-sample-per-line `.txt` files, and `.json` files where supported by the evaluator.
- Audio inputs generally accept folders, `list[str]`, `.txt` path lists, and `.json` path lists where supported by the evaluator.
- For `zh`, `ja`, and `ko`, speech consistency reports `CER_Consistency`; other languages report `WER_Consistency`.
- Evaluators that accept pretrained model sources use a local-first rule. If the supplied local path exists, OpenSTBench uses it; otherwise it falls back to the configured remote model id.
- Optional dependencies are loaded only when the corresponding evaluator needs them.


## Acknowledgements

- We especially thank [SimulEval](https://github.com/facebookresearch/SimulEval), from which parts of OpenSTBench's latency evaluation components are adapted
- [sacreBLEU](https://github.com/mjpost/sacrebleu), [COMET](https://github.com/Unbabel/COMET), and [bleurt-pytorch](https://github.com/lucadiliello/bleurt-pytorch), a PyTorch port of [BLEURT](https://github.com/google-research/bleurt), for translation quality evaluation
- [Whisper](https://github.com/openai/whisper), [SpeechMOS/UTMOS](https://github.com/tarepan/SpeechMOS), [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), and [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) for speech quality and speaker similarity evaluation
- [FunASR](https://github.com/modelscope/FunASR) and [Emotion2Vec](https://modelscope.cn/models/iic/emotion2vec_plus_large) for emotion preservation evaluation
- [CLAP](https://huggingface.co/laion/clap-htsat-fused) and [Hugging Face Transformers](https://github.com/huggingface/transformers) for paralinguistic event evaluation

## Citation
If you find our work useful, please cite as：

```bibtex
@misc{an2026openstbenchsemanticevaluationspeech,
      title={OpenSTBench: Beyond Semantic Evaluation for Speech Translation}, 
      author={Yanjie An and Yuxiang Zhao and Yichi Zhang and Qixi Zheng and Yujie Tu and Keqi Deng and Kai Yu and Xie Chen},
      year={2026},
      eprint={2605.30792},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2605.30792}, 
}
```

## License

OpenSTBench's original code is released under the MIT License. See [LICENSE](LICENSE).

Some latency evaluation components include code adapted from [SimulEval](https://github.com/facebookresearch/SimulEval), which is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). Those adapted portions are distributed under CC BY-SA 4.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

The datasets referenced by OpenSTBench, including the datasets used in the paper, are not covered by the OpenSTBench code license. They are provided by their original authors or distributors under their own licenses and terms of use. Some datasets are restricted to research or non-commercial use.
