# OpenSTBench

[English](./README.md) | 中文

[![PyPI version](https://badge.fury.io/py/OpenSTBench.svg)](https://pypi.org/project/OpenSTBench/)
[![Python 3.9-3.10](https://img.shields.io/badge/python-3.9--3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenSTBench 是一个语音与翻译评测工具包，覆盖文本翻译质量、语音质量、说话人与风格保留、时序一致性和流式延迟评测。

## 安装

```bash
pip install OpenSTBench
```

本地开发：

```bash
pip install -e .
```

可选依赖：

```bash
pip install "OpenSTBench[comet]"
pip install "OpenSTBench[whisper]"
pip install "OpenSTBench[speech_quality]"
pip install "OpenSTBench[emotion]"
pip install "OpenSTBench[paralinguistics]"
pip install "OpenSTBench[all]"
```

BLEURT 需要单独安装：

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## 包名

- PyPI 包名：`OpenSTBench`
- Python 导入名：`openstbench`

## 评测器

| 评测器 | 用途 | 主要输出 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | MT 与 S2TT 文本质量 | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | 生成语音质量与文本一致性 | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | 说话人保留 | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | 情感保留或情感分类 | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | 非语言与副语言信息保留 | `Paralinguistic_Fidelity_Cosine`, `Acoustic_Event_Count_F1`, `Acoustic_Event_Localization_F1` |
| `TemporalConsistencyEvaluator` | 源语音与目标语音的时长一致性 | `Duration_Consistency_SLC_*`、时长诊断信息 |
| `LatencyEvaluator` | 流式与同传语音翻译延迟 | `StartOffset`, `ATD`, `CustomATD`, `RTF` |

## 示例

示例代码统一放在 `examples/`。

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

延迟评测也可以通过模块 CLI 运行：

```bash
python -m openstbench.latency.cli --help
```

## 约定

- 文本输入通常支持 `list[str]`、每行一个样本的 `.txt` 文件和 `.json` 文件。
- 音频输入通常支持文件夹、`list[str]`、`.txt` 路径列表和 `.json` 路径列表。
- 对 `zh`、`ja`、`ko`，文本侧评测使用 CJK 友好处理；语音一致性返回 `CER_Consistency`，而不是 `WER_Consistency`。
- 模型路径参数采用本地优先规则：如果传入的本地路径存在，就使用本地路径；否则回退到配置的远端模型 id。
- 可选依赖只会在对应评测器需要时加载。

