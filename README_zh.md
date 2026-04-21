# OpenSTBench

[English](./README.md) | 中文

[![PyPI version](https://badge.fury.io/py/OpenSTBench.svg)](https://pypi.org/project/OpenSTBench/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenSTBench 是一个面向机器翻译和语音翻译的评测工具包，提供统一的文本质量、语音质量、保留性指标和流式延迟评测接口。

## 适用方向

- MT 或 S2TT 文本侧评测：`BLEU`、`chrF++`、`COMET`、`BLEURT`
- S2ST 综合评测：文本质量、语音质量、说话人相似度、延迟
- 流式或同传语音翻译延迟评测
- 语音翻译输出的保留性分析，包括说话人、情感和副语言信息

## 核心模块

| 模块 | 主要用途 | 常见指标 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | 文本翻译质量 | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | 语音自然度和文本语音一致性 | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | 说话人保留 | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | 情感保留或分类准确率 | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | 副语言和非言语信息保留 | `Paralinguistic_Fidelity_Cosine`, `Acoustic_Event_Preservation_Rate`, `Acoustic_Event_Preservation_Macro_F1`, `Acoustic_Event_Preservation_Macro_Recall`, `Event_Aligned_Preservation_Rate`, `Conditional_Relative_Onset_Error` |
| `LatencyEvaluator` | 流式 / 同传延迟评测 | `StartOffset`, `ATD`, `CustomATD`, `RTF`, `Model_Generate_RTF` |

## 安装

基础安装：

```bash
pip install OpenSTBench
```

可选 extras：

```bash
pip install "OpenSTBench[comet]"
pip install "OpenSTBench[whisper]"
pip install "OpenSTBench[speech_quality]"
pip install "OpenSTBench[emotion]"
pip install "OpenSTBench[paralinguistics]"
pip install "OpenSTBench[all]"
```

如果需要 BLEURT：

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## 导入方式

PyPI 包名：

```python
OpenSTBench
```

Python import 名：

```python
openstbench
```

示例：

```python
from openstbench import TranslationEvaluator, SpeechQualityEvaluator
```

## 快速开始

示例脚本位于 `examples/` 目录。

Python 示例：

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/paralinguistic_identity_baseline.py`
- `examples/python/latency_eval.py`

Shell 示例：

- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`

延迟输出区分两种 RTF：

- `Real_Time_Factor_(RTF)`：系统级 RTF，包含 agent 策略、前后处理等模型推理之外的开销。
- `Model_Generate_RTF`：模型级 RTF，仅在 agent 显式记录模型推理时间时返回。

## 输入约定

常见文本输入支持：

- Python `List[str]`
- 每行一个样本的 `.txt`
- `.json`

常见音频输入支持：

- 文件夹路径
- Python `List[str]`
- `.txt`
- `.json`

## 说明

- 对 `zh` / `ja` / `ko`，工具包采用 CJK 友好的文本评测处理方式。
- `SpeechQualityEvaluator` 对 `zh` / `ja` / `ko` 返回 `CER_Consistency`，对大多数其他语言返回 `WER_Consistency`。
- `ParalinguisticEvaluator` 始终支持 `Paralinguistic_Fidelity_Cosine`，即基于 CLAP 的源语音与目标语音连续相似度。
- 离散保留性分支是 utterance-level 单标签任务。若提供源侧金标标签，会返回 `Acoustic_Event_Preservation_Rate`、`Acoustic_Event_Preservation_Macro_F1` 和 `Acoustic_Event_Preservation_Macro_Recall`。
- 若提供 `source_onsets_ms`，还可以返回对齐感知指标：`Event_Aligned_Preservation_Rate` 和 `Conditional_Relative_Onset_Error`。
- 对齐基于相对起始位置而非绝对时间，更适合跨语言 S2ST 场景。
- 如果没有目标侧起始时间戳，默认 localizer 会基于 CLAP 滑窗打分估计事件位置。
- 若没有源侧金标标签，评测器仍可在 prediction-only 模式下运行，并返回 `Predicted_Event_Consistency_Rate`、`Predicted_Event_Consistency_Macro_F1` 和 `Predicted_Event_Consistency_Macro_Recall`。
- 对离线环境，`clap_model_path` 可以传 Hugging Face repo id，也可以传本地模型目录或 snapshot。
- 在 S2S 延迟评测中，若模型本身不产出文本，评测器可选用 ASR fallback 生成对齐文本。
- 某些模块依赖可选依赖项或离线模型路径。

## License

MIT License
