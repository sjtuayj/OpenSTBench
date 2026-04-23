# OpenSTBench

[English](./README.md) | 中文

[![PyPI version](https://badge.fury.io/py/OpenSTBench.svg)](https://pypi.org/project/OpenSTBench/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenSTBench 是一个以翻译和语音翻译为核心的评测工具包。它提供了一套统一的方式，用于评估文本翻译质量、语音输出质量、各类保留属性以及流式延迟。

## 适用方向

这个项目特别适合以下场景：

- MT 或 S2TT 的文本侧评测，包括 `BLEU`、`chrF++`、`COMET` 和 `BLEURT`
- S2ST 综合评测，将文本质量、语音质量、说话人相似度和延迟结合起来
- 使用自定义 agent 的流式或同传语音翻译延迟评测
- 对语音翻译输出进行保留属性分析，包括说话人、情感和副语言相似性
- 对语音翻译或配音输出进行时序一致性分析，包括时长符合度和时长误差

## 核心模块

| 模块 | 主要用途 | 常见指标 |
| :--- | :--- | :--- |
| `TranslationEvaluator` | 文本翻译质量 | `sacreBLEU`, `chrF++`, `COMET`, `BLEURT` |
| `SpeechQualityEvaluator` | 语音自然度和文本语音一致性 | `UTMOS`, `WER_Consistency`, `CER_Consistency` |
| `SpeakerSimilarityEvaluator` | 说话人保留 | `wavlm_similarity`, `resemblyzer_similarity` |
| `EmotionEvaluator` | 情感保留或分类准确率 | `Emotion2Vec_Cosine_Similarity`, `Audio_Emotion_Accuracy` |
| `ParalinguisticEvaluator` | 非语言和副语言信息保留 | `Paralinguistic_Fidelity_Cosine`, `Acoustic_Event_Preservation_Rate`, `Acoustic_Event_Preservation_Macro_F1`, `Acoustic_Event_Preservation_Macro_Recall`, `Event_Aligned_Preservation_Rate`, `Conditional_Relative_Onset_Error` |
| `TemporalConsistencyEvaluator` | 源语音与目标语音的时长结构一致性 | `Duration_Consistency_SLC_0.2`, `Duration_Consistency_SLC_0.4` |
| `LatencyEvaluator` | 流式 / 同传翻译延迟 | `StartOffset`, `ATD`, `CustomATD`, `RTF`, `Model_Generate_RTF` |

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

如果你需要 BLEURT：

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

## 导入方式

PyPI 包名：

```python
OpenSTBench
```

Python 导入名：

```python
openstbench
```

示例：

```python
from openstbench import (
    TranslationEvaluator,
    SpeechQualityEvaluator,
    TemporalConsistencyEvaluator,
)
```

## 快速开始

快速开始脚本位于 `examples/` 目录下。

Python 示例：

- `examples/python/translation_eval.py`
- `examples/python/speech_quality_eval.py`
- `examples/python/speaker_similarity_eval.py`
- `examples/python/emotion_eval.py`
- `examples/python/paralinguistic_eval.py`
- `examples/python/paralinguistic_identity_baseline.py`
- `examples/python/temporal_consistency_eval.py`
- `examples/python/latency_eval.py`

Shell 示例：

- `examples/bash/install_extras.sh`
- `examples/bash/run_latency_cli.sh`

最小的时序一致性示例：

```python
from openstbench import TemporalConsistencyEvaluator

evaluator = TemporalConsistencyEvaluator(
    thresholds=(0.2, 0.4),
)

results, diagnostics = evaluator.evaluate_all(
    source_audio="./source_wavs",
    target_audio="./generated_wavs",
    sample_ids=["sample_1", "sample_2"],
    return_diagnostics=True,
)
```

延迟输出区分两种 RTF：

- `Real_Time_Factor_(RTF)`：系统级 RTF。它包含 agent 策略开销、前后处理以及模型推理之外的其他运行时成本。
- `Model_Generate_RTF`：模型级 RTF。仅当 agent 通过 `record_model_inference_time(...)` 显式记录模型推理时间，或在 `Segment.config["model_inference_time"]` 中返回该值时才会报告。

## 输入约定

常见文本输入支持：

- Python `List[str]`
- 每行一个样本的 `.txt` 文件
- `.json` 文件

常见音频输入支持：

- 文件夹路径
- Python `List[str]`
- `.txt` 文件
- `.json` 文件

## 说明

- 对于 `zh` / `ja` / `ko`，工具包会在文本侧评测中采用 CJK 友好的处理方式。
- `SpeechQualityEvaluator` 对 `zh` / `ja` / `ko` 返回 `CER_Consistency`，对大多数其他语言返回 `WER_Consistency`。
- `ParalinguisticEvaluator` 始终支持 `Paralinguistic_Fidelity_Cosine`，即基于 CLAP 的源语音与目标语音连续相似度分数。
- `TemporalConsistencyEvaluator` 对 `source_audio` 和 `target_audio` 都支持 `List[str]`、音频文件夹、`.txt` 路径列表和 `.json` 路径列表。
- `TemporalConsistencyEvaluator` 会报告阈值式时长符合度指标（`Duration_Consistency_SLC_*`）。
- 离散保留分支是一个 utterance-level 的单标签任务。若提供源侧金标准标签，它会返回 `Acoustic_Event_Preservation_Rate`、`Acoustic_Event_Preservation_Macro_F1` 和 `Acoustic_Event_Preservation_Macro_Recall`。
- 如果提供了 `source_onsets_ms`，评测器还可以报告对齐感知指标：`Event_Aligned_Preservation_Rate` 和 `Conditional_Relative_Onset_Error`。
- 对齐基于相对起始位置，而不是绝对墙钟时间。这使它更适合跨语种 S2ST，因为源语音和目标语音的时长天然可能不同。
- 如果没有目标侧起始时间戳，默认 localizer 会基于目标事件标签，使用 CLAP 滑窗打分来估计时间位置。
- 这些对齐指标应被理解为弱监督、粗粒度的对齐信号，而不是精确到时间戳的事件定位基准。
- 如果没有源侧金标准标签，评测器仍然可以在 prediction-only 模式下运行，并报告 `Predicted_Event_Consistency_Rate`、`Predicted_Event_Consistency_Macro_F1` 和 `Predicted_Event_Consistency_Macro_Recall`。
- 默认的离散预测器是一个基于 `candidate_labels` 的封闭集合 CLAP 分类器。用户也可以替换成任何实现了 `predict(audio_paths, candidate_labels)` 的自定义预测器对象。
- 默认事件定位器也可以替换。自定义 localizer 只需要实现 `localize(audio_paths, labels, candidate_labels)`。
- 数据集特定的标签映射被有意放在核心包之外。请在调用时传入 `candidate_labels` 和 `label_normalizer`，这样同一个评测器就可以在不修改核心代码的情况下适配不同数据集。
- 对于离线环境，`clap_model_path` 可以传 Hugging Face repo id，也可以传本地模型目录或 snapshot。
- `clap_model_path`、`wavlm_model_path`、`whisper_model`、`e2v_model_path`、`comet_model` 和 `bleurt_path` 等模型加载参数现在统一采用“本地优先”规则：如果传入的本地路径存在，就优先使用本地；否则回退到默认远端模型 id。
- 在 S2S 延迟评测中，如果模型本身可提供原生转写，系统会优先使用它来做对齐；如果模型只输出音频，评测器可以选择使用 ASR fallback 来准备对齐文本。
- 对于 S2S forced alignment，请通过 `alignment_acoustic_model` 和 `alignment_dictionary_model` 传入与语言匹配的 MFA 模型。默认值是英文。
- 某些模块依赖可选依赖项，或者在离线环境中依赖本地模型路径。

## License

MIT License
