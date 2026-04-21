import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from numpy.linalg import norm

def _load_data_list(
    input_data: Union[str, List[str]], 
    name: str,
    target_type: str = "audio"
) -> List[str]:
    if isinstance(input_data, list):
        return input_data
    if not isinstance(input_data, str):
        raise ValueError(f"{name} 必须是 文件路径(str) 或 列表(List[str])")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} 文件不存在: {input_data}")
    
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if target_type == "audio":
            candidate_keys = ["audio", "path", "file", "wav", "mp3"]
        elif target_type == "label":
            candidate_keys = ["label", "emotion", "class", "reference"]
        else:
            candidate_keys = ["text", "sentence", "content", "transcript"]

        if isinstance(data, list):
            if not data: return []
            if isinstance(data[0], str): return data
            if isinstance(data[0], dict):
                for key in candidate_keys:
                    if key in data[0]:
                        return [item[key] for item in data]
                raise ValueError(f"JSON 列表项中未找到常见{target_type}字段: {candidate_keys}")

        if isinstance(data, dict):
            plural_candidates = [k + "s" for k in candidate_keys]
            plural_candidates += ["target_text", "hypothesis", "source_text"] if target_type == "text" else []
            for key in plural_candidates:
                if key in data:
                    return data[key]
            raise ValueError(f"JSON 字典中未找到常见{target_type}列表字段")
        raise ValueError("不支持的 JSON 格式")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def _load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    audio_files = sorted(audio_files, key=lambda x: x.stem)
    if not audio_files:
        raise ValueError(f"文件夹中没有音频文件: {folder_path}")
    return [str(f) for f in audio_files]


class EmotionEvaluator:
    """
    基于 Emotion2Vec+ 的情感评测器 
    支持：
    [1] 跨语种保真度评测：基于 768-d 高维嵌入特征的 Cosine Similarity 保真度。
    [2] 离散情感分类评测：基于模型 Zero-Shot 分类能力的识别准确度评测。
    """
    
    # 唯一依赖的基座模型
    DEFAULT_E2V_MODEL = "iic/emotion2vec_plus_large"

    def __init__(self, 
                 e2v_model_path: Optional[str] = None,
                 custom_label_map: Optional[Dict[str, str]] = None,
                 device: Optional[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.e2v_model_path = e2v_model_path or self.DEFAULT_E2V_MODEL
        self.e2v_model = None
        
        # 将用户的自定义标签映射统一全部小写，确保健壮性
        self.custom_label_map = {k.lower(): v.lower() for k, v in (custom_label_map or {}).items()}

    def _load_e2v_model(self):
        if self.e2v_model is not None: return
        print(f"⏳ 正在加载 Emotion2Vec+ 大模型: {self.e2v_model_path}")
        try:
            from funasr import AutoModel
            self.e2v_model = AutoModel(model=self.e2v_model_path, device=self.device, disable_update=True)
            print("✅ Emotion2Vec+ 大模型加载成功！")
        except ImportError:
            print("❌ Emotion2Vec+ 依赖缺失。请执行: pip install funasr modelscope")
        except Exception as e:
            print(f"❌ Emotion2Vec+ 分类模型加载失败: {e}")

    # =============== 特征提取部分 ===============

    def _extract_e2v_embeddings(self, audio_paths: List[str]) -> List[np.ndarray]:
        """提取 Emotion2Vec+ 的综合高维特征 Embedding 用于衡量总体保真度相似度"""
        self._load_e2v_model()
        if not self.e2v_model: return [np.zeros(768)] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 音频 E2V 高维特征", unit="file"):
            try:
                res = self.e2v_model.generate(path, output_dir=None, granularity="utterance", extract_embedding=True)
                if isinstance(res, list) and len(res) > 0 and 'feats' in res[0]:
                    emb = np.array(res[0]['feats']).squeeze()
                    results.append(emb)
                else:
                    results.append(np.zeros(768))
            except Exception as e:
                results.append(np.zeros(768))
        return results

    def _extract_cls_emotion(self, audio_paths: List[str]) -> List[str]:
        """提取 Emotion2Vec+ 的离散情感分类预测结果"""
        self._load_e2v_model()
        if not self.e2v_model: return ["unknown"] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 音频 E2V 离散分类标签 예측", unit="file"):
            try:
                # 默认分类模式，返回具体的 label
                res = self.e2v_model.generate(path, output_dir=None, granularity="utterance", extract_embedding=False)
                if isinstance(res, list) and len(res) > 0 and 'labels' in res[0] and 'scores' in res[0]:
                    labels = res[0]['labels']
                    scores = res[0]['scores']
                    
                    # 1. 找到得分最高的标签索引
                    best_idx = np.argmax(scores)
                    best_label_str = labels[best_idx]
                    
                    # 2. 解析出英文部分 (因为输出格式是 '生气/angry')
                    if "/" in best_label_str:
                        label = best_label_str.split("/")[-1].lower()
                    else:
                        label = best_label_str.lower()
                        
                    # 3. 清洗可能存在的特殊占位符
                    label = label.replace("<|", "").replace("|>", "").strip()
                else:
                    label = "unknown"
                
                # 应用自定义对齐
                if label in self.custom_label_map:
                    label = self.custom_label_map[label]
                results.append(label)
            except Exception as e:
                results.append("unknown")
        return results

    # =============== 评测主入口 ===============

    def evaluate_all(self, 
                     source_audio: Optional[Union[List[str], str]] = None, 
                     target_audio: Optional[Union[List[str], str]] = None,
                     reference_labels: Optional[Union[List[str], str]] = None,
                     verbose: bool = True) -> Dict[str, float]:
        """
        动态融合引擎入口：
        - 仅传 target_audio 与 reference_labels -> 计算 Accuracy 
        - 仅传 source_audio 与 target_audio -> 计算 E2V 特征保真度
        """
        
        if target_audio is None:
            if source_audio is not None and reference_labels is not None:
                target_audio = source_audio
                source_audio = None
            else:
                raise ValueError("🚨 必须提供 target_audio 参数运行评测。")

        if isinstance(target_audio, str) and os.path.isdir(target_audio):
            tgt_paths = _load_audio_from_folder(target_audio)
        else:
            tgt_paths = _load_data_list(target_audio, "Target Audio Paths", "audio")
        
        num_samples = len(tgt_paths)
        if num_samples == 0:
            raise ValueError("没有找到目标音频数据，评测停止。")

        run_fidelity = source_audio is not None
        run_classification = reference_labels is not None
        results = {}

        # ==================== 分支 1：保真度相似度计算 ====================
        if run_fidelity:
            if verbose: print(f"\n📝 启动【跨模态综合保真度运算】 ({num_samples} 个 Source-Target 音频距离比对)...")
            
            if isinstance(source_audio, str) and os.path.isdir(source_audio):
                src_paths = _load_audio_from_folder(source_audio)
            else:
                src_paths = _load_data_list(source_audio, "Source Audio Paths", "audio")

            if len(src_paths) != num_samples:
                raise ValueError(f"数目不一致: Source ({len(src_paths)}) != Target ({num_samples})")

            src_e2v_embs = self._extract_e2v_embeddings(src_paths)
            tgt_e2v_embs = self._extract_e2v_embeddings(tgt_paths)

            e2v_cosine_sim_total = 0.0 
            valid_e2v_count = 0
            
            for i in range(num_samples):
                s_emb = src_e2v_embs[i]
                t_emb = tgt_e2v_embs[i]
                
                n_s = norm(s_emb)
                n_t = norm(t_emb)
                if n_s > 0 and n_t > 0:
                    sim = np.dot(s_emb, t_emb) / (n_s * n_t)
                    e2v_cosine_sim_total += float(sim)
                    valid_e2v_count += 1

            final_cosine = (e2v_cosine_sim_total / valid_e2v_count) if valid_e2v_count > 0 else 0.0
            results["Emotion2Vec_Cosine_Similarity"] = round(final_cosine, 4)


        # ==================== 分支 2：离散情感识别计算 ====================
        if run_classification:
            if verbose: print(f"\n📝 启动【Emotion2Vec+ 离散分类准确率运算】 ({num_samples} 个特征识别与金标准比对)...")
            refs = _load_data_list(reference_labels, "Reference Labels", "label")
            
            if len(refs) != num_samples:
                raise ValueError(f"参考标签数量 ({len(refs)}) 与目标音频数量 ({num_samples}) 不匹配！")

            preds = self._extract_cls_emotion(tgt_paths)
            
            correct = 0
            for p, r in zip(preds, refs):
                if p.strip() == r.strip().lower():
                    correct += 1
            
            acc = correct / len(refs) if len(refs) > 0 else 0.0
            results["Audio_Emotion_Accuracy"] = round(acc, 4)


        # ==================== 输出反馈 ====================
        if verbose:
            print("\n📊 [EmotionEvaluator] Emotion2Vec+ 综合评测报告:")
            print(f"   - 有效评测样本量: {num_samples}条")
            
            if run_fidelity:
                print("\n   [深度学习综合特征保真度] (数值范围[-1, 1]，越接近 1.00 代表整体情感越相似)")
                print(f"   - Emotion2Vec+ 综合情感特征余弦相似度:         {results['Emotion2Vec_Cosine_Similarity']:.4f}")
            
            if run_classification:
                print("\n   [深度学习离散情感识别分类率] (数值越高代表预设标签识别越准)")
                print(f"   - Audio Emotion Accuracy (离散情感识别准确率): {results['Audio_Emotion_Accuracy']:.2%}")
                
        return results