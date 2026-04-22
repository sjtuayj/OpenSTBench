import torch
import torchaudio
import numpy as np
import os
import librosa
from tqdm import tqdm

from ._model_loading import resolve_pretrained_source

DEFAULT_WAVLM_MODEL_SOURCE = "microsoft/wavlm-base-plus-sv"

# 如果你想用本地的 Resemblyzer，确保之前项目能 import 到
try:
    from resemblyzer import preprocess_wav, VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SpeakerSimilarityEvaluator:
    """
    评估提取两段音频说话人特征（Speaker Embeddings）并计算余弦相似度的打分器。
    """
    def __init__(self, 
                 model_type="wavlm", 
                 device=None, 
                 wavlm_model_path=DEFAULT_WAVLM_MODEL_SOURCE,
                 resemblyzer_weights_path="pretrained.pt"):
        """
        初始化打分器，仅在初始化时加载一次模型。
        
        Args:
            model_type (str): "wavlm" 或 "resemblyzer" 或 "both"。
            device (str): "cuda", "mps" 或 "cpu"，如果为 None 则自动检测。
            wavlm_model_path (str): WavLM 模型的路径或 HuggingFace ID。
            resemblyzer_weights_path (str): Resemblyzer 的 pretrained.pt 权重路径。
        """
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_type = model_type.lower()
        print(f"Initializing SpeakerSimilarityEvaluator with model(s): {self.model_type} on {self.device}")

        self.wavlm_feature_extractor = None
        self.wavlm_model = None
        self.resemblyzer_encoder = None

        if self.model_type in ["wavlm", "both"]:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Please install transformers to use WavLM: pip install transformers")
            model_source, source_kind = resolve_pretrained_source(
                wavlm_model_path,
                fallback_source=DEFAULT_WAVLM_MODEL_SOURCE,
            )
            print(f"Loading WavLM ({source_kind}) from {model_source}...")
            self.wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_source)
            self.wavlm_model = WavLMForXVector.from_pretrained(model_source).to(self.device).eval()

        if self.model_type in ["resemblyzer", "both"]:
            if not RESEMBLYZER_AVAILABLE:
                raise ImportError("Please ensure 'resemblyzer' module is available in your Python path.")
            print("Loading Resemblyzer VoiceEncoder...")
            # 兼容绝对路径或默认行为
            if os.path.exists(resemblyzer_weights_path):
                self.resemblyzer_encoder = VoiceEncoder(weights_fpath=resemblyzer_weights_path)
            else:
                self.resemblyzer_encoder = VoiceEncoder() # 内部会自动处理没传 path 的默认下载

    @torch.no_grad()
    def evaluate(self, ref_wav_path, synth_wav_path):
        """
        计算单对音频的相似度得分。
        
        Args:
            ref_wav_path (str): 参考音频路径（Prompt / Target）
            synth_wav_path (str): 你模型合成出来的音频路径
            
        Returns:
            dict: 包含计算结果的字典，例如 {"wavlm": 0.85, "resemblyzer": 0.91}
        """
        results = {}

        if self.model_type in ["wavlm", "both"]:
            # WavLM 强制要求 16000 采样率
            ref_waves_16k, _ = librosa.load(ref_wav_path, sr=16000)
            synth_waves_16k, _ = librosa.load(synth_wav_path, sr=16000)

            ref_inputs = self.wavlm_feature_extractor(
                torch.tensor(ref_waves_16k), padding=True, return_tensors="pt", sampling_rate=16000
            ).to(self.device)
            synth_inputs = self.wavlm_feature_extractor(
                torch.tensor(synth_waves_16k), padding=True, return_tensors="pt", sampling_rate=16000
            ).to(self.device)

            ref_embeddings = self.wavlm_model(**ref_inputs).embeddings
            synth_embeddings = self.wavlm_model(**synth_inputs).embeddings

            # L2 归一化后计算余弦相似度
            ref_embeddings = torch.nn.functional.normalize(ref_embeddings, dim=-1)
            synth_embeddings = torch.nn.functional.normalize(synth_embeddings, dim=-1)
            similarity_wavlm = torch.nn.functional.cosine_similarity(ref_embeddings, synth_embeddings, dim=-1)
            
            results["wavlm_similarity"] = similarity_wavlm.item()

        if self.model_type in ["resemblyzer", "both"]:
            ref_wav_res = preprocess_wav(ref_wav_path)
            synth_wav_res = preprocess_wav(synth_wav_path)

            ref_embed_res = self.resemblyzer_encoder.embed_utterance(ref_wav_res)
            synth_embed_res = self.resemblyzer_encoder.embed_utterance(synth_wav_res)
            
            similarity_res = np.inner(ref_embed_res, synth_embed_res)
            results["resemblyzer_similarity"] = float(similarity_res)

        return results

    def evaluate_batch(self, ref_wav_paths, synth_wav_paths):
        """
        批量计算文件夹下的多对音频相似度。
        
        Args:
            ref_wav_paths (list of str): 参考音频列表
            synth_wav_paths (list of str): 合成音频列表（必须与参考列表一一对应）
            
        Returns:
            dict: 包含各个模型平均分的字典
        """
        assert len(ref_wav_paths) == len(synth_wav_paths), "参考音频与合成音频数量必须对应！"
        
        batch_results = {"details": []}
        wavlm_scores = []
        res_scores = []

        print("Evaluating speaker similarity...")
        for ref_p, syn_p in tqdm(zip(ref_wav_paths, synth_wav_paths), total=len(ref_wav_paths)):
            res = self.evaluate(ref_p, syn_p)
            batch_results["details"].append({"ref": ref_p, "synth": syn_p, "score": res})
            
            if "wavlm_similarity" in res:
                wavlm_scores.append(res["wavlm_similarity"])
            if "resemblyzer_similarity" in res:
                res_scores.append(res["resemblyzer_similarity"])

        if wavlm_scores:
            batch_results["average_wavlm_similarity"] = sum(wavlm_scores) / len(wavlm_scores)
        if res_scores:
            batch_results["average_resemblyzer_similarity"] = sum(res_scores) / len(res_scores)
            
        return batch_results
