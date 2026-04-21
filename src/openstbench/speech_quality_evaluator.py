import os
import torch
import torchaudio
import jiwer
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm

# 复用现有的加载工具
from .translation_evaluator import load_text_from_file_or_list, load_audio_from_folder

try:
    import whisper
except ImportError:
    whisper = None

class SpeechQualityEvaluator:
    """
    语音质量与一致性评测器
    专门用于：
    1. UTMOS (音频自然度/听感质量)
    2. WER/CER 一致性计算 (使用模型自己生成的文本与其生成的音频进行比对)
    """
    def __init__(self, 
                 use_wer: bool = True,
                 use_utmos: bool = True,
                 whisper_model: str = "medium",
                 utmos_model_path: Optional[str] = None,
                 utmos_ckpt_path: Optional[str] = None,
                 device: Optional[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wer = use_wer
        self.use_utmos = use_utmos
        
        self.whisper_model_name = whisper_model
        self.utmos_path = utmos_model_path
        self.utmos_ckpt = utmos_ckpt_path
        
        self.whisper_model = None
        self.utmos_model = None
        
        if self.use_wer:
            self.wer_transform = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemoveEmptyStrings(),
            ])

    def _load_whisper(self):
        if self.whisper_model is None and self.use_wer:
            if not whisper:
                print("⚠️ Whisper 未安装，跳过加载")
                self.use_wer = False
                return
            print(f"⏳ 正在加载 Whisper ({self.whisper_model_name})...")
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)

    def _load_utmos(self):
        if self.utmos_model is None and self.use_utmos:
            print("⏳ 正在加载 UTMOS 模型...")
            try:
                source = "github"
                repo_or_dir = "tarepan/SpeechMOS"
                if self.utmos_path and os.path.exists(self.utmos_path):
                    source = "local"
                    repo_or_dir = self.utmos_path

                load_pretrained = (self.utmos_ckpt is None)
                self.utmos_model = torch.hub.load(
                    repo_or_dir, "utmos22_strong", source=source, trust_repo=True, pretrained=load_pretrained
                )
                
                if self.utmos_ckpt and os.path.exists(self.utmos_ckpt):
                    state_dict = torch.load(self.utmos_ckpt, map_location=self.device)
                    self.utmos_model.load_state_dict(state_dict)
                    
                self.utmos_model.to(self.device).eval()
            except Exception as e:
                print(f"⚠️ UTMOS 模型加载失败: {e}")
                self.use_utmos = False

    def _transcribe(self, audio_paths: List[str]) -> List[str]:
        self._load_whisper()
        if not self.whisper_model:
            return [""] * len(audio_paths)
            
        results = []
        for path in tqdm(audio_paths, desc="🎙️ Whisper 转写中"):
            res = self.whisper_model.transcribe(path, fp16=torch.cuda.is_available() and "cuda" in self.device)
            results.append(res["text"].strip())
        return results

    def _compute_utmos(self, audio_paths: List[str]) -> float:
        self._load_utmos()
        if not self.utmos_model: return 0.0
        
        scores = []
        target_sr = 16000
        for path in tqdm(audio_paths, desc="🎧 计算 UTMOS"):
            try:
                wave, sr = torchaudio.load(path)
                if sr != target_sr:
                    wave = torchaudio.functional.resample(wave, sr, target_sr)
                if wave.shape[0] > 1:
                    wave = torch.mean(wave, dim=0, keepdim=True)
                
                wave = wave.to(self.device)
                with torch.no_grad():
                    s = self.utmos_model(wave, target_sr)
                    scores.append(s.item())
            except Exception as e:
                print(f"⚠️ UTMOS Error on {path}: {e}")
                
        return sum(scores) / len(scores) if scores else 0.0

    def _preprocess_for_wer(self, text: str, lang: str) -> str:
        text = self.wer_transform(text)
        if lang in ["zh", "ja", "ko"]:
            text = text.replace(" ", "")
            return " ".join(list(text))
        return text

    def evaluate_all(self, 
                     target_audio: Union[List[str], str],
                     target_text: Optional[Union[List[str], str]] = None,
                     target_lang: str = "en") -> Dict[str, float]:
        """
        :param target_audio: 模型生成的音频路径列表或文件夹
        :param target_text: 模型同步生成的文本，用作 WER 计算的参照
        :param target_lang: 音频与文本的目标语言
        """
        results = {}
        print(f"\n--- 开始语音质量评测 (Target Lang: {target_lang}) ---")
        
        # 加载目标音频
        if isinstance(target_audio, str) and os.path.exists(target_audio) and os.path.isdir(target_audio):
            audio_paths = load_audio_from_folder(target_audio)
        else:
            audio_paths = target_audio if isinstance(target_audio, list) else [target_audio]

        # 1. 计算 UTMOS
        if self.use_utmos:
            print("   ➤ 计算 UTMOS (音频自然度)...")
            results["UTMOS"] = round(self._compute_utmos(audio_paths), 4)

        # 2. 计算 Consistency WER/CER
        if self.use_wer:
            if not target_text:
                print("   ⚠️ 未提供 target_text (模型同时生成的文本)，无法计算文本-语音一致性，跳过 WER/CER。")
            else:
                texts = load_text_from_file_or_list(target_text, "target_text")
                if len(texts) != len(audio_paths):
                    raise ValueError(f"生成的文本行数 ({len(texts)}) 与 生成的音频数量 ({len(audio_paths)}) 不一致！")
                
                print("   ➤ 正在进行音频转写，准备计算一致性...")
                asr_texts = self._transcribe(audio_paths)
                
                clean_refs = [self._preprocess_for_wer(t, target_lang) for t in texts]
                clean_hyps = [self._preprocess_for_wer(t, target_lang) for t in asr_texts]
                
                if clean_refs:
                    error_rate = jiwer.wer(clean_refs, clean_hyps)
                    metric_name = "CER_Consistency" if target_lang in ['zh', 'ja', 'ko'] else "WER_Consistency"
                    results[metric_name] = round(error_rate, 4)
                else:
                    print("   ⚠️ 提供的生成文本清洗后为空，无法计算分布错误率。")

        return results