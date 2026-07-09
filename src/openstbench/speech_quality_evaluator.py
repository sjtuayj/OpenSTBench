import os
import torch
import torchaudio
import jiwer
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm

# Reuse existing loading utilities
from ._model_loading import resolve_pretrained_source
from .language_policy import normalize_language_code, speech_consistency_unit, whisper_language_code
from .translation_evaluator import load_text_from_file_or_list, load_audio_from_folder

try:
    import whisper
except ImportError:
    whisper = None

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None

class SpeechQualityEvaluator:
    DEFAULT_WHISPER_MODEL = "medium"
    """
    Speech Quality and Consistency Evaluator.
    Specifically designed for:
    1. UTMOS (Audio naturalness / perceived quality).
    2. WER/CER Consistency Calculation (compares the model's generated text with its generated audio).
    """
    def __init__(self, 
                 use_wer: bool = True,
                 use_utmos: bool = True,
                 whisper_model: str = DEFAULT_WHISPER_MODEL,
                 whisper_language: Optional[str] = None,
                 utmos_model_path: Optional[str] = None,
                 utmos_ckpt_path: Optional[str] = None,
                 device: Optional[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wer = use_wer
        self.use_utmos = use_utmos
        
        self.whisper_model_name = whisper_model
        self.whisper_language = self._normalize_whisper_language(whisper_language)
        self.utmos_path = utmos_model_path
        self.utmos_ckpt = utmos_ckpt_path
        self.zh_converter = OpenCC("t2s") if OpenCC is not None else None
        self._warned_missing_opencc = False
        
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

    @staticmethod
    def _normalize_whisper_language(language: Optional[str]) -> Optional[str]:
        return whisper_language_code(language)

    def _load_whisper(self):
        if self.whisper_model is None and self.use_wer:
            if not whisper:
                print("⚠️ Whisper is not installed, skipping load.")
                self.use_wer = False
                return
            print(f"⏳ Loading Whisper ({self.whisper_model_name})...")
            model_source, source_kind = resolve_pretrained_source(
                self.whisper_model_name,
                fallback_source=self.DEFAULT_WHISPER_MODEL,
            )
            print(f"Loading Whisper ({source_kind}) from {model_source}...")
            self.whisper_model = whisper.load_model(model_source, device=self.device)

    def _load_utmos(self):
        if self.utmos_model is None and self.use_utmos:
            print("⏳ Loading UTMOS model...")
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
                print(f"⚠️ UTMOS model failed to load: {e}")
                self.use_utmos = False

    def _transcribe(self, audio_paths: List[str]) -> List[str]:
        self._load_whisper()
        if not self.whisper_model:
            return [""] * len(audio_paths)

        transcribe_kwargs = {
            "fp16": torch.cuda.is_available() and "cuda" in self.device,
            "task": "transcribe",
        }
        if self.whisper_language:
            transcribe_kwargs["language"] = self.whisper_language

        results = []
        for path in tqdm(audio_paths, desc="Whisper transcription"):
            res = self.whisper_model.transcribe(path, **transcribe_kwargs)
            results.append(res["text"].strip())
        return results

    def _compute_utmos(self, audio_paths: List[str]) -> float:
        self._load_utmos()
        if not self.utmos_model: return 0.0
        
        scores = []
        target_sr = 16000
        for path in tqdm(audio_paths, desc="🎧 Computing UTMOS"):
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
        normalized_lang = normalize_language_code(lang)
        text = self.wer_transform(text)
        if normalized_lang == "zh":
            if self.zh_converter is not None:
                text = self.zh_converter.convert(text)
            elif not self._warned_missing_opencc:
                print("Warning: OpenCC is not installed; Chinese CER will not normalize traditional/simplified variants.")
                self._warned_missing_opencc = True
        if speech_consistency_unit(normalized_lang) == "cer":
            text = text.replace(" ", "")
            return " ".join(list(text))
        return text

    def evaluate_all(self, 
                     target_audio: Union[List[str], str],
                     target_text: Optional[Union[List[str], str]] = None,
                     target_lang: str = "en") -> Dict[str, float]:
        """
        :param target_audio: List of paths or directory to the model-generated audio.
        :param target_text: Text generated synchronously by the model, used as reference for WER/CER calculation.
        :param target_lang: Target language for both audio and text.
        """
        results = {}
        print(f"\n--- Starting Speech Quality Evaluation (Target Lang: {target_lang}) ---")
        
        # Load target audio
        if isinstance(target_audio, str) and os.path.exists(target_audio) and os.path.isdir(target_audio):
            audio_paths = load_audio_from_folder(target_audio)
        else:
            audio_paths = target_audio if isinstance(target_audio, list) else [target_audio]

        # 1. evaluate UTMOS
        if self.use_utmos:
            print("   ➤ Computing UTMOS (Audio Naturalness)...")
            results["UTMOS"] = round(self._compute_utmos(audio_paths), 4)

        # 2. evaluate Consistency WER/CER
        if self.use_wer:
            if not target_text:
                print(" ⚠️ target_text (synchronously generated text) not provided. Text-speech consistency cannot be evaluated, skipping WER/CER.")
            else:
                texts = load_text_from_file_or_list(target_text, "target_text")
                if len(texts) != len(audio_paths):
                    raise ValueError(f"Number of generated text lines ({len(texts)}) does not match number of audio files ({len(audio_paths)})!")
                
                print("   ➤ Transcribing audio, preparing consistency calculation...")
                asr_texts = self._transcribe(audio_paths)
                
                clean_refs = [self._preprocess_for_wer(t, target_lang) for t in texts]
                clean_hyps = [self._preprocess_for_wer(t, target_lang) for t in asr_texts]
                
                if clean_refs:
                    error_rate = jiwer.wer(clean_refs, clean_hyps)
                    metric_name = "CER_Consistency" if speech_consistency_unit(target_lang) == "cer" else "WER_Consistency"
                    results[metric_name] = round(error_rate, 4)
                else:
                    print("   ⚠️ The provided generated text is empty after cleaning, cannot compute distribution error rate.") 

        return results
