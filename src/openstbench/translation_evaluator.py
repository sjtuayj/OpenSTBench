import os
import gc
import json
import numpy as np
import sacrebleu
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path

from ._model_loading import resolve_pretrained_source

# ==================== Configuration ====================

CACHE_PATHS = {
    "huggingface": os.path.expanduser("~/.cache/huggingface/hub"),
}

# ==================== Optional Dependencies ====================

try:
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    HAS_BLEURT = True
except ImportError:
    HAS_BLEURT = False

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    download_model = None
    load_from_checkpoint = None

# ==================== Input Loaders ====================

def load_text_from_file_or_list(input_data: Union[str, List[str]], name: str = "text") -> List[str]:

    if isinstance(input_data, list):
        return input_data
    
    if not isinstance(input_data, str):
        raise ValueError(f"{name} must be a file path (str) or a list of strings (List[str])")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {input_data}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if len(data) == 0: return []
            if isinstance(data[0], str): return data
            if isinstance(data[0], dict):
                for key in ["target_text", "hypothesis", "text", "ref", "reference", "src", "source"]:
                    if key in data[0]:
                        return [item[key] for item in data]
                raise ValueError(f"JSON list items do not contain common text fields")

        if isinstance(data, dict):
            for key in ["target_text", "hypothesis", "text", "ref", "reference", "src", "source"]:
                if key in data:
                    return data[key]
            raise ValueError("JSON dictionary does not contain common text fields")
            
        raise ValueError("Unsupported JSON format")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

def load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    audio_files = sorted(audio_files, key=lambda x: x.stem)
    if not audio_files:
        raise ValueError(f"Folder contains no audio files: {folder_path}")
    return [str(f) for f in audio_files]


# ==================== Evaluator Core Class ====================

DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"
DEFAULT_BLEURT_MODEL = "lucadiliello/BLEURT-20"

class TranslationEvaluator:
    """
    Text-side Translation Quality Evaluator: Supports Text Translation Extraction and Comparison (BLEU, COMET, BLEURT...)
    """

    def __init__(self, 
                 use_bleu: bool = True,
                 use_chrf: bool = True,
                 use_comet: bool = True,      
                 use_bleurt: bool = False,    
                 comet_model: str = DEFAULT_COMET_MODEL,
                 bleurt_path: Optional[str] = None,
                 bleurt_model: Optional[str] = None,
                 device: Optional[str] = None):
        
        self.use_bleu = use_bleu
        self.use_chrf = use_chrf
        self.use_comet = use_comet
        self.use_bleurt = use_bleurt

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"🚀 Initializing Translation Evaluator (Text-only) on {self.device}...")

        # Load language models
        self.comet = None
        if self.use_comet:
            self.comet = self._load_comet(comet_model)

        self.bleurt_model = None
        self.bleurt_tokenizer = None
        if self.use_bleurt:
            self._load_bleurt(bleurt_path, bleurt_model)

        print("✅ Translation text evaluation metrics system is ready!")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.cleanup()
        return False

    def cleanup(self):
        for attr in ['comet', 'bleurt_model', 'bleurt_tokenizer']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _resolve_local_comet_checkpoint(self, model_name: str) -> Optional[str]:
        candidate = Path(model_name).expanduser()
        if not candidate.exists():
            return None

        if candidate.is_file():
            return str(candidate.resolve())

        common_ckpts = [
            candidate / "checkpoints" / "model.ckpt",
            candidate / "model.ckpt",
        ]
        for ckpt in common_ckpts:
            if ckpt.exists() and ckpt.is_file():
                return str(ckpt.resolve())
        return None

    def _load_comet(self, model_name: str):
        if not download_model:
            print("⚠️ COMET is not installed, skipping...")
            return None
        try:
            model_source, source_kind = resolve_pretrained_source(
                model_name,
                fallback_source=DEFAULT_COMET_MODEL,
            )
            local_ckpt = self._resolve_local_comet_checkpoint(model_source)
            if local_ckpt is not None:
                print(f"⏳ [Local] Loading COMET: {local_ckpt}")
                model = load_from_checkpoint(local_ckpt)
            else:
                remote_source = model_source if source_kind == "remote" else DEFAULT_COMET_MODEL
                cache = os.path.join(CACHE_PATHS["huggingface"], f"models--{remote_source.replace('/', '--')}")
                status = "[Local]" if os.path.exists(cache) else "[Online]"
                print(f"⏳ {status} Loading COMET: {model_name}")
                print(f"Loading COMET ({status}) from {remote_source}")
                model = load_from_checkpoint(download_model(remote_source))
            if self.device.startswith("cuda"):
                model = model.to(self.device)
            return model
        except Exception as e:
            print(f"❌ COMET loading failed: {e}")
            return None

    def _load_bleurt(self, path: Optional[str], model_name: Optional[str]):
        if not HAS_BLEURT:
            print("⚠️ bleurt-pytorch is not installed, skipping...")
            return
        
        if path:
            model_source, _source_kind = resolve_pretrained_source(
                path,
                fallback_source=model_name or DEFAULT_BLEURT_MODEL,
            )
        else:
            model_source, _source_kind = resolve_pretrained_source(model_name or DEFAULT_BLEURT_MODEL)
        print(f"⏳ Loading BLEURT: {model_source}")
        
        try:
            self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(model_source)
            self.bleurt_model = BleurtForSequenceClassification.from_pretrained(model_source)
            self.bleurt_model = self.bleurt_model.to(self.device).eval()
        except Exception as e:
            print(f"❌ BLEURT loading failed: {e}")


    def _get_bleu_tokenizer_name(self, lang: str) -> str:
        if lang == 'zh': return 'zh'
        elif lang == 'ja': return 'ja-mecab'
        elif lang == 'ko': return 'ko-mecab'
        else: return '13a'

    def _compute_bleurt_score(self, references: List[str], candidates: List[str]) -> float:
        all_scores = []
        batch_size = 32
        for i in range(0, len(references), batch_size):
            br = references[i:i+batch_size]
            bc = candidates[i:i+batch_size]
            with torch.no_grad():
                inputs = self.bleurt_tokenizer(br, bc, padding='longest', truncation=True, max_length=512, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.bleurt_model(**inputs).logits.flatten().tolist()
                all_scores.extend(scores)
        return float(np.mean(all_scores))

    def evaluate_all(
        self,
        reference: Union[List[str], str],
        target_text: Union[List[str], str],
        source: Optional[Union[List[str], str]] = None,
        target_lang: str = "en"
    ) -> Dict[str, float]:
        """
        Args:
            reference: Ground truth benchmark translation reference from the dataset (List or file path).
            target_text: Direct text translation output from the translation model (List or file path).
            source: Optional original source text (mandatory for COMET).
            target_lang: Target language, affects BLEU Tokenizer selection.
        """
        results = {}
        print(f"\n--- Starting Text Translation Quality Evaluation (Target Lang: {target_lang}) ---")

        final_ref = load_text_from_file_or_list(reference, "Reference")
        final_text = load_text_from_file_or_list(target_text, "Target Text")
        
        final_src = None
        if source:
            final_src = load_text_from_file_or_list(source, "Source")
            if len(final_src) != len(final_ref):
                raise ValueError("Source and Reference have different lengths")
        
        if len(final_text) != len(final_ref):
            raise ValueError("Target Text and Reference have different lengths")

        # 1. sacreBLEU
        if self.use_bleu:
            tokenizer_name = self._get_bleu_tokenizer_name(target_lang)
            try:
                results["sacreBLEU"] = sacrebleu.corpus_bleu(final_text, [final_ref], tokenize=tokenizer_name).score
            except Exception as e:
                results["sacreBLEU"] = -1.0

        # 2. chrF++
        if self.use_chrf:
            try:
                results["chrF++"] = sacrebleu.corpus_chrf(final_text, [final_ref], word_order=2).score
            except: results["chrF++"] = -1.0

        # 3. BLEURT
        if self.use_bleurt and self.bleurt_model:
            try:
                results["BLEURT"] = self._compute_bleurt_score(final_ref, final_text)
            except Exception as e:
                results["BLEURT"] = -1.0

        # 4. COMET
        if self.use_comet and self.comet and final_src:
            try:
                data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(final_src, final_text, final_ref)]
                gpus = 1 if self.device.startswith("cuda") else 0
                results["COMET"] = self.comet.predict(data, batch_size=8, gpus=gpus).system_score
            except Exception as e:
                results["COMET"] = -1.0
        
        return {k: round(v, 4) if v >= 0 else v for k, v in results.items()}
