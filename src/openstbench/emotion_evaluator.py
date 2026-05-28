import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from numpy.linalg import norm

from ._model_loading import resolve_pretrained_source

def _load_data_list(
    input_data: Union[str, List[str]], 
    name: str,
    target_type: str = "audio"
) -> List[str]:
    if isinstance(input_data, list):
        return input_data
    if not isinstance(input_data, str):
        raise ValueError(f"{name} must be a file path (str) or a list (List[str]).")

    path = Path(input_data)
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {input_data}")
    
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
                raise ValueError(f"JSON list items do not contain common {target_type} fields: {candidate_keys}")

        if isinstance(data, dict):
            plural_candidates = [k + "s" for k in candidate_keys]
            plural_candidates += ["target_text", "hypothesis", "source_text"] if target_type == "text" else []
            for key in plural_candidates:
                if key in data:
                    return data[key]
            raise ValueError(f"JSON dictionary does not contain common {target_type} list fields")
        raise ValueError("Unsupported JSON format")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def _load_audio_from_folder(folder_path: str, extensions: tuple = (".wav", ".mp3", ".flac")) -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    audio_files = []
    for ext in extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    audio_files = sorted(audio_files, key=lambda x: x.stem)
    if not audio_files:
        raise ValueError(f"Folder contains no audio files: {folder_path}")
    return [str(f) for f in audio_files]


class EmotionEvaluator:
    """
    Emotion Evaluator based on Emotion2Vec+
    Supports:
    [1] Cross-lingual Fidelity Evaluation: Cosine Similarity fidelity based on 768-d high-dimensional embeddings.
    [2] Discrete Emotion Classification Evaluation: Recognition accuracy evaluation based on the model's Zero-Shot classification capability.
    """
    
    # The only dependent base model
    DEFAULT_E2V_MODEL = "iic/emotion2vec_plus_large"

    def __init__(self, 
                 e2v_model_path: Optional[str] = None,
                 custom_label_map: Optional[Dict[str, str]] = None,
                 device: Optional[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.e2v_model_path = e2v_model_path or self.DEFAULT_E2V_MODEL
        self.e2v_model = None
        
        # Convert user's custom label mapping to lowercase entirely to ensure robustness
        self.custom_label_map = {k.lower(): v.lower() for k, v in (custom_label_map or {}).items()}

    def _load_e2v_model(self):
        if self.e2v_model is not None: return
        print(f"⏳ Loading Emotion2Vec+ large model: {self.e2v_model_path}")
        try:
            from funasr import AutoModel
            model_source, source_kind = resolve_pretrained_source(
                self.e2v_model_path,
                fallback_source=self.DEFAULT_E2V_MODEL,
            )
            print(f"Loading Emotion2Vec+ ({source_kind}) from {model_source}...")
            self.e2v_model = AutoModel(model=model_source, device=self.device, disable_update=True)
            print("✅ Emotion2Vec+ large model loaded successfully!")
        except ImportError:
            print("❌ Emotion2Vec+ dependencies are missing. Please run: pip install funasr modelscope")
        except Exception as e:
            print(f"❌ Failed to load Emotion2Vec+ classification model: {e}")

    # =============== Feature Extraction Section ===============

    def _extract_e2v_embeddings(self, audio_paths: List[str]) -> List[np.ndarray]:
        """Extracts Emotion2Vec+ comprehensive high-dimensional embeddings to measure overall fidelity similarity."""
        self._load_e2v_model()
        if not self.e2v_model: return [np.zeros(768)] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 Audio E2V High-dim Features", unit="file"):
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
        """Extracts discrete emotion classification prediction results from Emotion2Vec+."""
        self._load_e2v_model()
        if not self.e2v_model: return ["unknown"] * len(audio_paths)

        results = []
        for path in tqdm(audio_paths, desc="🔍 Audio E2V Discrete Classification Labels", unit="file"):
            try:
                # Default classification mode, returns the specific label
                res = self.e2v_model.generate(path, output_dir=None, granularity="utterance", extract_embedding=False)
                if isinstance(res, list) and len(res) > 0 and 'labels' in res[0] and 'scores' in res[0]:
                    labels = res[0]['labels']
                    scores = res[0]['scores']
                    
                    # 1. Find the index of the label with the highest score
                    best_idx = np.argmax(scores)
                    best_label_str = labels[best_idx]
                    
                    # 2. Parse out the English part (since the output format might be 'angry')
                    if "/" in best_label_str:
                        label = best_label_str.split("/")[-1].lower()
                    else:
                        label = best_label_str.lower()
                        
                    # 3. Clean up potential special placeholders
                    label = label.replace("<|", "").replace("|>", "").strip()
                else:
                    label = "unknown"
                
                # Apply custom alignment
                if label in self.custom_label_map:
                    label = self.custom_label_map[label]
                results.append(label)
            except Exception as e:
                results.append("unknown")
        return results

    # =============== Main Evaluation Entry ===============

    def evaluate_all(self, 
                     source_audio: Optional[Union[List[str], str]] = None, 
                     target_audio: Optional[Union[List[str], str]] = None,
                     reference_labels: Optional[Union[List[str], str]] = None,
                     verbose: bool = True) -> Dict[str, float]:
        """
        Dynamic fusion engine entry point:
        - Passing only target_audio and reference_labels -> Calculates Accuracy
        - Passing source_audio and target_audio -> Calculates E2V Feature Fidelity
        """
        
        if target_audio is None:
            if source_audio is not None and reference_labels is not None:
                target_audio = source_audio
                source_audio = None
            else:
                raise ValueError("🚨 Must provide target_audio parameter to run evaluation.")

        if isinstance(target_audio, str) and os.path.isdir(target_audio):
            tgt_paths = _load_audio_from_folder(target_audio)
        else:
            tgt_paths = _load_data_list(target_audio, "Target Audio Paths", "audio")
        
        num_samples = len(tgt_paths)
        if num_samples == 0:
            raise ValueError("No target audio data found, evaluation stopped.")

        run_fidelity = source_audio is not None
        run_classification = reference_labels is not None
        results = {}

        # ==================== Branch 1: Fidelity Similarity Calculation ====================
        if run_fidelity:
            if verbose: print(f"\n📝 Starting 【Cross-modal Comprehensive Fidelity Calculation】 ({num_samples} Source-Target Audio Distance Comparisons)...")

            if isinstance(source_audio, str) and os.path.isdir(source_audio):
                src_paths = _load_audio_from_folder(source_audio)
            else:
                src_paths = _load_data_list(source_audio, "Source Audio Paths", "audio")

            if len(src_paths) != num_samples:
                raise ValueError(f"Count mismatch: Source ({len(src_paths)}) != Target ({num_samples})")

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


        # ==================== Branch 2: Discrete Emotion Recognition Calculation ====================
        if run_classification:
            if verbose: print(f"\n📝 Starting 【Emotion2Vec+ Discrete Classification Accuracy Calculation】 ({num_samples} Feature Recognition vs Ground Truth Comparisons)...")
            refs = _load_data_list(reference_labels, "Reference Labels", "label")
            
            if len(refs) != num_samples:
                raise ValueError(f"Reference label count ({len(refs)}) does not match target audio count ({num_samples})!")

            preds = self._extract_cls_emotion(tgt_paths)
            
            correct = 0
            for p, r in zip(preds, refs):
                if p.strip() == r.strip().lower():
                    correct += 1
            
            acc = correct / len(refs) if len(refs) > 0 else 0.0
            results["Audio_Emotion_Accuracy"] = round(acc, 4)


        # ==================== Output Feedback ====================
        if verbose:
            print("\n📊 [EmotionEvaluator] Emotion2Vec+ Comprehensive Evaluation Report:")
            print(f"   - Valid Evaluation Samples: {num_samples} items")
            
            if run_fidelity:
                print("\n   [Cross-modal Comprehensive Fidelity] (Value range [-1, 1], closer to 1.00 indicates higher emotional similarity)")
                print(f"   - Emotion2Vec+ Comprehensive Emotional Feature Cosine Similarity:         {results['Emotion2Vec_Cosine_Similarity']:.4f}")
            
            if run_classification:
                print("\n   [Deep Learning Discrete Emotion Recognition Accuracy] (Higher values indicate better preset label recognition)")
                print(f"   - Audio Emotion Accuracy (Discrete Emotion Recognition Accuracy): {results['Audio_Emotion_Accuracy']:.2%}")
                
        return results
