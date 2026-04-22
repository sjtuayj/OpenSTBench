import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import textgrid

from .._model_loading import resolve_pretrained_source

try:
    import whisper
except ImportError:
    whisper = None


_WHISPER_MODELS = {}


def submit_slurm(args, python_script_path):
    cmd = f"python {python_script_path} " + " ".join([a for a in sys.argv[1:] if "--slurm" not in a])
    script = f"""#!/bin/bash
#SBATCH --job-name=latency_eval
#SBATCH --output={args.output}/slurm.log
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

mkdir -p {args.output}
{cmd}
"""
    script_path = Path(args.output) / "run.sh"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"Submitting Slurm job: {script_path}")
    os.system(f"sbatch {script_path}")
    sys.exit(0)


class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "visual"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, instance_data):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, skipping visualization.")
            return

        data = instance_data
        idx = data["index"]
        delays = [d / 1000 for d in data["delays"]]
        if not delays:
            return

        pred = data["prediction"]
        is_text = not str(pred).endswith(".wav")

        plt.figure(figsize=(10, 6))
        x_points = [0] + delays
        y_points = list(range(len(x_points)))
        plt.step(x_points, y_points, where="post", marker="o")
        plt.xlabel("Source Time (s)")
        plt.ylabel("Output Unit")
        plt.title(f"Instance {idx}")

        if is_text:
            words = pred.split()
            for i, txt in enumerate(words):
                if i + 1 < len(delays):
                    plt.text(delays[i + 1], i + 1, txt)
        else:
            plt.text(0, 0, f"Saved to {pred}")

        plt.grid(True)
        plt.savefig(self.output_dir / f"{idx}.png")
        plt.close()


def tokenize_latency_units(text, unit="word"):
    text = " ".join(str(text or "").strip().split())
    if not text:
        return []
    if unit == "char":
        compact = "".join(text.split())
        return [ch for ch in compact if ch.strip()]
    return [part for part in text.split(" ") if part]


def _prepare_alignment_transcript(text, unit="word"):
    units = tokenize_latency_units(text, unit=unit)
    return units, " ".join(units)


def _load_whisper_model(model_name="medium", device=None):
    if whisper is None:
        return None
    key = (model_name, device or "auto")
    if key not in _WHISPER_MODELS:
        resolved_model_name, _source_kind = resolve_pretrained_source(
            model_name,
            fallback_source="medium",
        )
        kwargs = {}
        if device:
            kwargs["device"] = device
        _WHISPER_MODELS[key] = whisper.load_model(resolved_model_name, **kwargs)
    return _WHISPER_MODELS[key]


def transcribe_audio_with_whisper(audio_paths, model_name="medium", device=None):
    model = _load_whisper_model(model_name=model_name, device=device)
    if model is None:
        return [""] * len(audio_paths)

    results = []
    use_fp16 = False
    if device:
        use_fp16 = "cuda" in device

    for path in audio_paths:
        try:
            res = model.transcribe(path, fp16=use_fp16)
            results.append(str(res.get("text", "")).strip())
        except Exception:
            results.append("")
    return results


def materialize_s2s_alignment_artifacts(
    instances,
    output_dir,
    unit="word",
    asr_fallback=False,
    asr_model="medium",
    asr_device=None,
):
    wav_dir = Path(output_dir) / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    pending_asr = []
    for ins in instances.values():
        wav_path = Path(ins.get_prediction_content())
        ins.prediction_audio_path = str(wav_path)
        txt_path = wav_path.with_suffix(".txt")

        transcript = (getattr(ins, "prediction_text", "") or "").strip()
        transcript_source = getattr(ins, "prediction_text_source", "none")

        if transcript:
            units, align_text = _prepare_alignment_transcript(transcript, unit=unit)
            ins.target_units = units
            ins.alignment_mode = transcript_source
            if units:
                txt_path.write_text(align_text, encoding="utf-8")
            elif txt_path.exists():
                txt_path.unlink()
        elif asr_fallback:
            pending_asr.append((ins, wav_path))
            ins.target_units = []
            ins.alignment_mode = "pending_asr_fallback"
        else:
            ins.target_units = []
            ins.alignment_mode = "none"
            if txt_path.exists():
                txt_path.unlink()

    if pending_asr:
        wav_paths = [str(path) for _, path in pending_asr]
        transcripts = transcribe_audio_with_whisper(
            wav_paths,
            model_name=asr_model,
            device=asr_device,
        )
        for (ins, wav_path), transcript in zip(pending_asr, transcripts):
            txt_path = wav_path.with_suffix(".txt")
            transcript = (transcript or "").strip()
            if not transcript:
                ins.target_units = []
                ins.alignment_mode = "none"
                if txt_path.exists():
                    txt_path.unlink()
                continue

            ins.prediction_text = transcript
            ins.prediction_text_source = "asr_fallback"
            ins.alignment_mode = "asr_fallback"

            units, align_text = _prepare_alignment_transcript(transcript, unit=unit)
            ins.target_units = units
            if units:
                txt_path.write_text(align_text, encoding="utf-8")
            elif txt_path.exists():
                txt_path.unlink()


def map_audio_offsets_to_output_times_and_chunks(offsets_ms, chunk_times_ms, chunk_durations_ms):
    if not offsets_ms or not chunk_times_ms or not chunk_durations_ms:
        return [], []

    count = min(len(chunk_times_ms), len(chunk_durations_ms))
    if count <= 0:
        return [], []

    chunk_times_ms = list(chunk_times_ms[:count])
    chunk_durations_ms = list(chunk_durations_ms[:count])

    output_times = []
    output_chunk_ids = []
    cumulative_end = 0.0
    chunk_starts = []
    for duration in chunk_durations_ms:
        chunk_starts.append(cumulative_end)
        cumulative_end += float(duration)

    for offset in offsets_ms:
        offset = float(offset)
        if offset < 0:
            offset = 0.0

        chosen_idx = count - 1
        for idx, start in enumerate(chunk_starts):
            end = start + float(chunk_durations_ms[idx])
            if offset <= end or idx == count - 1:
                chosen_idx = idx
                break

        local_offset = offset - chunk_starts[chosen_idx]
        local_offset = max(0.0, min(local_offset, float(chunk_durations_ms[chosen_idx])))
        output_times.append(float(chunk_times_ms[chosen_idx]) + local_offset)
        output_chunk_ids.append(chosen_idx + 1)

    return output_times, output_chunk_ids


def map_audio_offsets_to_output_times(offsets_ms, chunk_times_ms, chunk_durations_ms):
    output_times, _ = map_audio_offsets_to_output_times_and_chunks(
        offsets_ms,
        chunk_times_ms,
        chunk_durations_ms,
    )
    return output_times


class Aligner:
    def __init__(self, output_dir, acoustic_model="english_mfa", dictionary_model="english_mfa"):
        self.output_dir = Path(output_dir)
        self.align_dir = self.output_dir / "align"
        self.wav_dir = self.output_dir / "wavs"
        self.temp_dir = self.output_dir / "mfa_temp"
        self.acoustic_model = acoustic_model
        self.dictionary_model = dictionary_model

    def run_mfa(self):
        if self.align_dir.exists() and any(self.align_dir.iterdir()):
            return
        try:
            subprocess.check_output("mfa version", shell=True)
        except Exception:
            print("Error: 'mfa' command not found. Cannot run alignment.")
            return

        if not self.wav_dir.exists() or not any(self.wav_dir.glob("*.txt")):
            return

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.align_dir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"mfa align {self.wav_dir} {self.dictionary_model} {self.acoustic_model} {self.align_dir} "
            f"--clean --overwrite --temporary_directory {self.temp_dir} --verbose"
        )
        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            print(f"MFA Alignment failed: {e}")

    def get_unit_alignment(self, index):
        tg_path = self.align_dir / f"{index}_pred.TextGrid"
        if not tg_path.exists():
            return None, None

        tg = textgrid.TextGrid.fromFile(str(tg_path))
        tier = tg[0] if len(tg) > 0 else None
        if tier is None:
            return None, None

        units = []
        offsets_ms = []
        for interval in tier:
            mark = str(interval.mark or "").strip()
            if not mark or mark == "<eps>":
                continue
            units.append(mark)
            offsets_ms.append(float(interval.maxTime) * 1000.0)
        return units, offsets_ms
