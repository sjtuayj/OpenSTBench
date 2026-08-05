#!/usr/bin/env bash

# Latency CLI example.
#
# Required arguments:
# - --source: text file containing one source audio path per line.
# - --agent-script: Python file containing the streaming agent class.
# - --agent-class: class name to load from --agent-script.
#
# Optional arguments:
# - --target: text file containing one reference per line.
# - --output: output directory for traces and artifacts.
# - --task: s2t or s2s.
# - --segment-size: source chunk duration in milliseconds.
# - --poll-interval-ms: streaming loop polling interval in milliseconds.
# - --target-language: target language used by auto tokenization and S2S ASR.
# - --latency-unit: auto, word, or char. Auto selects from target language;
#   explicit word/char preserve the historical behavior.
# - --disable-asr-fallback: disable Whisper fallback for S2S transcript
#   materialization.
# - --asr-model: Whisper model name/path for S2S fallback transcripts.
# - --alignment-acoustic-model: MFA acoustic model for speech alignment.
# - --alignment-dictionary-model: MFA dictionary model for speech alignment.
# - --computation-aware: also compute latency with recorded model time removed.
# - --slurm: submit the same command through the helper in latency.utils.
#
# MFA models are not selected automatically. For S2S alignment, replace the
# model names below with acoustic and dictionary models matching the target
# language.
python -m openstbench.latency.cli \
  --source data/source.txt \
  --target data/ref.txt \
  --output ./output \
  --task s2t \
  --agent-script my_agent.py \
  --agent-class MyAgent \
  --segment-size 20 \
  --poll-interval-ms 10 \
  --target-language ja \
  --latency-unit auto \
  --asr-model medium \
  --alignment-acoustic-model english_mfa \
  --alignment-dictionary-model english_mfa \
  --computation-aware
