#!/usr/bin/env bash

# For S2S alignment, replace the MFA model names below with the target-language
# acoustic and dictionary models that match your dataset.
python -m openst.latency.cli \
  --source data/source.txt \
  --target data/ref.txt \
  --output ./output \
  --task s2t \
  --agent-script my_agent.py \
  --agent-class MyAgent \
  --segment-size 20 \
  --latency-unit char \
  --asr-model medium \
  --alignment-acoustic-model mandarin_mfa \
  --alignment-dictionary-model mandarin_mfa \
  --computation-aware \
  --quality
