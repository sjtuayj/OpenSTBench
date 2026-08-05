#!/usr/bin/env bash

# Install everything, including Whisper and Japanese/Korean BLEU tokenizers.
pip install "OpenSTBench[all]"

# Minimal multilingual alternatives:
# pip install "OpenSTBench[whisper]"
# pip install "OpenSTBench[tokenizer-ja]"
# pip install "OpenSTBench[tokenizer-ko]"
# pip install "OpenSTBench[whisper,tokenizer-ja,tokenizer-ko]"
