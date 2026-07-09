# This file contains code adapted from SimulEval:
# https://github.com/facebookresearch/SimulEval
#
# SimulEval is licensed under the Creative Commons Attribution-ShareAlike
# 4.0 International License (CC BY-SA 4.0):
# https://creativecommons.org/licenses/by-sa/4.0/
#
# The adapted portions in this file are distributed under CC BY-SA 4.0.
# Modifications were made for OpenSTBench.

from .agent import GenericAgent, AgentPipeline
from .basics import ReadAction, WriteAction, TextSegment, SpeechSegment

__all__ = [
    "GenericAgent",
    "AgentPipeline",
    "ReadAction",
    "WriteAction",
    "TextSegment",
    "SpeechSegment",
    "LatencyEvaluator",
    "register",
]


def __getattr__(name):
    if name == "LatencyEvaluator":
        from .cli import LatencyEvaluator

        return LatencyEvaluator
    if name == "register":
        from .metrics import register

        return register
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | set(__all__))
