from dataclasses import dataclass, field
from typing import Union, List

class Action:
    def is_read(self) -> bool: raise NotImplementedError

class ReadAction(Action):
    def is_read(self) -> bool: return True
    def __repr__(self): return "ReadAction"

@dataclass
class WriteAction(Action):
    content: Union[str, List[float]]
    finished: bool
    def is_read(self) -> bool: return False

@dataclass
class Segment:
    index: int = 0
    content: list = field(default_factory=list)
    finished: bool = False
    is_empty: bool = False
    data_type: str = None
    config: dict = field(default_factory=dict)

@dataclass
class EmptySegment(Segment):
    is_empty: bool = True

@dataclass
class TextSegment(Segment):
    content: str = ""
    data_type: str = "text"

@dataclass
class SpeechSegment(Segment):
    sample_rate: int = 16000
    data_type: str = "speech"

class AgentStates:
    def __init__(self): self.reset()
    def reset(self):
        self.source, self.target = [], []
        self.source_finished = self.target_finished = False
        self.upstream_states = []
    
    def update_source(self, segment: Segment):
        self.source_finished = segment.finished
        if not segment.is_empty:
            if isinstance(segment, TextSegment): self.source.append(segment.content)
            elif isinstance(segment, SpeechSegment): self.source += segment.content
            
    def update_target(self, segment: Segment):
        self.target_finished = segment.finished
        if not segment.is_empty and not self.target_finished:
            if isinstance(segment, TextSegment): self.target.append(segment.content)
            elif isinstance(segment, SpeechSegment): self.target += segment.content