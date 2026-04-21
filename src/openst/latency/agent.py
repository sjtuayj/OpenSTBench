from inspect import signature
from typing import List

from .basics import (
    Action,
    AgentStates,
    EmptySegment,
    Segment,
    SpeechSegment,
    TextSegment,
)


class GenericAgent:
    """Base agent interface for latency evaluation."""

    def __init__(self):
        self.states = AgentStates()
        self._pending_model_inference_time = None
        self.reset()

    def reset(self):
        self.states.reset()
        self._pending_model_inference_time = None

    def record_model_inference_time(self, seconds: float) -> None:
        if seconds is None:
            return
        seconds = float(seconds)
        if seconds < 0:
            return
        if self._pending_model_inference_time is None:
            self._pending_model_inference_time = 0.0
        self._pending_model_inference_time += seconds

    def consume_model_inference_time(self):
        value = self._pending_model_inference_time
        self._pending_model_inference_time = None
        return value

    def policy(self, states=None) -> Action:
        raise NotImplementedError("Please implement policy().")

    def push(self, segment: Segment, states=None) -> None:
        (states or self.states).update_source(segment)

    def pop(self, states=None) -> Segment:
        state = states or self.states
        if state.target_finished:
            return EmptySegment(finished=True)

        if len(signature(self.policy).parameters) > 0:
            action = self.policy(state)
        else:
            action = self.policy()

        if action.is_read():
            return EmptySegment()

        if isinstance(action.content, Segment):
            segment = action.content
            if action.finished is not None:
                segment.finished = action.finished
        elif isinstance(action.content, str):
            segment = TextSegment(content=action.content, finished=action.finished)
        elif isinstance(action.content, list):
            segment = SpeechSegment(content=action.content, finished=action.finished)
        else:
            raise ValueError(f"Unknown content type: {type(action.content)}")

        state.update_target(segment)
        return segment

    def pushpop(self, segment: Segment) -> Segment:
        self.push(segment)
        return self.pop()


class AgentPipeline(GenericAgent):
    """Sequentially compose multiple agents."""

    def __init__(self, agents: List[GenericAgent]):
        self.pipeline = agents
        self.states = [agent.states for agent in agents]
        self._pending_model_inference_time = None

    def reset(self):
        self._pending_model_inference_time = None
        for agent in self.pipeline:
            agent.reset()

    def pushpop(self, segment: Segment) -> Segment:
        current_input = segment
        for index, agent in enumerate(self.pipeline):
            if index == 0:
                agent.push(current_input)
                current_output = agent.pop()
            else:
                if not current_output.is_empty:
                    agent.push(current_output)
                current_output = agent.pop()

            if hasattr(agent, "consume_model_inference_time"):
                self.record_model_inference_time(agent.consume_model_inference_time())

            if current_output.is_empty and index < len(self.pipeline) - 1:
                return EmptySegment()
        return current_output
