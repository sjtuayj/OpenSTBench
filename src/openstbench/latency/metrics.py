# This file contains code adapted from SimulEval:
# https://github.com/facebookresearch/SimulEval
#
# SimulEval is licensed under the Creative Commons Attribution-ShareAlike
# 4.0 International License (CC BY-SA 4.0):
# https://creativecommons.org/licenses/by-sa/4.0/
#
# The adapted portions in this file are distributed under CC BY-SA 4.0.
# Modifications were made for OpenSTBench.

from math import isfinite
from statistics import mean

from .instance import SpeechToSpeechInstance
from .utils import Aligner, map_audio_offsets_to_output_times_and_chunks


SCORERS = {}


def register(name):
    def _reg(cls):
        SCORERS[name] = cls
        return cls

    return _reg


def with_alignment(scorer_cls):
    class AlignedScorer(scorer_cls):
        def __init__(
            self,
            computation_aware=False,
            output_dir="./output",
            alignment_acoustic_model="english_mfa",
            alignment_dictionary_model="english_mfa",
        ):
            super().__init__(computation_aware)
            self.aligner = Aligner(
                output_dir,
                acoustic_model=alignment_acoustic_model,
                dictionary_model=alignment_dictionary_model,
            )

        def __call__(self, instances):
            self.aligner.run_mfa()
            activated = []
            try:
                for idx, ins in instances.items():
                    if not isinstance(ins, SpeechToSpeechInstance):
                        continue
                    if not ins.durations or not ins.delays:
                        continue

                    units, offsets_ms = self.aligner.get_unit_alignment(idx)
                    if not units or not offsets_ms:
                        continue

                    chunk_times_ms = ins.elapsed if self.computation_aware else ins.delays
                    aligned_times, output_chunk_ids = map_audio_offsets_to_output_times_and_chunks(
                        offsets_ms,
                        chunk_times_ms,
                        ins.durations,
                    )
                    if not aligned_times:
                        continue

                    ins._alignment_delays = aligned_times
                    ins._alignment_units = units
                    ins._alignment_output_type = "text"
                    ins._alignment_output_chunk_ids = output_chunk_ids
                    activated.append(ins)

                if not activated:
                    return None

                return super().__call__(instances)
            finally:
                for ins in activated:
                    for attr in [
                        "_alignment_delays",
                        "_alignment_units",
                        "_alignment_output_type",
                        "_alignment_output_chunk_ids",
                    ]:
                        if hasattr(ins, attr):
                            delattr(ins, attr)

    return AlignedScorer


class LatencyScorer:
    def __init__(self, computation_aware=False):
        self.computation_aware = computation_aware

    def subtract(self, arr1, arr2):
        return [x - y for x, y in zip(arr1, arr2)]

    def get_incremental_compute_times(self, ins, count):
        if not self.computation_aware:
            return [0.0] * count

        elapsed = list(getattr(ins, "elapsed", []) or [])
        delays = list(getattr(ins, "delays", []) or [])
        if not elapsed or not delays:
            return [0.0] * count

        limit = min(len(elapsed), len(delays), count)
        if limit <= 0:
            return [0.0] * count

        compute_elapsed = [max(0.0, e - d) for e, d in zip(elapsed[:limit], delays[:limit])]
        incremental = []
        prev = 0.0
        for value in compute_elapsed:
            incremental.append(max(0.0, value - prev))
            prev = value

        if limit < count:
            incremental.extend([0.0] * (count - limit))
        return incremental

    def get_delays(self, ins):
        override = getattr(ins, "_alignment_delays", None)
        if override is not None:
            return override
        return ins.elapsed if self.computation_aware else ins.delays

    def get_output_type(self, ins):
        return getattr(
            ins,
            "_alignment_output_type",
            "speech" if isinstance(ins, SpeechToSpeechInstance) else "text",
        )

    def split_duration_into_tokens(self, duration_ms, token_len_ms):
        if token_len_ms <= 0:
            return [0]
        num, rest = divmod(float(duration_ms), float(token_len_ms))
        tokens = int(num) * [float(token_len_ms)]
        if rest > 0:
            tokens.append(float(rest))
        return tokens

    def build_source_timeline(self, boundaries, token_len_ms=300.0):
        chunk_sizes = [0]
        token_to_chunk = [0]
        token_to_time = [0.0]

        prev_boundary = 0.0
        for chunk_id, boundary in enumerate(boundaries, 1):
            duration = max(0.0, float(boundary) - prev_boundary)
            prev_boundary = float(boundary)
            tokens = self.split_duration_into_tokens(duration, token_len_ms)
            chunk_sizes.append(len(tokens))
            for token_duration in tokens:
                token_to_time.append(token_to_time[-1] + token_duration)
                token_to_chunk.append(chunk_id)

        return chunk_sizes, token_to_chunk, token_to_time

    def compute_algo(self, chunk_sizes, token_to_chunk, token_to_time):
        tgt_to_src = []
        for t in range(1, len(token_to_chunk["tgt"])):
            chunk_id = token_to_chunk["tgt"][t]
            acc_x = sum(chunk_sizes["src"][:chunk_id])
            acc_y = sum(chunk_sizes["tgt"][:chunk_id])
            s_est = t - max(0, acc_y - acc_x)
            curr_src = sum(chunk_sizes["src"][: chunk_id + 1])
            s_idx = s_est if s_est < curr_src else curr_src
            src_time_idx = min(s_idx, len(token_to_time["src"]) - 1)
            tgt_to_src.append((t, src_time_idx))

        delays = []
        for t, s in tgt_to_src:
            val = token_to_time["tgt"][t] - token_to_time["src"][s]
            delays.append(val)
        return float(mean(delays)) if delays else 0.0


@register("StartOffset")
class StartOffset(LatencyScorer):
    def __call__(self, instances):
        scores = []
        for ins in instances.values():
            d = self.get_delays(ins)
            scores.append(d[0] if d else 0)
        return mean(scores) if scores else 0


@register("StartOffset_SpeechAlign")
@with_alignment
class StartOffsetAligned(StartOffset):
    pass


@register("ATD")
class ATDScorer(LatencyScorer):
    """Average Token Delay with SimulEval-compatible speech chunking."""

    add_target_duration = True
    speech_token_len_ms = 300.0

    def __init__(self, computation_aware=False):
        super().__init__(computation_aware)
        self.diagnostics = self._empty_diagnostics()

    @staticmethod
    def _empty_diagnostics():
        return {
            "evaluated_instances": 0,
            "skipped_instances": 0,
            "skipped_instance_ids_by_reason": {},
        }

    def _skip(self, instance_id, reason):
        self.diagnostics["skipped_instances"] += 1
        by_reason = self.diagnostics["skipped_instance_ids_by_reason"]
        by_reason.setdefault(reason, []).append(instance_id)

    @staticmethod
    def _as_finite_floats(values):
        try:
            numbers = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        return numbers if all(isfinite(value) for value in numbers) else None

    @staticmethod
    def _is_non_decreasing(values):
        return all(curr >= prev for prev, curr in zip(values, values[1:]))

    @staticmethod
    def _ordered_unique(values):
        unique = []
        for value in values:
            if not unique or value != unique[-1]:
                unique.append(value)
        return unique

    def _compute_times(self, delays, elapsed):
        if not self.computation_aware:
            return [0.0] * len(delays)
        if elapsed == [0.0] * len(delays):
            return list(elapsed)

        compute_elapsed = self.subtract(elapsed, delays)
        return self.subtract(compute_elapsed, [0.0] + compute_elapsed[:-1])

    def _validate_and_prepare_base(self, ins):
        delays = self._as_finite_floats(getattr(ins, "delays", []) or [])
        if not delays:
            return None, "missing_or_invalid_delays"
        if not self._is_non_decreasing(delays):
            return None, "non_monotonic_delays"

        elapsed = []
        if self.computation_aware:
            elapsed = self._as_finite_floats(getattr(ins, "elapsed", []) or [])
            if elapsed is None or len(elapsed) != len(delays):
                return None, "delay_elapsed_length_mismatch"

        return {
            "delays": delays,
            "compute_times": self._compute_times(delays, elapsed),
            "source_boundaries": self._ordered_unique(delays),
        }, None

    def _prepare_text_target(self, ins, base):
        delays = base["delays"]
        compute_times = base["compute_times"]
        aligned_delays = getattr(ins, "_alignment_delays", None)
        alignment_chunk_ids = getattr(ins, "_alignment_output_chunk_ids", None)

        if aligned_delays is not None:
            target_delays = self._as_finite_floats(aligned_delays)
            if not target_delays:
                return None, "missing_or_invalid_alignment_delays"
            if not isinstance(alignment_chunk_ids, (list, tuple)) or len(alignment_chunk_ids) != len(target_delays):
                return None, "alignment_length_mismatch"

            output_action_ids = []
            action_id = 0
            prev_delay = None
            for delay in delays:
                if prev_delay is None or delay != prev_delay:
                    action_id += 1
                output_action_ids.append(action_id)
                prev_delay = delay

            target_chunk_ids = []
            for output_chunk_id in alignment_chunk_ids:
                try:
                    output_idx = int(output_chunk_id) - 1
                except (TypeError, ValueError):
                    return None, "invalid_alignment_chunk_id"
                if output_idx < 0 or output_idx >= len(output_action_ids):
                    return None, "invalid_alignment_chunk_id"
                target_chunk_ids.append(output_action_ids[output_idx])

            return {
                "delays": target_delays,
                "chunk_ids": target_chunk_ids,
                "token_lens": [0.0] * len(target_delays),
                # Aligned timestamps already use delay/elapsed as selected by
                # the alignment decorator, so adding compute again would
                # double count it.
                "compute_times": [0.0] * len(target_delays),
            }, None

        target_chunk_ids = []
        action_id = 0
        prev_delay = None
        for delay in delays:
            if prev_delay is None or delay != prev_delay:
                action_id += 1
            target_chunk_ids.append(action_id)
            prev_delay = delay

        return {
            "delays": delays,
            "chunk_ids": target_chunk_ids,
            "token_lens": [0.0] * len(delays),
            "compute_times": compute_times,
        }, None

    def _prepare_speech_target(self, ins, base):
        delays = base["delays"]
        durations = self._as_finite_floats(getattr(ins, "durations", []) or [])
        if durations is None or len(durations) != len(delays):
            return None, "delay_duration_length_mismatch"
        if any(duration < 0 for duration in durations):
            return None, "negative_target_duration"

        groups = []
        for delay, duration, compute_time in zip(delays, durations, base["compute_times"]):
            if groups and groups[-1][0] == delay:
                groups[-1][1] += duration
                groups[-1][2] += compute_time
            else:
                groups.append([delay, duration, compute_time])

        if any(duration <= 0 for _, duration, _ in groups):
            return None, "non_positive_target_action_duration"

        target_delays = []
        target_chunk_ids = []
        target_token_lens = []
        target_compute_times = []
        for chunk_id, (delay, duration, compute_time) in enumerate(groups, 1):
            token_lens = self.split_duration_into_tokens(duration, self.speech_token_len_ms)
            if not token_lens:
                return None, "empty_target_action"
            per_token_compute = compute_time / len(token_lens)
            target_delays.extend([delay] * len(token_lens))
            target_chunk_ids.extend([chunk_id] * len(token_lens))
            target_token_lens.extend(token_lens)
            target_compute_times.extend([per_token_compute] * len(token_lens))

        return {
            "delays": target_delays,
            "chunk_ids": target_chunk_ids,
            "token_lens": target_token_lens,
            "compute_times": target_compute_times,
        }, None

    def _score_instance(self, ins):
        base, reason = self._validate_and_prepare_base(ins)
        if reason is not None:
            return None, reason

        source_boundaries = base["source_boundaries"]
        if not source_boundaries:
            return None, "missing_source_boundaries"

        chunk_sizes = {"src": [], "tgt": []}
        token_to_chunk = {"src": [], "tgt": [0]}
        token_to_time = {"src": [], "tgt": [0.0]}
        (
            chunk_sizes["src"],
            token_to_chunk["src"],
            token_to_time["src"],
        ) = self.build_source_timeline(source_boundaries, token_len_ms=self.speech_token_len_ms)

        if self.get_output_type(ins) == "text":
            target, reason = self._prepare_text_target(ins, base)
        else:
            target, reason = self._prepare_speech_target(ins, base)
        if reason is not None:
            return None, reason

        target_chunk_sizes = [0] * (len(source_boundaries) + 1)
        for chunk_id in target["chunk_ids"]:
            if chunk_id <= 0 or chunk_id >= len(target_chunk_sizes):
                return None, "invalid_target_chunk_id"
            target_chunk_sizes[chunk_id] += 1
            token_to_chunk["tgt"].append(chunk_id)
        chunk_sizes["tgt"] = target_chunk_sizes

        for delay, compute_time, token_len in zip(
            target["delays"],
            target["compute_times"],
            target["token_lens"],
        ):
            target_start = max(delay, token_to_time["tgt"][-1])
            playback_time = token_len if self.add_target_duration else 0.0
            token_to_time["tgt"].append(target_start + playback_time + compute_time)

        if len(token_to_chunk["tgt"]) <= 1:
            return None, "missing_target_tokens"
        return self.compute_algo(chunk_sizes, token_to_chunk, token_to_time), None

    def __call__(self, instances) -> float:
        self.diagnostics = self._empty_diagnostics()
        scores = []
        for instance_id, ins in instances.items():
            score, reason = self._score_instance(ins)
            if reason is not None:
                self._skip(instance_id, reason)
                continue
            scores.append(score)
            self.diagnostics["evaluated_instances"] += 1

        return mean(scores) if scores else 0.0


@register("ATD_SpeechAlign")
@with_alignment
class ATDScorerAligned(ATDScorer):
    pass


@register("CustomATD")
class CustomATD(ATDScorer):
    add_target_duration = False


@register("CustomATD_SpeechAlign")
@with_alignment
class CustomATDAligned(CustomATD):
    pass


@register("RTF")
class RTFScorer(LatencyScorer):
    def __call__(self, instances) -> float:
        scores = []
        for ins in instances.values():
            if not hasattr(ins, "total_inference_time"):
                continue

            src_len_sec = ins.source_length / 1000.0
            if src_len_sec <= 0:
                continue

            scores.append(ins.total_inference_time / src_len_sec)

        return mean(scores) if scores else 0.0


@register("ModelGenerateRTF")
class ModelGenerateRTFScorer(LatencyScorer):
    def __call__(self, instances) -> float:
        scores = []
        for ins in instances.values():
            model_sec = getattr(ins, "total_model_inference_time", None)
            if model_sec is None:
                continue

            src_len_sec = ins.source_length / 1000.0
            if src_len_sec <= 0:
                continue

            scores.append(float(model_sec) / src_len_sec)

        return mean(scores) if scores else None
