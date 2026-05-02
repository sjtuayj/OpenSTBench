from bisect import bisect_right
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

    def get_source_chunk_end_times(self, ins):
        boundaries = list(getattr(ins, "source_chunk_end_times_ms", []) or [])
        if boundaries:
            return boundaries

        delays = list(getattr(ins, "delays", []) or [])
        if delays:
            return sorted(set(delays))

        source_length = float(getattr(ins, "source_length", 0.0) or 0.0)
        return [source_length] if source_length > 0 else []

    def source_chunk_id_from_delay(self, boundaries, delay_ms):
        if not boundaries:
            return 1
        idx = bisect_right(boundaries, float(delay_ms))
        idx = max(1, idx)
        return min(idx, len(boundaries))

    def split_duration_into_tokens(self, duration_ms, token_len_ms):
        if token_len_ms <= 0:
            return [0]
        num, rest = divmod(float(duration_ms), float(token_len_ms))
        tokens = int(num) * [float(token_len_ms)]
        if rest > 0:
            tokens.append(float(rest))
        return tokens or [float(duration_ms)]

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
    def __call__(self, instances) -> float:
        scores = []
        for ins in instances.values():
            delays = self.get_delays(ins)
            if not delays:
                continue

            source_boundaries = self.get_source_chunk_end_times(ins)
            if not source_boundaries:
                continue

            chunk_sizes = {"src": [], "tgt": []}
            token_to_chunk = {"src": [], "tgt": [0]}
            token_to_time = {"src": [], "tgt": [0.0]}

            (
                chunk_sizes["src"],
                token_to_chunk["src"],
                token_to_time["src"],
            ) = self.build_source_timeline(source_boundaries, token_len_ms=300.0)

            output_type = self.get_output_type(ins)
            target_chunk_sizes = [0] * (len(source_boundaries) + 1)

            target_delays = []
            target_chunk_ids = []
            target_token_lens = []
            compute_times = []

            if output_type == "text":
                alignment_chunk_ids = getattr(ins, "_alignment_output_chunk_ids", None)
                if alignment_chunk_ids:
                    output_emit_delays = list(ins.delays)
                    for delay, output_chunk_id in zip(delays, alignment_chunk_ids):
                        output_chunk_idx = max(0, min(output_chunk_id - 1, len(output_emit_delays) - 1))
                        source_chunk_id = self.source_chunk_id_from_delay(
                            source_boundaries,
                            output_emit_delays[output_chunk_idx],
                        )
                        target_delays.append(float(delay))
                        target_chunk_ids.append(source_chunk_id)
                        target_token_lens.append(0.0)
                        compute_times.append(0.0)
                else:
                    base_emit_delays = list(getattr(ins, "delays", []) or [])
                    text_compute_times = self.get_incremental_compute_times(ins, len(delays))

                    for idx, delay in enumerate(delays):
                        emit_delay = base_emit_delays[idx] if idx < len(base_emit_delays) else delay
                        source_chunk_id = self.source_chunk_id_from_delay(source_boundaries, emit_delay)
                        target_delays.append(float(delay))
                        target_chunk_ids.append(source_chunk_id)
                        target_token_lens.append(0.0)
                        extra_compute = 0.0 if self.computation_aware else float(text_compute_times[idx])
                        compute_times.append(extra_compute)
            else:
                speech_compute_times = self.get_incremental_compute_times(ins, len(delays))

                for idx, duration_ms in enumerate(getattr(ins, "durations", []) or []):
                    source_chunk_id = self.source_chunk_id_from_delay(source_boundaries, ins.delays[idx])
                    token_lens = self.split_duration_into_tokens(duration_ms, token_len_ms=300.0)
                    if self.computation_aware:
                        per_token_compute = 0.0
                    else:
                        per_token_compute = float(speech_compute_times[idx]) / len(token_lens) if token_lens else 0.0
                    for token_len in token_lens:
                        target_delays.append(float(delays[idx]))
                        target_chunk_ids.append(source_chunk_id)
                        target_token_lens.append(float(token_len))
                        compute_times.append(per_token_compute)

            if not target_delays:
                continue

            for chunk_id in target_chunk_ids:
                target_chunk_sizes[chunk_id] += 1
                token_to_chunk["tgt"].append(chunk_id)

            chunk_sizes["tgt"] = target_chunk_sizes

            for delay, comp, token_len in zip(target_delays, compute_times, target_token_lens):
                prev_tgt = token_to_time["tgt"][-1]
                start = max(float(delay), prev_tgt)
                token_to_time["tgt"].append(start + float(token_len) + float(comp))

            scores.append(self.compute_algo(chunk_sizes, token_to_chunk, token_to_time))

        return mean(scores) if scores else 0.0


@register("ATD_SpeechAlign")
@with_alignment
class ATDScorerAligned(ATDScorer):
    pass


@register("CustomATD")
class CustomATD(ATDScorer):
    def __call__(self, instances) -> float:
        scores = []
        for ins in instances.values():
            delays = self.get_delays(ins)
            if not delays:
                continue

            source_boundaries = self.get_source_chunk_end_times(ins)
            if not source_boundaries:
                continue

            chunk_sizes = {"src": [], "tgt": []}
            token_to_chunk = {"src": [], "tgt": [0]}
            token_to_time = {"src": [], "tgt": [0.0]}

            (
                chunk_sizes["src"],
                token_to_chunk["src"],
                token_to_time["src"],
            ) = self.build_source_timeline(source_boundaries, token_len_ms=300.0)

            output_type = self.get_output_type(ins)
            target_chunk_sizes = [0] * (len(source_boundaries) + 1)

            target_delays = []
            target_chunk_ids = []
            compute_times = []

            if output_type == "text":
                alignment_chunk_ids = getattr(ins, "_alignment_output_chunk_ids", None)
                if alignment_chunk_ids:
                    output_emit_delays = list(ins.delays)
                    for delay, output_chunk_id in zip(delays, alignment_chunk_ids):
                        output_chunk_idx = max(0, min(output_chunk_id - 1, len(output_emit_delays) - 1))
                        source_chunk_id = self.source_chunk_id_from_delay(
                            source_boundaries,
                            output_emit_delays[output_chunk_idx],
                        )
                        target_delays.append(float(delay))
                        target_chunk_ids.append(source_chunk_id)
                        compute_times.append(0.0)
                else:
                    base_emit_delays = list(getattr(ins, "delays", []) or [])
                    text_compute_times = self.get_incremental_compute_times(ins, len(delays))

                    for idx, delay in enumerate(delays):
                        emit_delay = base_emit_delays[idx] if idx < len(base_emit_delays) else delay
                        source_chunk_id = self.source_chunk_id_from_delay(source_boundaries, emit_delay)
                        target_delays.append(float(delay))
                        target_chunk_ids.append(source_chunk_id)
                        extra_compute = 0.0 if self.computation_aware else float(text_compute_times[idx])
                        compute_times.append(extra_compute)
            else:
                speech_compute_times = self.get_incremental_compute_times(ins, len(delays))

                for idx, duration_ms in enumerate(getattr(ins, "durations", []) or []):
                    source_chunk_id = self.source_chunk_id_from_delay(source_boundaries, ins.delays[idx])
                    token_lens = self.split_duration_into_tokens(duration_ms, token_len_ms=300.0)
                    if self.computation_aware:
                        per_token_compute = 0.0
                    else:
                        per_token_compute = float(speech_compute_times[idx]) / len(token_lens) if token_lens else 0.0
                    for _ in token_lens:
                        target_delays.append(float(delays[idx]))
                        target_chunk_ids.append(source_chunk_id)
                        compute_times.append(per_token_compute)

            if not target_delays:
                continue

            for chunk_id in target_chunk_ids:
                target_chunk_sizes[chunk_id] += 1
                token_to_chunk["tgt"].append(chunk_id)

            chunk_sizes["tgt"] = target_chunk_sizes

            for delay, comp in zip(target_delays, compute_times):
                prev_tgt = token_to_time["tgt"][-1]
                start = max(float(delay), prev_tgt)
                token_to_time["tgt"].append(start + float(comp))

            scores.append(self.compute_algo(chunk_sizes, token_to_chunk, token_to_time))

        return mean(scores) if scores else 0.0


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
