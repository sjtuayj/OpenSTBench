import time

from openstbench import GenericAgent, LatencyEvaluator, ReadAction, WriteAction


"""
Streaming latency example.

Required evaluation inputs:
- source_files: source speech paths.
- ref_files: reference text for alignment/quality context.
- agent: a GenericAgent subclass that emits ReadAction and WriteAction.

Configurable LatencyEvaluator parameters:
- segment_size: source chunk duration in milliseconds.
- poll_interval_ms: loop polling interval in milliseconds.
- latency_unit: "word" or "char" tokenization for text latency.
- asr_fallback_for_s2s_alignment: transcribe S2S output if no transcript is
  supplied by the agent.
- asr_model: Whisper model name/path for S2S transcript fallback.
- asr_device: optional device for ASR fallback.
- alignment_acoustic_model: MFA acoustic model name for speech alignment.
- alignment_dictionary_model: MFA dictionary model name for speech alignment.

Configurable run parameters:
- task: "s2t" or "s2s".
- output_dir: directory for latency traces and optional artifacts.
- visualize: write timeline plots.

Configurable compute_latency parameters:
- computation_aware: subtract recorded model inference time where supported.
- show_all_metrics: include raw scorer outputs in detailed_all_metrics.

Output metrics:
- First_Audio_Delay_(StartOffset_ms)
- Overall_Translation_Delay_(ATD_ms)
- End_Action_Delay_(CustomATD_ms)
- Real_Time_Factor_(RTF)
- Model_Generate_RTF
"""


class WaitUntilEndAgent(GenericAgent):
    def fake_model_generate(self):
        return "hello world"

    def policy(self, states=None):
        states = states or self.states

        if not states.source_finished:
            return ReadAction()

        if not states.target_finished:
            t0 = time.perf_counter()
            prediction = self.fake_model_generate()
            t1 = time.perf_counter()
            self.record_model_inference_time(t1 - t0)
            return WriteAction(prediction, finished=True)

        return ReadAction()


def main():
    agent = WaitUntilEndAgent()
    evaluator = LatencyEvaluator(
        agent=agent,
        segment_size=20,
        poll_interval_ms=10.0,
        latency_unit="char",
        asr_fallback_for_s2s_alignment=True,
        asr_model="medium",
        asr_device=None,
        alignment_acoustic_model="mandarin_mfa",
        alignment_dictionary_model="mandarin_mfa",
    )

    evaluator.run(
        source_files=["./data/a.wav", "./data/b.wav"],
        ref_files=["你好", "世界"],
        task="s2t",
        output_dir="./latency_output",
        visualize=False,
    )

    scores = evaluator.compute_latency(
        computation_aware=True,
        output_dir="./latency_output",
        show_all_metrics=False,
    )

    print(scores)

    # For S2S agents, return a SpeechSegment and put the native transcript and
    # pure model time in seg.config, for example:
    # config={
    #     "transcript": pred_text,
    #     "transcript_source": "native_transcript",
    #     "model_inference_time": generate_seconds,
    # }


if __name__ == "__main__":
    main()
