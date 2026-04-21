import time

from openst import GenericAgent, LatencyEvaluator, ReadAction, WriteAction


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
    evaluator = LatencyEvaluator(agent, segment_size=20)

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
