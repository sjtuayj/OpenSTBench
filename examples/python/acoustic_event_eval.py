"""Evaluate acoustic-event content with the threshold-free BEATs_strong_1 model."""

from __future__ import annotations

import argparse
from pathlib import Path

from openstbench import BEATsStrongEventDetector, compute_event_content_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Event Content mAP for one positive and one negative clip."
    )
    parser.add_argument("positive_audio", type=Path, help="Audio containing laughter.")
    parser.add_argument("negative_audio", type=Path, help="Audio without laughter.")
    parser.add_argument(
        "--beats_model_path",
        type=Path,
        default=None,
        help="Optional BEATs_strong_1 checkpoint. Omit to download it to the user cache.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu, cuda, or cuda:0. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = ["positive", "negative"]

    # BEATs_strong_1 is used as published: there is no training, fine-tuning,
    # or manually selected detection threshold in this evaluation.
    detector = BEATsStrongEventDetector(
        model_path=args.beats_model_path,
        device=args.device,
        show_progress=True,
    )
    frame_scores = detector.predict_files(
        [args.positive_audio, args.negative_audio],
        sample_ids=sample_ids,
        progress_description="BEATs acoustic events",
    )

    result = compute_event_content_map(
        frame_scores,
        {
            "positive": ["Laughter"],
            "negative": [],
        },
        event_classes=["Laughter"],
    )
    print(result)


if __name__ == "__main__":
    main()
