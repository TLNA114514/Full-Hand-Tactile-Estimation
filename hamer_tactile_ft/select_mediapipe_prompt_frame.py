#!/usr/bin/env python3
"""Choose a SAM text-anchor frame with optional MediaPipe hand visibility.

MediaPipe is deliberately limited to selecting a frame index. Its landmarks,
handedness and boxes never enter SAM candidate selection or tactile inference.
"""

from __future__ import annotations

import argparse
import math
import sys


def sampled_indices(frame_count, sample_count):
    if frame_count <= 0:
        return []
    sample_count = min(frame_count, max(1, int(sample_count)))
    if sample_count == 1:
        return [0]
    return sorted(
        {
            int(round(index * (frame_count - 1) / (sample_count - 1)))
            for index in range(sample_count)
        }
    )


def load_hands_api():
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions"):
            return mp.solutions.hands
        from mediapipe.python.solutions import hands

        return hands
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MediaPipe Hands is unavailable. Install optional 'mediapipe' in the "
            "tactile or sam3bbox environment, or pass an explicit --prompt_frame."
        ) from exc


def choose_frame(video_path, sample_count, max_num_hands, min_detection_confidence):
    import cv2

    hands_api = load_hands_api()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sampled_indices(frame_count, sample_count)
    best = None
    try:
        with hands_api.Hands(
            static_image_mode=True,
            max_num_hands=int(max_num_hands),
            min_detection_confidence=float(min_detection_confidence),
        ) as detector:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = list(result.multi_hand_landmarks or ())
                handedness = list(result.multi_handedness or ())
                if not landmarks:
                    continue
                confidences = []
                areas = []
                for hand_index, hand_landmarks in enumerate(landmarks):
                    if hand_index < len(handedness):
                        classification = handedness[hand_index].classification
                        confidence = float(classification[0].score) if classification else 0.0
                    else:
                        confidence = 0.0
                    xs = [float(item.x) for item in hand_landmarks.landmark]
                    ys = [float(item.y) for item in hand_landmarks.landmark]
                    area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
                    confidences.append(confidence)
                    areas.append(area)
                visible_count = min(len(landmarks), int(max_num_hands))
                mean_confidence = sum(confidences) / max(1, len(confidences))
                mean_area = sum(areas) / max(1, len(areas))
                quality = (
                    mean_confidence
                    + 0.25 * min(mean_area / 0.12, 1.0)
                    + 0.05 * visible_count
                )
                candidate = (quality, mean_confidence, mean_area, -frame_index)
                if best is None or candidate > best[0]:
                    best = (candidate, frame_index, visible_count)
    finally:
        capture.release()
    if best is None:
        raise RuntimeError(
            f"MediaPipe found no hand in {len(indices)} sampled frames of {video_path}"
        )
    metrics, frame_index, visible_count = best
    print(
        f"MediaPipe auxiliary selected frame={frame_index}, hands={visible_count}, "
        f"confidence={metrics[1]:.3f}, area={metrics[2]:.4f}",
        file=sys.stderr,
    )
    return int(frame_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--max_num_hands", type=int, default=1)
    parser.add_argument("--min_detection_confidence", type=float, default=0.55)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.max_num_hands < 1:
        parser.error("--max_num_hands must be positive")
    if not math.isfinite(args.min_detection_confidence) or not (
        0.0 < args.min_detection_confidence < 1.0
    ):
        parser.error("--min_detection_confidence must lie in (0, 1)")
    try:
        selected = choose_frame(
            args.video,
            args.samples,
            args.max_num_hands,
            args.min_detection_confidence,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
