"""
extract_frames.py
------------------
Reads a video file and yields sampled frames at a fixed time interval
(instead of every single frame) to keep OCR calls cheap and avoid
processing near-duplicate frames.
"""

import cv2
from typing import Generator, Tuple


def extract_frames(
    video_path: str,
    sample_interval_sec: float = 0.5,
) -> Generator[Tuple[int, float, "cv2.Mat"], None, None]:
    """
    Yields (frame_index, timestamp_sec, frame) for frames sampled every
    `sample_interval_sec` seconds of video.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    sample_interval_sec : float
        How often (in seconds of video time) to sample a frame.

    Yields
    ------
    (frame_index, timestamp_sec, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps * sample_interval_sec)))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / fps
            yield frame_idx, timestamp_sec, frame

        frame_idx += 1

    cap.release()


def get_video_metadata(video_path: str) -> dict:
    """Returns basic metadata about the video (fps, frame count, duration, resolution)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0

    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/bowling_scoreboard.mp4"
    meta = get_video_metadata(path)
    print("Video metadata:", meta)

    count = 0
    for idx, ts, frame in extract_frames(path, sample_interval_sec=0.5):
        count += 1
    print(f"Sampled {count} frames at 0.5s interval.")
