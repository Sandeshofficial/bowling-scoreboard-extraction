"""
save_sample_frame.py
----------------------
Utility to dump a single frame from the video as a JPG so you can inspect it
(e.g. open in an image viewer / Photos app) and determine scoreboard pixel
coordinates manually — useful on headless machines / servers where the
interactive cv2.selectROI window (in detect_scoreboard.calibrate_roi) won't
work.

Usage:
    python src/save_sample_frame.py --video data/bowling_scoreboard.mp4 --time 5.0 --out output/sample_frame.jpg
"""

import argparse
import cv2


def save_frame_at_time(video_path: str, time_sec: float, out_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_number = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError(f"Could not read frame at t={time_sec}s")

    cv2.imwrite(out_path, frame)
    print(f"Saved frame at t={time_sec}s -> {out_path}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    print(
        "\nOpen this image in any viewer, find the scoreboard's pixel box "
        "(x, y = top-left corner; w, h = width/height), then edit "
        "config/roi.json accordingly."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=5.0, help="Timestamp in seconds to grab.")
    parser.add_argument("--out", default="output/sample_frame.jpg")
    args = parser.parse_args()

    save_frame_at_time(args.video, args.time, args.out)
