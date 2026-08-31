"""
pipeline.py
------------
End-to-end orchestrator:
    video -> sampled frames -> ROI crop -> OCR -> parsed records -> dedup -> outputs

Produces:
    output/scoreboard_data.json
    output/scoreboard_data.csv
    output/annotated_video.mp4   (original video with ROI box + live OCR overlay)

Usage:
    python src/pipeline.py --video data/bowling_scoreboard.mp4 --roi_calibrate
    python src/pipeline.py --video data/bowling_scoreboard.mp4
"""

import argparse
import json
import os
import sys

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from extract_frames import extract_frames, get_video_metadata
from detect_scoreboard import load_roi, crop_roi, draw_roi_box, calibrate_roi, ROI_CONFIG_PATH
from ocr_extract import ocr_scoreboard
from parse_score import build_scoreboard_record, is_duplicate


def run_pipeline(
    video_path: str,
    output_dir: str = "output",
    sample_interval_sec: float = 0.5,
    roi: dict = None,
    write_annotated_video: bool = True,
) -> list:
    os.makedirs(output_dir, exist_ok=True)

    if roi is None:
        roi = load_roi()

    meta = get_video_metadata(video_path)
    print(f"Video metadata: {meta}")

    records = []
    prev_record = None

    writer = None
    if write_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(output_dir, "annotated_video.mp4")
        writer = cv2.VideoWriter(
            out_path, fourcc, meta["fps"], (meta["width"], meta["height"])
        )

    frames_iter = extract_frames(video_path, sample_interval_sec=sample_interval_sec)
    approx_total = int(meta["duration_sec"] / sample_interval_sec) + 1

    for frame_idx, timestamp_sec, frame in tqdm(frames_iter, total=approx_total, desc="Processing frames"):
        cropped = crop_roi(frame, roi)
        ocr_results = ocr_scoreboard(cropped)
        record = build_scoreboard_record(ocr_results, frame_idx, timestamp_sec)

        if not is_duplicate(prev_record, record):
            records.append(record)
            prev_record = record

        if writer is not None:
            annotated = draw_roi_box(frame, roi)
            label = f"{record['player_name'] or '?'}: {record['running_total'] if record['running_total'] is not None else '?'}"
            cv2.putText(
                annotated, label, (roi["x"], roi["y"] + roi["h"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            writer.write(annotated)

    if writer is not None:
        writer.release()

    json_path = os.path.join(output_dir, "scoreboard_data.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    csv_path = os.path.join(output_dir, "scoreboard_data.csv")
    if records:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)

    print(f"\nDone. {len(records)} unique scoreboard states extracted.")
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")
    if write_annotated_video:
        print(f"Video -> {os.path.join(output_dir, 'annotated_video.mp4')}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Extract scoreboard data from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--output_dir", default="output", help="Directory to write outputs.")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds.")
    parser.add_argument(
        "--roi_calibrate",
        action="store_true",
        help="Interactively select the scoreboard ROI on the first frame before processing (requires a display).",
    )
    parser.add_argument("--no_video_output", action="store_true", help="Skip writing the annotated video.")
    args = parser.parse_args()

    if args.roi_calibrate:
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IOError("Could not read first frame for ROI calibration.")
        tmp_path = os.path.join(args.output_dir, "_calib_frame.jpg")
        os.makedirs(args.output_dir, exist_ok=True)
        cv2.imwrite(tmp_path, frame)
        calibrate_roi(tmp_path)

    run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_interval_sec=args.interval,
        write_annotated_video=not args.no_video_output,
    )


if __name__ == "__main__":
    main()
