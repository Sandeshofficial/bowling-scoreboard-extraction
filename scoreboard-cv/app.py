"""
app.py
-------
Streamlit demo UI for the Scoreboard Data Extraction pipeline.

Run locally:
    streamlit run app.py

Deploy on Hugging Face Spaces (Streamlit SDK) or Streamlit Community Cloud
by pointing at this file as the entry point.
"""

import json
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

from src.extract_frames import get_video_metadata
from src.detect_scoreboard import load_roi, crop_roi, draw_roi_box, ROI_CONFIG_PATH
from src.ocr_extract import ocr_scoreboard
from src.parse_score import build_scoreboard_record, is_duplicate

st.set_page_config(page_title="Scoreboard Data Extractor", layout="wide")

st.title("🎳 Scoreboard Data Extraction from Video")
st.caption(
    "Upload a bowling scoreboard video. The app localizes the scoreboard, "
    "runs OCR on it, and extracts structured score data over time."
)

with st.sidebar:
    st.header("Settings")
    sample_interval = st.slider("Sample interval (seconds)", 0.2, 2.0, 0.5, 0.1)
    max_frames = st.slider("Max frames to process (demo limit)", 5, 200, 40, 5)
    st.markdown("---")
    st.markdown(
        "**ROI Calibration**\n\n"
        "This demo uses a pre-calibrated static region for the scoreboard "
        "(`config/roi.json`), since the sample video has a fixed-position "
        "scoreboard overlay. See README for how to recalibrate for a new video."
    )

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    if not os.path.exists(ROI_CONFIG_PATH):
        st.error(
            "No ROI config found. Run `python src/pipeline.py --video <path> "
            "--roi_calibrate` locally first (needs a display) and commit the "
            "resulting config/roi.json, or set default coordinates manually."
        )
    else:
        roi = load_roi()
        meta = get_video_metadata(video_path)

        col1, col2 = st.columns(2)
        col1.metric("Duration (s)", f"{meta['duration_sec']:.1f}")
        col2.metric("Resolution", f"{meta['width']}x{meta['height']}")

        if st.button("Run Extraction", type="primary"):
            cap = cv2.VideoCapture(video_path)
            fps = meta["fps"] or 25.0
            frame_interval = max(1, int(round(fps * sample_interval)))

            progress = st.progress(0.0, text="Processing frames...")
            preview_col1, preview_col2 = st.columns(2)
            frame_placeholder = preview_col1.empty()
            roi_placeholder = preview_col2.empty()

            records = []
            prev_record = None
            frame_idx = 0
            processed = 0

            while processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp_sec = frame_idx / fps
                    cropped = crop_roi(frame, roi)
                    ocr_results = ocr_scoreboard(cropped)
                    record = build_scoreboard_record(ocr_results, frame_idx, timestamp_sec)

                    if not is_duplicate(prev_record, record):
                        records.append(record)
                        prev_record = record

                    annotated = draw_roi_box(frame, roi)
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame @ {timestamp_sec:.1f}s",
                    )
                    roi_placeholder.image(
                        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                        caption="Detected Scoreboard ROI",
                    )

                    processed += 1
                    progress.progress(
                        min(processed / max_frames, 1.0),
                        text=f"Processed {processed}/{max_frames} sampled frames",
                    )

                frame_idx += 1

            cap.release()
            progress.empty()

            st.success(f"Extraction complete — {len(records)} unique scoreboard states found.")

            if records:
                df = pd.DataFrame(records)
                st.subheader("Extracted Scoreboard Data")
                st.dataframe(df, use_container_width=True)

                col_a, col_b = st.columns(2)
                col_a.download_button(
                    "⬇ Download JSON",
                    data=json.dumps(records, indent=2),
                    file_name="scoreboard_data.json",
                    mime="application/json",
                )
                col_b.download_button(
                    "⬇ Download CSV",
                    data=df.to_csv(index=False),
                    file_name="scoreboard_data.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No scoreboard data detected — try adjusting the ROI or sample interval.")
else:
    st.info("Upload a video to get started, or place a sample at data/bowling_scoreboard.mp4 and run the CLI pipeline (see README).")
