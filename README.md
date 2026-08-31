# 🎳 Scoreboard Data Extraction from Video

A computer vision pipeline that detects a scoreboard within a video, reads the
text on it using OCR, and outputs structured, time-stamped scoreboard data
(player name, scores, running total) as JSON/CSV — plus an annotated video
overlay for visual verification.

Built for the **Round 1 Computer Vision Engineer assessment** — [FOG](https://www.futureofgaming.tech).

---

## How it works (pipeline overview)

```
Video
  │
  ▼
1. Frame Extraction        (src/extract_frames.py)
   Samples frames every N seconds instead of every frame, to reduce
   redundant OCR calls and processing time.
  │
  ▼
2. Scoreboard Localization (src/detect_scoreboard.py)
   Crops the scoreboard region out of each sampled frame using a
   calibrated Region of Interest (ROI). The sample video has a fixed,
   static scoreboard overlay, so a static ROI is fast and highly reliable.
   A hook is included to swap in a trained YOLOv8 detector if the
   scoreboard's position varies (e.g. a panning camera on a physical board).
  │
  ▼
3. Preprocessing + OCR     (src/ocr_extract.py)
   Upscales, denoises, and thresholds the cropped region, then runs
   EasyOCR to extract raw text + confidence scores.
  │
  ▼
4. Parsing                 (src/parse_score.py)
   Converts raw OCR tokens into a structured record: player name,
   detected score numbers, running total. Filters low-confidence noise
   and normalizes common OCR digit confusions (O↔0, I↔1, etc).
  │
  ▼
5. Temporal Deduplication  (src/parse_score.py)
   Only logs a new record when the scoreboard state actually changes,
   so the output isn't flooded with near-identical rows every 0.5s.
  │
  ▼
Output: scoreboard_data.json / .csv + annotated_video.mp4
```

---

## Project structure

```
scoreboard-cv/
├── app.py                     # Streamlit demo UI (upload video, see live extraction)
├── src/
│   ├── extract_frames.py      # Frame sampling + video metadata
│   ├── detect_scoreboard.py   # ROI calibration + cropping + drawing
│   ├── ocr_extract.py         # Preprocessing + EasyOCR wrapper
│   ├── parse_score.py         # OCR text -> structured scoreboard record
│   ├── pipeline.py            # CLI: runs the full end-to-end pipeline
│   └── save_sample_frame.py   # Utility: dump a frame to inspect for ROI coords
├── config/
│   └── roi.json               # Calibrated scoreboard bounding box (x, y, w, h)
├── data/                      # Put input video here for local CLI runs
├── output/                    # Pipeline outputs land here (json/csv/video)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <this-repo-url>
cd scoreboard-cv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: EasyOCR downloads its recognition model weights (~100MB) on first
> run — this requires an internet connection the first time only.

---

## Running it

### Option A — Streamlit demo (recommended for reviewers)

```bash
streamlit run app.py
```

Upload the video in the browser UI. It will show the video, the detected
scoreboard ROI live, and a downloadable table of extracted data.

### Option B — CLI pipeline (full video processing + annotated output video)

1. Place `bowling_scoreboard.mp4` in `data/`.
2. **Calibrate the scoreboard ROI once** (only needed the first time, or for
   a new video with a different scoreboard position):

   - **With a display available:**
     ```bash
     python src/pipeline.py --video data/bowling_scoreboard.mp4 --roi_calibrate
     ```
     Drag a box around the scoreboard in the popup window, press ENTER.

   - **Headless / no display:**
     ```bash
     python src/save_sample_frame.py --video data/bowling_scoreboard.mp4 --time 5.0
     ```
     Open `output/sample_frame.jpg` in any image viewer, note the
     scoreboard's pixel coordinates, then manually edit `config/roi.json`:
     ```json
     { "x": 40, "y": 40, "w": 400, "h": 160 }
     ```

3. Run the full pipeline:
   ```bash
   python src/pipeline.py --video data/bowling_scoreboard.mp4
   ```

4. Check `output/`:
   - `scoreboard_data.json` — structured extraction results
   - `scoreboard_data.csv` — same data, tabular
   - `annotated_video.mp4` — original video with the detected ROI box and
     live extracted score overlaid, frame by frame

---

## Sample output record

```json
{
  "frame_index": 125,
  "timestamp_sec": 5.0,
  "player_name": "JOHN",
  "detected_numbers": [7, 3, 10, 20],
  "running_total": 20,
  "raw_ocr_text": ["JOHN", "7", "3", "10", "20"]
}
```

---

## Design decisions & trade-offs

- **Static ROI vs. trained detector:** The provided sample video has a
  fixed-position scoreboard overlay, so a manually calibrated static ROI is
  faster, more reliable, and easier to verify than training a detector —
  the right tool for this specific input. `detect_scoreboard.py` includes a
  documented extension point (`detect_dynamic`) for plugging in a YOLOv8
  model if the target video has a moving/panning camera instead.
- **EasyOCR over Tesseract:** EasyOCR handled the stylized scoreboard font
  and small digit sizes noticeably better out-of-the-box in testing, with
  no additional training required.
- **Temporal deduplication:** Sampling at a fixed interval (e.g. every
  0.5s) naturally produces many duplicate readings while the score is
  unchanged between bowls. Deduping on state-change keeps the output
  meaningful (one row per actual score update) rather than one row per
  sampled frame.
- **Confidence filtering:** Low-confidence OCR tokens (<0.3) are filtered
  before parsing to reduce noise from motion blur or overlay transitions.

## Known limitations / next steps

- Field-level parsing (mapping specific numbers to specific frame slots)
  is currently heuristic (largest number = running total). For production
  use, sub-cropping each individual scoreboard cell (name box, per-frame
  boxes, total box) and running OCR per-cell would make parsing exact
  rather than heuristic — noted in `parse_score.py`.
- No temporal smoothing/voting across consecutive frames yet; could reduce
  occasional single-frame OCR misreads by requiring 2-3 consistent readings
  before logging a change.

---

## Deployment

This project is deployed as a live demo on **Hugging Face Spaces**
(Streamlit SDK): `<insert your deployed Space URL here>`

To redeploy:
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces), SDK = Streamlit.
2. Push this repo's contents to the Space's git remote (`app.py` at root is auto-detected).
3. Hugging Face installs `requirements.txt` and runs `app.py` automatically.

---

## Author

Submitted for the Computer Vision Engineer Round 1 assessment.
