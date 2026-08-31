"""
detect_scoreboard.py
---------------------
Localizes the scoreboard region within a video frame.

Two modes are supported:

1. STATIC ROI (default) — for fixed-camera / fixed-overlay footage where the
   scoreboard always appears in the same pixel region. Coordinates are
   supplied via config/roi.json (generated once using `calibrate_roi`).

2. DYNAMIC DETECTION (optional/extendable) — a hook (`detect_dynamic`) is
   provided for plugging in a trained YOLOv8 model if the scoreboard moves
   or the camera pans/zooms. See README for how to extend this.
"""

import json
import os
import cv2
import numpy as np

ROI_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "roi.json")


def calibrate_roi(sample_frame_path: str, save_path: str = ROI_CONFIG_PATH) -> dict:
    """
    Interactive helper (run locally, NOT in headless/deployed environments).
    Opens a window on a sample frame and lets you drag-select the scoreboard
    region. Press ENTER/SPACE to confirm, ESC to cancel.

    Saves {"x": , "y": , "w": , "h": } to config/roi.json
    """
    frame = cv2.imread(sample_frame_path)
    if frame is None:
        raise IOError(f"Could not read sample frame: {sample_frame_path}")

    x, y, w, h = cv2.selectROI("Select Scoreboard Region", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    roi = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(roi, f, indent=2)

    print(f"Saved ROI {roi} to {save_path}")
    return roi


def load_roi(config_path: str = ROI_CONFIG_PATH) -> dict:
    """Loads a previously calibrated static ROI from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No ROI config found at {config_path}. "
            f"Run calibrate_roi() once, or pass an explicit roi dict."
        )
    with open(config_path, "r") as f:
        return json.load(f)


def crop_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    """Crops the scoreboard region out of a full frame given an roi dict."""
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    h_frame, w_frame = frame.shape[:2]

    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))

    return frame[y : y + h, x : x + w]


def detect_dynamic(frame: np.ndarray, model=None) -> dict:
    """
    Placeholder hook for a trained object-detection model (e.g. YOLOv8)
    that localizes the scoreboard when it moves between frames.

    To extend:
        from ultralytics import YOLO
        model = YOLO("weights/scoreboard_yolov8n.pt")
        results = model(frame)[0]
        # pick highest-confidence "scoreboard" box
        x1, y1, x2, y2 = results.boxes.xyxy[0].tolist()
        return {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}

    Returns an roi dict identical in shape to load_roi()'s output.
    """
    raise NotImplementedError(
        "Dynamic detection not trained for this assessment — static ROI is "
        "used since the sample video has a fixed-position scoreboard. "
        "See docstring for how to plug in a YOLOv8 model if needed."
    )


def draw_roi_box(frame: np.ndarray, roi: dict, label: str = "Scoreboard") -> np.ndarray:
    """Draws the detected ROI box + label on a copy of the frame (for demo/overlay video)."""
    out = frame.copy()
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        out, label, (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    return out
