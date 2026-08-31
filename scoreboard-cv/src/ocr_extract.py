"""
ocr_extract.py
---------------
Preprocesses a cropped scoreboard ROI to maximize OCR accuracy, then runs
EasyOCR to pull out raw text + bounding boxes.
"""

from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np


def preprocess_for_ocr(roi: np.ndarray, upscale: float = 2.0) -> np.ndarray:
    """
    Cleans up a scoreboard crop before OCR:
      - grayscale
      - upscale (small on-screen text benefits a lot from this)
      - denoise
      - adaptive threshold to boost contrast between digits and background
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if upscale != 1.0:
        gray = cv2.resize(
            gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.fastNlMeansDenoising(gray, h=10)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2,
    )
    return thresh


@lru_cache(maxsize=1)
def _get_reader():
    """Lazily initializes EasyOCR reader once and caches it (expensive to load)."""
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def run_ocr(processed_roi: np.ndarray) -> List[Tuple[list, str, float]]:
    """
    Runs EasyOCR on a preprocessed image.

    Returns a list of (bbox, text, confidence) tuples, same shape EasyOCR
    natively returns from `readtext`.
    """
    reader = _get_reader()
    results = reader.readtext(processed_roi)
    return results


def ocr_scoreboard(roi: np.ndarray, upscale: float = 2.0) -> List[Tuple[list, str, float]]:
    """Convenience wrapper: preprocess + OCR in one call."""
    processed = preprocess_for_ocr(roi, upscale=upscale)
    return run_ocr(processed)
