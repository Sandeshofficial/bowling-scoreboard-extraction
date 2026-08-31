"""
parse_score.py
---------------
Converts raw OCR output (list of (bbox, text, confidence)) into a
structured scoreboard record: player name, per-frame scores, running total.

Bowling scoreboard OCR text is noisy (strike/spare marks, small digits), so
this module is deliberately defensive: it extracts whatever numeric/text
tokens it can and normalizes them, rather than assuming a rigid layout.
Field-level sub-ROIs (see README) make this far more reliable than parsing
one OCR blob for the whole scoreboard.
"""

import re
from typing import List, Tuple, Optional


NUMBER_RE = re.compile(r"\d{1,3}")


def clean_token(text: str) -> str:
    """Normalizes a raw OCR token: strips whitespace, fixes common OCR confusions."""
    t = text.strip()
    # common OCR digit confusions on scoreboard fonts
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace("I", "1").replace("l", "1")
    t = t.replace("S", "5")
    return t


def extract_numbers(ocr_results: List[Tuple[list, str, float]], min_conf: float = 0.3) -> List[int]:
    """Pulls out plausible score numbers from OCR results, filtering low-confidence noise."""
    numbers = []
    for bbox, text, conf in ocr_results:
        if conf < min_conf:
            continue
        cleaned = clean_token(text)
        for match in NUMBER_RE.findall(cleaned):
            numbers.append(int(match))
    return numbers


def extract_player_name(ocr_results: List[Tuple[list, str, float]], min_conf: float = 0.3) -> Optional[str]:
    """
    Heuristic: the player name is usually the longest alphabetic (non-numeric)
    token detected with reasonable confidence.
    """
    candidates = []
    for bbox, text, conf in ocr_results:
        if conf < min_conf:
            continue
        cleaned = text.strip()
        if cleaned.isalpha() and len(cleaned) > 1:
            candidates.append(cleaned)

    if not candidates:
        return None
    return max(candidates, key=len)


def build_scoreboard_record(
    ocr_results: List[Tuple[list, str, float]],
    frame_index: int,
    timestamp_sec: float,
) -> dict:
    """
    Builds a structured record for one sampled frame.

    Output shape:
    {
        "frame_index": int,
        "timestamp_sec": float,
        "player_name": str | None,
        "detected_numbers": [int, ...],   # raw numeric tokens found
        "running_total": int | None,      # heuristically the largest number seen
        "raw_ocr_text": [str, ...]        # all raw text tokens, for debugging
    }
    """
    numbers = extract_numbers(ocr_results)
    name = extract_player_name(ocr_results)
    raw_text = [text.strip() for _, text, conf in ocr_results if conf >= 0.3]

    running_total = max(numbers) if numbers else None

    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 2),
        "player_name": name,
        "detected_numbers": numbers,
        "running_total": running_total,
        "raw_ocr_text": raw_text,
    }


def is_duplicate(prev_record: Optional[dict], curr_record: dict) -> bool:
    """
    Temporal dedup: skip logging a new record if it's effectively identical
    to the previous one (same name + same number set), avoiding redundant
    rows every 0.5s when the scoreboard hasn't actually changed.
    """
    if prev_record is None:
        return False
    return (
        prev_record.get("player_name") == curr_record.get("player_name")
        and prev_record.get("detected_numbers") == curr_record.get("detected_numbers")
    )
