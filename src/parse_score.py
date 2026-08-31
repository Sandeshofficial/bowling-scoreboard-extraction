def extract_numbers(ocr_results: List[Tuple[list, str, float]], min_conf: float = 0.3) -> List[int]:
    """Pulls out plausible score numbers from OCR results, filtering low-confidence noise."""
    numbers = []
    for bbox, text, conf in ocr_results:
        if conf < min_conf:
            continue
        cleaned = text.strip()
        # Only apply letter->digit confusion fixes to short, numeric-looking tokens
        # (avoids corrupting words like "JOHN" or "SCORE" which contain O/S/I letters)
        if re.fullmatch(r"[0-9OoIlSB]{1,4}", cleaned):
            cleaned = clean_token(cleaned)
        for match in NUMBER_RE.findall(cleaned):
            numbers.append(int(match))
    return numbers
