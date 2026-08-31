def preprocess_for_ocr(roi: np.ndarray, upscale: float = 3.0) -> np.ndarray:
    """
    Cleans up a scoreboard crop before OCR: grayscale, upscale, denoise.
    Skips hard binarization — EasyOCR generally handles clean grayscale
    better than aggressively thresholded images, which can clip thin
    digit strokes (e.g. losing a '7').
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if upscale != 1.0:
        gray = cv2.resize(
            gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray
