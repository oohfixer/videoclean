"""Optional OCR text recognition for detected text crops.

Requires ``rapidocr-onnxruntime`` (install with ``pip install videoclean[ocr]``).
Importing this module fails gracefully when the dependency is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OCRRegion:
    """One RapidOCR detection with polygon, text, and confidence."""

    points: np.ndarray
    text: str
    confidence: float

_engine = None


def _get_engine():
    """Lazily initialize the RapidOCR engine."""
    global _engine
    if _engine is not None:
        return _engine
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ImportError(
            "rapidocr-onnxruntime is not installed. "
            "Install it with: pip install videoclean[ocr]"
        ) from exc
    _engine = RapidOCR()
    return _engine


def detect_regions(image: np.ndarray) -> list[OCRRegion]:
    """Detect and recognize text regions in a BGR image."""
    result, _ = _get_engine()(image)
    regions: list[OCRRegion] = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            points = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
            confidence = float(item[2])
        except (TypeError, ValueError):
            continue
        if points.shape[0] < 3 or not np.isfinite(points).all():
            continue
        if confidence <= 0:
            continue
        regions.append(OCRRegion(points, str(item[1] or ""), confidence))
    return regions


def recognize_text(image_crop: np.ndarray) -> Optional[str]:
    """Recognize text in a single image crop.

    Args:
        image_crop: BGR image as a numpy array (H, W, 3).

    Returns:
        Recognized text string, or ``None`` if no text is found.
    """
    regions = detect_regions(image_crop)
    texts = [region.text for region in regions if region.text]
    return " ".join(texts) if texts else None


def recognize_regions(image: np.ndarray) -> list[OCRRegion]:
    """Alias for :func:`detect_regions` for detector integrations."""
    return detect_regions(image)
