"""Conservative glyph-shaped mask refinement."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RefinedMask:
    mask: np.ndarray
    confidence: float


def refine_overlay_mask(frame: np.ndarray, coarse_mask: np.ndarray, *, min_confidence: float = 0.20) -> RefinedMask:
    """Keep high-contrast glyph-like pixels inside a detector mask.

    The coarse mask remains the safety fallback when local evidence is
    ambiguous; no pixels outside it are ever introduced.
    """
    mask = np.asarray(coarse_mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    coarse = mask > 0
    if frame.ndim != 3 or frame.shape[:2] != coarse.shape or not coarse.any():
        return RefinedMask(coarse.astype(np.uint8), 0.0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 1.2)
    contrast = np.abs(gray - local)
    roi_values = contrast[coarse]
    if roi_values.size == 0:
        return RefinedMask(coarse.astype(np.uint8), 0.0)
    threshold = max(8.0, float(np.percentile(roi_values, 55.0)))
    glyph = coarse & (contrast >= threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glyph = cv2.morphologyEx(glyph.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, kernel) > 0
    retained = float(glyph.sum()) / float(coarse.sum())
    contrast_score = min(1.0, float(np.mean(roi_values)) / 32.0)
    confidence = min(1.0, 0.65 * contrast_score + 0.35 * min(1.0, retained * 2.0))
    if confidence < min_confidence or glyph.sum() < 2:
        return RefinedMask(coarse.astype(np.uint8), confidence)
    return RefinedMask(glyph.astype(np.uint8), confidence)
