"""Bounded temporal background reconstruction helpers."""
from __future__ import annotations

import cv2
import numpy as np


def temporal_consensus(
    frames: list[np.ndarray], mask: np.ndarray, *, current: np.ndarray | None = None,
    min_agreement: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a robust background estimate and pixels supported by consensus.

    The method is deliberately conservative: if all supplied frames contain
    the same overlay, their disagreement/contrast prevents a fabricated fill.
    """
    if not frames:
        raise ValueError("at least one frame is required")
    stack = np.stack(frames).astype(np.float32)
    estimate = np.median(stack, axis=0).astype(np.uint8)
    spread = np.max(stack, axis=0) - np.min(stack, axis=0)
    stable = np.mean(spread, axis=2) <= 12.0
    if current is not None:
        current_f = current.astype(np.float32)
        distance = np.mean(np.abs(estimate.astype(np.float32) - current_f), axis=2)
        # Reject a temporal estimate that agrees with the overlay itself.
        stable &= distance > 8.0
    support = (mask > 0) & stable
    del min_agreement
    if len(frames) < 2:
        support[:] = False
    return estimate, (support.astype(np.uint8) * 255)


def align_translation(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float]:
    """Align a neighboring frame using phase correlation."""
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift, response = cv2.phaseCorrelate(ref, cur)
    if abs(shift[0]) > reference.shape[1] * 0.1 or abs(shift[1]) > reference.shape[0] * 0.1:
        return candidate, 0.0
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    aligned = cv2.warpAffine(candidate, matrix, (candidate.shape[1], candidate.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return aligned, float(response)
