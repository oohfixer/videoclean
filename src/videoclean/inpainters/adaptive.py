"""Conservative adaptive OpenCV replacement."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from videoclean.inpainters.base import InpaintJob, InpaintOutcome
from videoclean.reconstruct import temporal_consensus
from videoclean.refine import refine_overlay_mask


class AdaptiveInpainter:
    """Use tight masks and temporal evidence, then fall back to OpenCV."""

    name = "adaptive"

    def __init__(self, method: str = "telea", radius: float = 3.0, dilate: int = 0):
        if method not in {"telea", "ns"}:
            raise ValueError("method must be 'telea' or 'ns'")
        self.method = method
        self.radius = float(radius)
        self.dilate = int(dilate)
        self._method_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def load(self, weight_path=None, device: str = "auto") -> None:
        del weight_path, device

    def cleanup(self) -> None:
        pass

    def _mask(self, job: InpaintJob, frame_index: int) -> np.ndarray:
        mask = job.frame_mask(frame_index) if job.frame_mask is not None else job.mask
        if mask is None:
            return np.zeros((job.height, job.width), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if self.dilate:
            size = self.dilate * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _clean_frame(self, frame: np.ndarray, mask: np.ndarray, history: list[np.ndarray]) -> np.ndarray:
        if not np.any(mask):
            return frame
        refined = refine_overlay_mask(frame, mask)
        final_mask = refined.mask.astype(np.uint8) * 255
        if not np.any(final_mask):
            return frame
        result = frame.copy()
        if history:
            estimate, supported = temporal_consensus(history, final_mask, current=frame)
            result[supported > 0] = estimate[supported > 0]
            remaining = (final_mask > 0) & (supported == 0)
        else:
            remaining = final_mask > 0
        if np.any(remaining):
            residual_mask = remaining.astype(np.uint8) * 255
            fallback = cv2.inpaint(frame, residual_mask, self.radius, self._method_flag)
            result[remaining] = fallback[remaining]
        return result

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        if job.reader is None:
            raise ValueError("AdaptiveInpainter requires job.reader")
        stem = os.path.splitext(os.path.basename(job.video_path))[0] if job.video_path else "output"
        output_path = os.path.join(job.output_dir, f"{stem}_{job.output_suffix}.mp4")
        os.makedirs(job.output_dir, exist_ok=True)
        out_h = job.height * 2 if job.dual else job.height
        command = ["ffmpeg", "-y", "-loglevel", "error", "-nostats", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{job.width}x{out_h}", "-pix_fmt", "bgr24", "-r", str(job.fps), "-i", "-"]
        if job.video_path:
            command += ["-i", job.video_path]
        command += ["-map", "0:v"]
        if job.video_path:
            command += ["-map", "1:a?"]
        command += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
        if job.video_path:
            command += ["-c:a", "aac"]
        command += ["-movflags", "+faststart", output_path]
        stderr = tempfile.TemporaryFile()
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)
        history: list[np.ndarray] = []
        done = 0
        started = time.monotonic()
        try:
            while True:
                ok, frame = job.reader.read()
                if not ok:
                    break
                mask = self._mask(job, done)
                cleaned = self._clean_frame(frame, mask, history)
                if job.dual:
                    cleaned = np.vstack([frame, cleaned])
                pipe.stdin.write(cleaned.tobytes())
                history.append(frame.copy())
                if len(history) > 3:
                    history.pop(0)
                done += 1
                if job.progress:
                    job.progress(done, job.frame_count)
            pipe.stdin.close()
            pipe.wait()
        finally:
            if pipe.poll() is None:
                pipe.terminate()
                pipe.wait()
            stderr.close()
        if pipe.returncode:
            raise RuntimeError(f"FFmpeg exited with code {pipe.returncode}")
        if isinstance(job.metrics, dict):
            job.metrics["inpainting_s"] = round(time.monotonic() - started, 3)
        return InpaintOutcome(output_path=output_path, backend=self.name)
