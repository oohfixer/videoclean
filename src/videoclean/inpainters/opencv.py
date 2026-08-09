"""Fixed-overlay OpenCV inpainter.

Uses one spatial mask on every frame, which is appropriate for persistent
watermarks and logos that temporal video inpainters can reproduce.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

from videoclean.inpainters.base import InpaintJob, InpaintOutcome


class OpenCVInpainter:
    """Inpaint every frame with ``cv2.inpaint`` and a stable mask."""

    name = "opencv"

    def __init__(self, method: str = "telea", radius: float = 3.0, dilate: int = 0):
        if method not in {"telea", "ns"}:
            raise ValueError("method must be 'telea' or 'ns'")
        if radius <= 0:
            raise ValueError("radius must be greater than zero")
        if dilate < 0:
            raise ValueError("dilate must be non-negative")
        self.method = method
        self.radius = float(radius)
        self.dilate = int(dilate)
        self._method_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def load(self, weight_path=None, device: str = "auto") -> None:
        """OpenCV inpainting has no model weights to load."""

    def cleanup(self) -> None:
        """OpenCV inpainting has no persistent resources."""

    def _prepare_mask(self, job: InpaintJob) -> np.ndarray:
        mask = np.asarray(job.mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.ndim != 2 or mask.shape != (job.height, job.width):
            raise ValueError(
                f"OpenCV mask must have shape {(job.height, job.width)}, got {mask.shape}"
            )
        if np.issubdtype(mask.dtype, np.floating):
            max_value = float(np.max(mask)) if mask.size else 0.0
            if max_value <= 1.0:
                mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            else:
                mask = np.clip(mask, 0, 255).astype(np.uint8)
        else:
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if self.dilate:
            size = self.dilate * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.dilate(mask, kernel, iterations=1)
        if not np.any(mask):
            raise ValueError("Mask has no inpaintable regions")
        return mask

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        if job.reader is None:
            raise ValueError("OpenCVInpainter requires job.reader (a cv2.VideoCapture)")
        mask = self._prepare_mask(job)
        stem = os.path.splitext(os.path.basename(job.video_path))[0] if job.video_path else "output"
        output_path = os.path.join(job.output_dir, f"{stem}_{job.output_suffix}.mp4")
        os.makedirs(job.output_dir, exist_ok=True)
        out_h = job.height * 2 if job.dual else job.height
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-nostats",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{job.width}x{out_h}", "-pix_fmt", "bgr24",
            "-r", str(job.fps), "-i", "-",
        ]
        if job.video_path:
            ffmpeg_cmd += ["-i", job.video_path]
        ffmpeg_cmd += ["-map", "0:v"]
        if job.video_path:
            ffmpeg_cmd += ["-map", "1:a?"]
        ffmpeg_cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]
        if job.video_path:
            ffmpeg_cmd += ["-c:a", "aac"]
        ffmpeg_cmd += ["-movflags", "+faststart", output_path]

        stderr_file = tempfile.TemporaryFile()
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=stderr_file)
        stdin_closed = False
        failed = False
        started = time.monotonic()
        done = 0
        try:
            while True:
                ok, frame = job.reader.read()
                if not ok:
                    break
                cleaned = cv2.inpaint(frame, mask, self.radius, self._method_flag)
                if job.dual:
                    cleaned = np.vstack([frame, cleaned])
                pipe.stdin.write(cleaned.tobytes())
                done += 1
                if job.progress is not None:
                    job.progress(done, job.frame_count)
            pipe.stdin.close()
            stdin_closed = True
            pipe.wait()
            if isinstance(job.metrics, dict):
                job.metrics["inpainting_s"] = round(time.monotonic() - started, 3)
        except Exception:
            failed = True
            raise
        finally:
            if not stdin_closed and pipe.stdin is not None:
                pipe.stdin.close()
            if pipe.poll() is None:
                pipe.terminate()
                pipe.wait()
            if failed:
                stderr_file.close()
        if pipe.returncode != 0:
            stderr_file.seek(0)
            error = stderr_file.read().decode(errors="replace")
            stderr_file.close()
            raise RuntimeError(f"FFmpeg exited with code {pipe.returncode}:\n{error}")
        stderr_file.close()
        print(f"Saved to {output_path}")
        return InpaintOutcome(output_path=output_path, backend=self.name)
