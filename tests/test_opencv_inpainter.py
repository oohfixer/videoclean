import cv2
import numpy as np
import pytest

from videoclean.cli import _build_parser
from videoclean.inpainters import OpenCVInpainter, get_registry
from videoclean.inpainters.base import InpaintJob


def _video(path, frames=4, width=96, height=64):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), 30 + i * 5, dtype=np.uint8)
        cv2.rectangle(frame, (60, 20), (78, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_registry_and_cli_options():
    assert "opencv" in get_registry().names()
    assert isinstance(get_registry().create("opencv", method="ns"), OpenCVInpainter)
    args = _build_parser().parse_args([
        "clean", "input.mp4", "--inpaint-method", "ns",
        "--inpaint-radius", "5", "--inpaint-dilate", "2",
    ])
    assert (args.inpaint_method, args.inpaint_radius, args.inpaint_dilate) == (
        "ns", 5.0, 2,
    )

def test_mask_validation_and_normalization(tmp_path):
    inpainter = OpenCVInpainter()
    base = dict(video_path="x.mp4", output_dir=str(tmp_path), fps=4, frame_count=1, width=8, height=6)
    with pytest.raises(ValueError, match="no inpaintable"):
        inpainter._prepare_mask(InpaintJob(mask=np.zeros((6, 8), np.float32), **base))
    mask = inpainter._prepare_mask(InpaintJob(mask=np.eye(6, 8, dtype=np.float32)[:, :, None], **base))
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}


def test_inpaints_all_frames_and_reports_progress(tmp_path):
    source = tmp_path / "source.mp4"
    _video(source)
    reader = cv2.VideoCapture(str(source))
    mask = np.zeros((64, 96, 1), dtype=np.float32)
    mask[20:31, 60:79] = 1
    progress = []
    metrics = {}
    outcome = OpenCVInpainter(radius=3, dilate=1).inpaint(InpaintJob(
        video_path=str(source), mask=mask, output_dir=str(tmp_path), fps=4,
        frame_count=4, width=96, height=64, reader=reader, progress=lambda d, t: progress.append((d, t)), metrics=metrics,
        output_suffix="clean",
    ))
    reader.release()
    assert outcome.backend == "opencv"
    assert progress[-1] == (4, 4)
    assert "inpainting_s" in metrics
    output = cv2.VideoCapture(outcome.output_path)
    count = 0
    while output.read()[0]:
        count += 1
    output.release()
    assert count == 4
