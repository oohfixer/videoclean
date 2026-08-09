"""Video metadata and mask helpers."""
import cv2


def read_frame_info(video_path: str):
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return reader, {
        "W_ori": int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
        "H_ori": int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
        "fps": reader.get(cv2.CAP_PROP_FPS) or 24.0,
        "len": int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5),
    }


def read_mask(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read mask image: {path}")
    _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return img[:, :, None]


def validate_mask_shape(mask, frame_info: dict) -> None:
    expected = (frame_info["H_ori"], frame_info["W_ori"])
    actual = mask.shape[:2]
    if actual != expected:
        raise ValueError(f"Mask shape {actual[1]}x{actual[0]} does not match video shape {expected[1]}x{expected[0]}")


class BaseTask:
    def __init__(self, gap: int = 200, dual: bool = False):
        self.gap = gap
        self.dual = dual
        self.inpainter = None
        self._bm = None
        self.feather_radius = 0

    def process_video(self, reader, frame_info, mask, output_dir: str, video_path: str = "", progress=None) -> str:
        raise NotImplementedError

    def cleanup(self):
        if self.inpainter is not None:
            self.inpainter.cleanup()
            self.inpainter = None
