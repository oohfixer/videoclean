"""Cache and download the DBNet detector weight."""
from __future__ import annotations

import os
import urllib.request

from tqdm import tqdm

_WEIGHTS_DIR = os.environ.get("VIDEOCLEAN_WEIGHTS_DIR", os.path.expanduser("~/.videoclean/weights"))
_WEIGHT_BASE = "https://huggingface.co/stevenlearns/videoclean-detector/resolve"
_MANUAL_DOWNLOAD_MSG = (
    "Automatic download failed. Download the detector weight manually from:\n"
    "  https://huggingface.co/stevenlearns/videoclean-detector\n"
    "Place it in ~/.videoclean/weights/ or set VIDEOCLEAN_WEIGHTS_DIR."
)


def get_weights_dir() -> str:
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    return _WEIGHTS_DIR


def ensure_weight(filename: str, version: str = "main") -> str:
    """Return the detector weight path, downloading it when absent."""
    if filename != "ppocrv5_det_mob.onnx":
        raise ValueError(f"Unsupported detector weight: {filename}")
    local_path = os.path.join(get_weights_dir(), filename)
    if os.path.isfile(local_path):
        return local_path
    url = f"{_WEIGHT_BASE}/{version}/{filename}"
    print(f"Downloading {filename} from {url}")
    try:
        _download_with_progress(url, local_path)
    except Exception as exc:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(f"Failed to download {filename}: {exc}\n{_MANUAL_DOWNLOAD_MSG}") from exc
    return local_path


def _download_with_progress(url: str, dest: str) -> None:
    tmp_path = dest + ".tmp"

    class _Progress(tqdm):
        def update_to(self, blocks: int = 1, block_size: int = 1, total: int | None = None):
            if total is not None:
                self.total = total
            self.update(blocks * block_size - self.n)

    with _Progress(unit="B", unit_scale=True, desc=os.path.basename(dest)) as progress:
        urllib.request.urlretrieve(url, tmp_path, reporthook=progress.update_to)
    os.replace(tmp_path, dest)
