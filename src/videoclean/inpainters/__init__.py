"""OpenCV inpainting backend and registry."""
from videoclean.inpainters.base import InpaintJob, InpaintOutcome, Inpainter
from videoclean.inpainters.opencv import OpenCVInpainter
from videoclean.inpainters.registry import InpainterRegistry, get_registry, register_inpainter

register_inpainter("opencv", OpenCVInpainter)

__all__ = ["Inpainter", "InpaintJob", "InpaintOutcome", "OpenCVInpainter", "InpainterRegistry", "get_registry", "register_inpainter"]
