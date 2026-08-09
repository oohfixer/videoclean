"""OpenCV-only video overlay removal."""
from videoclean.api import CancellationToken, ProgressEvent, WipeRequest, WipeResult
from videoclean.engine import WipeEngine, remove_text
from videoclean.errors import BackendUnavailableError, InvalidInputError, ProcessingCancelledError, ProcessingError, WipeError
from videoclean.inpainters import InpaintJob, InpaintOutcome, Inpainter, OpenCVInpainter, get_registry, register_inpainter
from videoclean.plan import Segment, Source, Track, WipePlan, build_wipe_plan, compute_source, is_temporal, load_wipe_plan, save_wipe_plan, validate_plan

__version__ = "0.1.0"
__all__ = ["CancellationToken", "ProgressEvent", "WipeRequest", "WipeResult", "WipeEngine", "remove_text", "WipeError", "InvalidInputError", "BackendUnavailableError", "ProcessingCancelledError", "ProcessingError", "InpaintJob", "InpaintOutcome", "Inpainter", "OpenCVInpainter", "get_registry", "register_inpainter", "Segment", "Source", "Track", "WipePlan", "build_wipe_plan", "compute_source", "is_temporal", "load_wipe_plan", "save_wipe_plan", "validate_plan"]
