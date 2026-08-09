"""Focused OpenCV video-cleaning engine."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from videoclean.api import CancellationToken, ProgressCallback, ProgressEvent, WipeRequest, WipeResult
from videoclean.errors import InvalidInputError, ProcessingCancelledError, ProcessingError, WipeError
from videoclean.inpainters import get_registry, InpaintJob
from videoclean.plan import (
    WipePlan, build_refined_wipe_plan, compute_source, is_temporal,
    load_wipe_plan, save_wipe_plan, validate_plan,
)
from videoclean.tasks.base import BaseTask, read_frame_info, read_mask, validate_mask_shape
from videoclean.tasks.detext import DetextTask

if TYPE_CHECKING:
    from videoclean.detect import TextDetector

_TASK_CLASSES = {"detext": DetextTask, "clean": DetextTask}
_DEFAULT_FEATHER_RADIUS = 4


class _ProgressCallbackError(Exception):
    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class WipeEngine:
    """Reusable engine for deterministic, fixed-mask OpenCV inpainting."""

    def __init__(
        self, task: str = "detext", *, verbose: bool = False, gap: int = 10,
        dual: bool = False, detector: Optional[TextDetector] = None,
        model: str = "opencv", model_options: Optional[dict] = None,
        inpaint_model: Optional[str] = None,
        detect_mode: str = "auto", ocr: str = "auto", **legacy,
    ):
        if task not in _TASK_CLASSES:
            raise ValueError(f"Unknown task: {task}. Choose from: {list(_TASK_CLASSES)}")
        model = inpaint_model or model
        if model not in {"opencv", "adaptive"}:
            raise ValueError("videoclean supports 'opencv' and 'adaptive' inpainters")
        self.task = task
        self._model = model
        self._verbose = verbose
        self._detector = detector
        self._detect_mode = detect_mode
        self._ocr = ocr
        self._model_options = dict(model_options or {})
        self._task_impl: BaseTask = _TASK_CLASSES[task](gap=gap, dual=dual)
        self._task_impl.inpainter = get_registry().create(model, **self._model_options)
        self._task_impl.backend = self._task_impl.inpainter
        self._task_impl.feather_radius = _DEFAULT_FEATHER_RADIUS
        if task == "clean":
            self._task_impl.output_suffix = "clean"
        self._active_cancellation = None
        self._active_progress = None
        self._run_lock = threading.Lock()
        self._last_warnings: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def run(self, request: WipeRequest, on_progress: Optional[ProgressCallback] = None,
            cancellation: Optional[CancellationToken] = None) -> WipeResult:
        if not isinstance(request, WipeRequest):
            raise InvalidInputError("request must be a WipeRequest instance")
        if on_progress is not None and not callable(on_progress):
            raise InvalidInputError("on_progress must be callable")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise InvalidInputError("cancellation must be a CancellationToken")
        try:
            video = os.fspath(request.video)
            output = os.fspath(request.output_dir)
            mask = os.fspath(request.mask) if request.mask is not None else None
        except TypeError as exc:
            raise InvalidInputError("video, mask, and output_dir must be filesystem paths", cause=exc) from exc
        if not video or not output:
            raise InvalidInputError("video and output_dir must not be empty")
        if not self._run_lock.acquire(blocking=False):
            raise ProcessingError("This WipeEngine is already processing a request", code="ENGINE_BUSY", retryable=True)
        self._active_cancellation = cancellation or CancellationToken()
        self._active_progress = on_progress
        before = self._artifact_snapshot(output)
        try:
            self._emit_progress(ProgressEvent("prepare", 0, 1))
            result_path = self.process(
                video, mask=mask, output=output, detector=request.detector,
                targets=list(request.targets), intent=request.intent, regions=list(request.regions),
                preview=request.preview, confirm=request.confirm, detect_mode=request.detect_mode,
                ocr=request.ocr, detector_mode=request.detector_mode, plan=request.plan,
                progress=lambda done, total: self._emit_progress(ProgressEvent("inpaint", done, total)),
            )
            result = self._build_result(request, result_path, before)
            self._emit_progress(ProgressEvent("complete", 1, 1), check_after=False)
            return result
        except _ProgressCallbackError as exc:
            raise exc.cause
        except (ProcessingCancelledError, WipeError):
            raise
        except ValueError as exc:
            raise InvalidInputError(str(exc), cause=exc) from exc
        except Exception as exc:
            raise ProcessingError(str(exc), code="PROCESSING_FAILED", cause=exc) from exc
        finally:
            self._active_cancellation = None
            self._active_progress = None
            self._run_lock.release()

    @staticmethod
    def _artifact_snapshot(output_dir: str) -> dict:
        result = {}
        for name in ("auto_mask.png", "clean_candidates.json", "clean_preview.jpg", "benchmark.json", "wipe_plan.json", "wipe_plan_masks.npz"):
            path = os.path.join(output_dir, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            result[name] = (stat.st_mtime_ns, stat.st_size)
        return result

    def _build_result(self, request, output_path, before):
        output_dir = os.fspath(request.output_dir)
        after = self._artifact_snapshot(output_dir)
        benchmark = {}
        try:
            with open(os.path.join(output_dir, "benchmark.json"), encoding="utf-8") as handle:
                benchmark = json.load(handle)
        except (OSError, ValueError):
            pass
        artifacts = [os.path.join(output_dir, name) for name, sig in after.items() if sig != before.get(name)]
        if os.path.isfile(output_path) and output_path not in artifacts:
            artifacts.append(output_path)
        return WipeResult(
            output_path=output_path, backend=benchmark.get("backend", "opencv"),
            mask_source="manual" if request.mask is not None else "auto",
            artifacts=tuple(artifacts), timings=dict(benchmark.get("timing", {})),
            warnings=tuple(self._last_warnings), preview=request.preview,
        )

    def _check_cancelled(self):
        if self._active_cancellation is not None:
            self._active_cancellation.raise_if_cancelled()

    def _emit_progress(self, event, *, check_after=True):
        self._check_cancelled()
        if self._active_progress is not None:
            try:
                self._active_progress(event)
            except Exception as exc:
                raise _ProgressCallbackError(exc) from exc
        if check_after:
            self._check_cancelled()

    def _ensure_model(self):
        if self._task_impl.inpainter is None:
            self._task_impl.inpainter = get_registry().create(self._model, **self._model_options)
        self._task_impl.backend = self._task_impl.inpainter

    def process(self, video: str, mask: str | None = None, output: str = "result/",
                detector: Optional[TextDetector] = None, targets=None, intent=None,
                agent=None, regions=None, preview=False, confirm=False,
                detect_mode=None, ocr=None, detector_mode=None, progress=None, plan=None) -> str:
        del agent
        if mask is not None and plan is not None:
            raise InvalidInputError("mask and plan are mutually exclusive")
        if plan is not None and self.task != "clean":
            raise InvalidInputError("plan is only supported for the clean task")
        self._last_warnings = []
        self._active_plan = None
        self._task_impl.frame_mask = None
        os.makedirs(output, exist_ok=True)
        started = time.monotonic()
        benchmark = {"video_path": video, "timing": {}, "mask_source": "manual" if mask else "auto"}
        if mask is not None:
            mask_arr = read_mask(mask)
            mask_path = mask
        elif self.task == "clean":
            detected_start = time.monotonic()
            wipe_plan = self._resolve_clean_plan(plan, video, detector, targets, intent, regions, detect_mode, ocr, output, confirm, detector_mode)
            self._last_warnings = list(wipe_plan.warnings)
            if plan is None and not wipe_plan.remove_tracks:
                passthrough = os.path.join(output, f"{os.path.splitext(os.path.basename(video))[0]}_clean.mp4")
                shutil.copy2(video, passthrough)
                print("[videoclean] no overlays found; copied input unchanged")
                return passthrough
            mask_arr = self._union_mask_from_plan(wipe_plan, (wipe_plan.source.height, wipe_plan.source.width))
            mask_path = os.path.join(output, "auto_mask.png")
            if not cv2.imwrite(mask_path, np.clip(mask_arr * 255, 0, 255).astype(np.uint8)):
                raise OSError(f"Failed to write image: {mask_path}")
            benchmark["timing"]["detection_s"] = round(time.monotonic() - detected_start, 3)
            if preview:
                print(f"Preview saved to {output}")
                return output
        else:
            from videoclean.detect import detect_subtitle_mask
            mask_arr = detect_subtitle_mask(video, detector=detector or self._detector)
            mask_path = os.path.join(output, "auto_mask.png")
            cv2.imwrite(mask_path, np.clip(mask_arr * 255, 0, 255).astype(np.uint8))
        if preview:
            print(f"Preview saved to {output}")
            return output
        if self._active_plan is not None:
            validate_plan(self._active_plan, require_remove=True)
        self._ensure_model()
        reader, info = read_frame_info(video)
        try:
            validate_mask_shape(mask_arr, info)
            benchmark.update(width=info["W_ori"], height=info["H_ori"], frame_count=info["len"], fps=round(info["fps"], 2))
            benchmark["mask_area_ratio"] = round(float(np.sum(mask_arr > 0)) / (info["H_ori"] * info["W_ori"]), 6)
            self._task_impl._bm = benchmark
            self._task_impl.mask_path = mask_path
            out_path = self._task_impl.process_video(reader, info, mask_arr, output, video_path=video, progress=progress)
            benchmark["backend"] = "opencv"
            return out_path
        finally:
            reader.release()
            benchmark["timing"]["total_s"] = round(time.monotonic() - started, 3)
            benchmark["output_path"] = locals().get("out_path")
            benchmark["error"] = None
            try:
                with open(os.path.join(output, "benchmark.json"), "w", encoding="utf-8") as handle:
                    json.dump(benchmark, handle, indent=2)
            except OSError:
                pass

    def cleanup(self):
        if not self._run_lock.acquire(blocking=False):
            raise ProcessingError("Cannot clean up a WipeEngine while a request is running", code="ENGINE_BUSY", retryable=True)
        try:
            self._task_impl.cleanup()
        finally:
            self._run_lock.release()

    @staticmethod
    def _build_recognizer(ocr_mode):
        if ocr_mode == "off":
            return None
        try:
            from videoclean.ocr import recognize_text, _get_engine
            _get_engine()
            return recognize_text
        except Exception:
            if ocr_mode == "rapidocr":
                raise RuntimeError("OCR mode 'rapidocr' requested but rapidocr-onnxruntime is not installed. Install it with: pip install videoclean[ocr]")
            return None

    def _resolve_clean_plan(self, plan, video, detector, targets, intent, regions, detect_mode, ocr, output, confirm, detector_mode):
        if plan is not None:
            return self._resolve_plan_argument(plan, video)
        result, selected_ids, snapshot, directed = self._detect_clean(video, detector, targets, intent, regions, detect_mode, ocr, output, confirm, detector_mode)
        wipe_plan = self._build_fresh_clean_plan(video, result, selected_ids, snapshot, directed)
        self._write_final_clean_artifacts(result, wipe_plan, output)
        save_wipe_plan(wipe_plan, output)
        return wipe_plan

    def _build_fresh_clean_plan(self, video, result, selected_ids, snapshot, directed):
        selection = self._explicit_selection_kwargs(result.candidates, selected_ids, directed)
        effective_mode = snapshot.get("effective_detect_mode", snapshot["detect_mode"])
        return build_refined_wipe_plan(
            video, result, compute_source(video), refine=effective_mode != "fast",
            request=snapshot, progress=lambda done, total: self._emit_progress(ProgressEvent("refine", done, total)),
            check_cancelled=self._check_cancelled, **selection,
        )

    @staticmethod
    def _write_final_clean_artifacts(result, wipe_plan, output):
        from videoclean.detect import write_clean_artifacts
        remove_ids = {track.id for track in wipe_plan.remove_tracks}
        write_clean_artifacts(result, [candidate for candidate in result.candidates if candidate.id in remove_ids], output)

    @staticmethod
    def _explicit_selection_kwargs(candidates, selected_ids, user_directed):
        if not user_directed:
            return {"explicit_remove_ids": set(), "explicit_keep_ids": set()}
        selected = set(selected_ids)
        return {"explicit_remove_ids": selected, "explicit_keep_ids": {candidate.id for candidate in candidates} - selected}

    def _detect_clean(self, video, detector, targets, intent, regions, detect_mode, ocr, output, confirm, detector_mode):
        from videoclean.detect import detect_clean_candidates, infer_regions_from_text, infer_targets_from_text, normalize_target, resolve_detect_params, select_clean_candidates, write_clean_artifacts
        targets = list(targets or [])
        requested_regions = list(regions or [])
        intent_text = " ".join([*targets, intent or ""])
        requested_regions.extend(infer_regions_from_text(intent_text))
        requested_regions = list(dict.fromkeys(requested_regions))
        effective_targets = list(targets) + infer_targets_from_text(" ".join(targets))
        if requested_regions:
            effective_targets.append("region")
        effective_targets = list(dict.fromkeys(effective_targets))
        normalized = {normalize_target(target) for target in effective_targets}
        detect_text = not requested_regions and not normalized or bool(normalized & {"subtitle", "timestamp", "watermark", "scene_text", "unknown_text"}) or bool(intent and not normalized)
        requested_mode = detect_mode or self._detect_mode
        if requested_mode not in {"auto", "fast", "balanced", "sensitive"}:
            raise ValueError("detect_mode must be auto, fast, balanced, or sensitive")
        recognizer = self._build_recognizer(ocr or self._ocr)
        mode = detector_mode or "dbnet"
        if mode not in {"dbnet", "hybrid"}:
            raise ValueError("detector_mode must be 'dbnet' or 'hybrid'")
        if detector is None:
            from videoclean.detect import HybridTextDetector, _default_detector
            detector = _default_detector()
            if mode == "hybrid":
                detector = HybridTextDetector(detector)
        def run_pass(mode_name):
            params = resolve_detect_params(mode_name, has_subtitle_target="subtitle" in normalized)
            self._emit_progress(ProgressEvent("detect", 0, params["sample_count"], message=f"Scanning {params['sample_count']} sampled frames ({mode_name})"))
            detected = detect_clean_candidates(
                video, detector=detector, regions=requested_regions,
                detect_text=detect_text, include_logo="logo" in normalized,
                include_translucent_watermark="watermark" in normalized,
                sample_count=params["sample_count"], consistency=params["consistency"],
                subtitle_fallback=params["subtitle_fallback"], recognizer=recognizer,
            )
            chosen = select_clean_candidates(detected.candidates, targets=effective_targets, intent=intent)
            return detected, chosen, mode_name

        if requested_mode == "auto":
            result, selected, effective_mode = run_pass("fast")
            if not result.candidates:
                print("[videoclean] no candidates in fast mode; retrying sensitive mode")
                self._emit_progress(ProgressEvent("detect", 0, 1, message="Retrying with sensitive mode"))
                result, selected, effective_mode = run_pass("sensitive")
        else:
            result, selected, effective_mode = run_pass(requested_mode)
        print(f"[videoclean] {effective_mode} mode found {len(result.candidates)} candidate(s); selected {len(selected)}")
        if not result.candidates:
            print("[videoclean] no text or overlay candidates were detected")
        write_clean_artifacts(result, selected, output)

        if confirm:
            selected = self._confirm_candidates(result.candidates, selected)
            write_clean_artifacts(result, selected, output)
        self._emit_progress(ProgressEvent("detect", 1, 1))
        snapshot = {
            "intent": intent, "targets": effective_targets, "regions": requested_regions,
            "detect_mode": requested_mode, "effective_detect_mode": effective_mode,
            "ocr": ocr or self._ocr, "detector_mode": mode,
        }
        return result, {candidate.id for candidate in selected}, snapshot, bool(targets or intent or regions or confirm)

    @staticmethod
    def _confirm_candidates(candidates, selected):
        selected_ids = {candidate.id for candidate in selected}
        for candidate in candidates:
            marker = "*" if candidate.id in selected_ids else " "
            print(f"{marker} {candidate.id}: {candidate.label} ({candidate.reason}, confidence {candidate.confidence:.2f})")
        answer = input("Remove candidate ids separated by commas, press Enter to accept, or type 'none': ").strip()
        if not answer:
            return list(selected)
        if answer.lower() in {"none", "cancel", "no"}:
            return []
        wanted = {item.strip() for item in answer.split(",") if item.strip()}
        return [candidate for candidate in candidates if candidate.id in wanted]

    @staticmethod
    def _resolve_plan_argument(plan, video):
        if isinstance(plan, WipePlan):
            validate_plan(plan, require_remove=True)
            actual = compute_source(video)
            if actual.sha256 != plan.source.sha256 or (actual.width, actual.height, actual.frame_count) != (plan.source.width, plan.source.height, plan.source.frame_count):
                raise InvalidInputError("plan source does not match the video")
            return plan
        return load_wipe_plan(os.fspath(plan), video_path=video)

    def _union_mask_from_plan(self, plan, frame_shape):
        from types import SimpleNamespace
        from videoclean.detect import mask_from_candidates
        adapters = []
        for track in plan.remove_tracks:
            if track.mask is None:
                raise InvalidInputError(f"remove track {track.id} has no precise mask; cannot build union")
            arr = np.asarray(track.mask)
            adapters.append(SimpleNamespace(mask=arr[:, :, None] if arr.ndim == 2 else arr, bbox=track.bbox))
        return mask_from_candidates(adapters, frame_shape, feather_radius=self._task_impl.feather_radius)


def remove_text(video: str, mask: str | None = None, output: str = "result/", gap: int = 10, dual: bool = False, detector=None) -> str:
    """Remove hardcoded subtitles using OpenCV inpainting."""
    with WipeEngine(task="detext", gap=gap, dual=dual, detector=detector) as engine:
        return engine.process(video=video, mask=mask, output=output)
