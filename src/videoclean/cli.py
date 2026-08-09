"""CLI entry point for videoclean."""
import argparse
import sys

from videoclean.api import ProgressEvent, WipeRequest
from videoclean.engine import WipeEngine


def _print_progress(event: ProgressEvent) -> None:
    labels = {
        "prepare": "Preparing",
        "detect": "Detecting overlay text",
        "candidates": "Analyzing candidates",
        "mask": "Building removal mask",
        "plan": "Building removal plan",
        "refine": "Refining temporal mask",
        "persist": "Saving detection artifacts",
        "inpaint": "Reconstructing background",
        "validate": "Validating output",
        "complete": "Complete",
        "error": "Failed",
    }
    label = labels.get(event.phase, event.phase.title())
    suffix = f" — {event.message}" if event.message else ""
    if event.total > 1:
        # Avoid flooding the terminal for per-frame refinement events.
        if event.phase == "refine" and not event.message:
            return
        print(f"[videoclean] {label}: {event.completed}/{event.total}{suffix}", flush=True)
    else:
        print(f"[videoclean] {label}{suffix}", flush=True)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="videoclean",
        description="OpenCV-only removal of hardcoded subtitles and overlays",
    )
    subparsers = parser.add_subparsers(dest="command")

    clean = subparsers.add_parser("clean", help="Detect and remove subtitles or overlays")
    clean.add_argument("video", help="Input video path")
    mask_or_plan = clean.add_mutually_exclusive_group()
    mask_or_plan.add_argument("-m", "--mask", help="Mask image path; skip detection")
    mask_or_plan.add_argument("--plan", help="Existing wipe_plan.json")
    clean.add_argument("-o", "--output", default="result/", help="Output directory")
    clean.add_argument("-g", "--gap", type=int, default=10, help="Compatibility segment-size option")
    clean.add_argument("--verbose", action="store_true")
    clean.add_argument("-d", "--dual", action="store_true", help="Show original beside cleaned video")
    clean.add_argument("--target", action="append", default=None)
    clean.add_argument("--region", action="append", default=None)
    clean.add_argument("--intent", default=None)
    clean.add_argument("--preview", action="store_true")
    clean.add_argument("--confirm", action="store_true")
    clean.add_argument("--detect-mode", choices=["auto", "fast", "balanced", "sensitive"], default="auto")
    clean.add_argument("--ocr", choices=["auto", "off", "rapidocr"], default="auto")
    clean.add_argument("--detector", dest="detector_mode", choices=["dbnet", "hybrid"], default="dbnet")
    clean.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    clean.add_argument("--inpaint-radius", type=float, default=3.0)
    clean.add_argument("--inpaint-dilate", type=int, default=0)
    clean.add_argument("--inpaint-model", choices=["opencv", "adaptive"], default="opencv")

    # Keep detext as a small compatibility alias for explicit-mask workflows.
    detext = subparsers.add_parser("detext", help="Alias for clean")
    detext.add_argument("-v", "--video", required=True)
    detext.add_argument("-m", "--mask", default=None)
    detext.add_argument("-o", "--output", default="result/")
    detext.add_argument("-g", "--gap", type=int, default=10)
    detext.add_argument("-d", "--dual", action="store_true")
    detext.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    detext.add_argument("--inpaint-radius", type=float, default=3.0)
    detext.add_argument("--inpaint-dilate", type=int, default=0)
    detext.add_argument("--inpaint-model", choices=["opencv", "adaptive"], default="opencv")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "detext":
        args.command = "detext"
        args.detect_mode = "fast"
        args.ocr = "off"
        args.detector_mode = "dbnet"
        args.target = None
        args.region = None
        args.intent = None
        args.preview = False
        args.confirm = False
        args.plan = None
        args.verbose = False

    engine = WipeEngine(
        task=args.command,
        gap=args.gap,
        dual=args.dual,
        model=getattr(args, "inpaint_model", "opencv"),
        model_options={
            "method": args.inpaint_method,
            "radius": args.inpaint_radius,
            "dilate": args.inpaint_dilate,
        },
        detect_mode=getattr(args, "detect_mode", "balanced"),
        ocr=getattr(args, "ocr", "auto"),
        verbose=getattr(args, "verbose", False),
    )
    try:
        engine.run(
            WipeRequest(
                video=args.video,
                mask=args.mask,
                output_dir=args.output,
                targets=getattr(args, "target", None) or (),
                intent=getattr(args, "intent", None),
                agent=getattr(args, "agent", None),
                regions=getattr(args, "region", None) or (),
                preview=getattr(args, "preview", False),
                confirm=getattr(args, "confirm", False),
                plan=getattr(args, "plan", None),
                detector_mode=getattr(args, "detector_mode", "dbnet"),
            ),
            on_progress=_print_progress,
        )
    except Exception as exc:
        print(f"videoclean: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
