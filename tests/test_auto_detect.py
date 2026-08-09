from types import SimpleNamespace

from videoclean.cli import _build_parser
from videoclean.engine import WipeEngine


def test_cli_defaults_to_auto_detection():
    args = _build_parser().parse_args(["clean", "input.mp4"])
    assert args.detect_mode == "auto"


def test_explicit_modes_remain_available():
    for mode in ("fast", "balanced", "sensitive"):
        args = _build_parser().parse_args(["clean", "input.mp4", "--detect-mode", mode])
        assert args.detect_mode == mode


def test_auto_mode_runs_balanced_then_sensitive(monkeypatch, tmp_path):
    empty = SimpleNamespace(candidates=[])
    found = SimpleNamespace(candidates=[])
    calls = []

    def fake_detect(*args, **kwargs):
        calls.append(kwargs["sample_count"])
        return empty if len(calls) == 1 else found

    monkeypatch.setattr("videoclean.detect.detect_clean_candidates", fake_detect)
    monkeypatch.setattr("videoclean.detect._default_detector", lambda: object())
    monkeypatch.setattr("videoclean.detect.write_clean_artifacts", lambda *args: None)
    monkeypatch.setattr("videoclean.engine.WipeEngine._build_recognizer", staticmethod(lambda mode: None))
    engine = WipeEngine(task="clean", detect_mode="auto", ocr="off")
    result, selected, snapshot, directed = engine._detect_clean(
        "input.mp4", object(), [], None, [], None, "off", str(tmp_path), False, "dbnet"
    )
    assert calls == [24, 80]
    assert snapshot["detect_mode"] == "auto"
    assert snapshot["effective_detect_mode"] == "sensitive"
    assert selected == set()
    assert directed is False


def test_explicit_fast_does_not_retry(monkeypatch, tmp_path):
    calls = []

    def fake_detect(*args, **kwargs):
        calls.append(kwargs["sample_count"])
        return SimpleNamespace(candidates=[])

    monkeypatch.setattr("videoclean.detect.detect_clean_candidates", fake_detect)
    monkeypatch.setattr("videoclean.detect._default_detector", lambda: object())
    monkeypatch.setattr("videoclean.detect.write_clean_artifacts", lambda *args: None)
    monkeypatch.setattr("videoclean.engine.WipeEngine._build_recognizer", staticmethod(lambda mode: None))
    engine = WipeEngine(task="clean", detect_mode="fast", ocr="off")
    _, _, snapshot, _ = engine._detect_clean(
        "input.mp4", object(), [], None, [], None, "off", str(tmp_path), False, "dbnet"
    )
    assert calls == [24]
    assert snapshot["effective_detect_mode"] == "fast"
