# videoclean

Open-source, CPU-friendly video cleanup for hardcoded subtitles, watermarks, logos, and timestamps.

`videoclean` uses OpenCV's Telea or Navier–Stokes inpainting with a stable mask applied to every frame. It has no PyTorch, ONNX Runtime, GPU, or model-weight dependency for inpainting. FFmpeg must be installed and available on `PATH`.

Automatic detection uses the separately distributed PP-OCRv5 detector weight. The weight is Apache-2.0 licensed and hosted at [Hugging Face](https://huggingface.co/stevenlearns/videoclean-detector); see [NOTICE](NOTICE), [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0), and [PROVENANCE.md](PROVENANCE.md). Manual masks work without downloading detector weights.

The source code is GPLv3; third-party detector weights retain their separate license.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

Canonical source: [GitHub](https://github.com/oohfixer/videoclean).


## Install

```bash
pip install videoclean
```

Optional OCR support:

```bash
pip install 'videoclean[ocr]'
```

## Usage

```bash
videoclean clean input.mp4 -o output-directory
vpipe -y --clean input.mp4
```

For a reviewed mask:

```bash
videoclean clean input.mp4 -m mask.png -o output-directory \
  --inpaint-method telea --inpaint-radius 5 --inpaint-dilate 2
```

No-candidate detection is a successful passthrough: the input is copied to the output directory unchanged. Use `--preview` to inspect detection artifacts before processing.

## License

GPLv3. See `LICENSE`.
