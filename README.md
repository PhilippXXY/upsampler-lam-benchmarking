<h1 align="center">Upsampler-LAM Benchmarking</h1>

<p align="center">
  <img src="assets/img/logo_qmul.jpg" alt="QMUL" height="96">
</p>

<p align="center">
  Benchmarking 4-to-32 channel acoustic upsamplers in front of Latent Acoustic Mapping (LAM)
  for direction-of-arrival estimation.
</p>

<p align="center">
  <a href="https://philippxxy.github.io/upsampler-lam-benchmarking/">
    <img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-222222">
  </a>
  <a href="https://github.com/PhilippXXY/upsampler-lam-benchmarking/releases">
    <img alt="Release" src="https://img.shields.io/github/v/release/PhilippXXY/upsampler-lam-benchmarking">
  </a>
  <img alt="Python 3.13.3" src="https://img.shields.io/badge/python-3.13.3-3776AB?logo=python&logoColor=white">
</p>

## What This Repository Covers

- Single-model inference on LOCATA and STARSS23
- Batch benchmarking with latency, peak-memory, GFLOPs, localisation, and correlation matrix distance outputs
- Training pipelines for standalone upsamplers and upsampler-plus-LAM wrapper models
- Reproducible documentation, config reference, and Docker entrypoints

The full documentation site lives at [philippxxy.github.io/upsampler-lam-benchmarking](https://philippxxy.github.io/upsampler-lam-benchmarking/).

## Quick Start

Requirements: Python `3.13.3`, [`uv`](https://docs.astral.sh/uv/). Docker is optional.

1. Create the environment.

```bash
pip install uv
uv python install 3.13.3
uv venv --python 3.13.3
source .venv/bin/activate
uv sync --all-groups
```

2. Point [`config/inference_config.yaml`](config/inference_config.yaml) at a real dataset root.
   If you want to use the checked-in LOCATA subset, set:

```yaml
inference:
  data_set: "locata"

dataset:
  data_audio_path: "data/locata/dev"
```

For STARSS23 runs, also set `dataset.data_ground_truth_path` to the matching metadata directory.

3. Run one model or a full comparison.

```bash
# Single-model inference
uv run python src/infer.py --config config/inference_config.yaml --device cpu

# Multi-model benchmarking
uv run python src/evaluate_inference.py --base-config config/inference_config.yaml --device cpu
```

Outputs are written under `output/`, with logs under `logs/`.

## Main Entry Points

| Path | Purpose |
| --- | --- |
| `src/infer.py` | Run one configured model on one dataset |
| `src/evaluate_inference.py` | Benchmark retained variants and generate comparison plots |
| `scripts/docker-infer.sh` | Run inference or benchmarking inside Docker |
| `src/train_upsamplers.py` | Train standalone upsamplers |
| `src/train_end_to_end.py` | Train end-to-end wrapper models |

## Attribution and Data Provenance

This project was developed as part of an MSc thesis at Queen Mary University of London.
The [`src/lam_min/`](src/lam_min/) subtree adapts the original [LAM](https://github.com/adrianSRoman/LAM) implementation by Roman et al. and is treated here as upstream methodology rather than a repository-native model
contribution.

Licensing and provenance for bundled upstream code and datasets are documented in the corresponding
subdirectories:

- [`src/lam_min/README.md`](src/lam_min/README.md)
- [`data/locata/README.md`](data/locata/README.md)
- [`data/starss23/README.md`](data/starss23/README.md)

For citations used throughout the project, see [`docs/references.md`](docs/references.md).
