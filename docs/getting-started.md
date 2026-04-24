# Getting Started

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.13.3 | Pinned in `pyproject.toml` |
| [`uv`](https://docs.astral.sh/uv/) | Environment and dependency management |
| Docker | Optional — only for the container workflow |
| Dataset access | Required for inference and training |

## Installation

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync --all-groups
```

Preview the docs locally:

```bash
uv run --group dev mkdocs serve
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/infer.py` | Single-run inference |
| `src/evaluate_inference.py` | Batch benchmarking |
| `src/train_upsamplers.py` | Standalone upsampler training |
| `src/train_end_to_end.py` | End-to-end wrapper training |
| `src/data/` | Dataset loaders |
| `src/upsampler/` | Standalone upsampler implementations |
| `src/lam_min/` | LAM-derived code and wrapper models |
| `config/` | YAML config files |
| `scripts/docker-infer.sh` | Docker wrapper for inference |

## Choose a Workflow

| Goal | Page |
| --- | --- |
| Run one model on one dataset | [Local Inference](workflows/local-inference.md) |
| Skip local setup | [Docker Inference](workflows/docker-inference.md) |
| Compare models and generate plots | [Batch Benchmarking](workflows/batch-benchmarking.md) |
| Train a standalone upsampler | [Upsampler Training](workflows/upsampler-training.md) |
| Train a combined wrapper model | [End-to-End Training](workflows/end-to-end-training.md) |

## Datasets

| Dataset | Used for | Reference |
| --- | --- | --- |
| LOCATA | 32-ch inference, LAM baseline | [R3](references.md#r3) |
| STARSS23 | 4-ch inference | [R4](references.md#r4) |
| AudibleLight | Training | [R5](references.md#r5) |
| EigenScape raw | Training | [R6](references.md#r6) |

Download scripts:

```bash
bash data/locata/download_data.sh
bash data/starss23/download_data.sh
bash data/AudibleLight_Eigenmike32-5_DCASE-STARSS23_Dataset/download_data.sh
bash data/eigenscape/download_data.sh
```
