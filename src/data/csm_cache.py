"""Helpers for persisting precomputed CSM tensors on disk."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch


def _stable_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]  # noqa: S324


def build_precomputed_csm_dir(  # noqa: PLR0913
    root: Path,
    *,
    dataset_name: str,
    dataset_root: Path,
    sampling_rate: int,
    nbands: int,
    low_channel_indices: tuple[int, ...],
    extra_tag: str = "",
) -> Path:
    """
    Build a config-specific directory for precomputed CSM tensors.

    Parameters
    ----------
    root
        Base directory for all precomputed CSM caches.
    dataset_name
        Name of the dataset.
    dataset_root
        Root path of the dataset (used for stable hashing).
    sampling_rate
        Sampling rate of the audio (for tagging).
    nbands
        Number of frequency bands in the CSM (for tagging).
    low_channel_indices
        Tuple of channel indices considered "low" for the CSM (for tagging).
    extra_tag
        Optional extra string tag to differentiate configurations (for tagging).

    Returns
    -------
    Path
        Directory path for the precomputed CSM tensors of the specified configuration.
    """
    low_tag = "-".join(str(index) for index in low_channel_indices)
    dataset_root_tag = _stable_token(str(dataset_root.resolve()))
    suffix = f"sr{sampling_rate}_nb{nbands}_low{low_tag}"
    if extra_tag:
        suffix = f"{suffix}_{extra_tag}"
    return root.joinpath(dataset_name, f"{dataset_root_tag}_{suffix}")


def build_precomputed_csm_path(cache_dir: Path, file_id: str) -> Path:
    """
    Map a file id to its cache file path.

    Parameters
    ----------
    cache_dir
        Directory where precomputed CSM tensors are stored.
    file_id
        Unique identifier for the audio file (e.g., relative path or filename).

    Returns
    -------
    Path
        Full path to the cached CSM tensor for the given file id.
    """
    return cache_dir.joinpath(f"{_stable_token(file_id)}.pt")


def load_precomputed_csm_pair(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load a cached low/high CSM tensor pair if it exists.

    Parameters
    ----------
    cache_path
        Path to the cached CSM tensor file.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | None
        The cached low and high CSM tensors, or None if the file does not exist.
    """
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    return payload["S_low"], payload["S_high"]


def save_precomputed_csm_pair(
    cache_path: Path, *, S_low: torch.Tensor, S_high: torch.Tensor
) -> None:
    """
    Atomically save a low/high CSM tensor pair to disk.

    Parameters
    ----------
    cache_path
        Path to the cache file.
    S_low
        The low CSM tensor to save.
    S_high
        The high CSM tensor to save.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"S_low": S_low.cpu(), "S_high": S_high.cpu()}, tmp_path)
    tmp_path.replace(cache_path)
