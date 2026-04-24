"""
EigenScape raw audio-to-CSM pair dataset for upsampler training.

This training dataset returns low-resolution and high-resolution complex CSM
pairs generated from raw multichannel WAV audio. It supports class-stratified
train/val/test splitting and 48k -> 24k resampling.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from data.csm_cache import (
    build_precomputed_csm_dir,
    build_precomputed_csm_path,
    load_precomputed_csm_pair,
    save_precomputed_csm_pair,
)
from lam_min.dataset.gen_dataset.gen_dataset import get_visibility_matrix

CLASS_REGEX = re.compile(
    r"^(Beach|BusyStreet|Park|PedestrianZone|QuietStreet|ShoppingCentre|TrainStation|Woodland)[-.](\d{1,2})"
)
SPLIT_PARTS = 3


def _parse_class_and_index(file_stem: str) -> tuple[str, int]:
    """
    Parse EigenScape class name and clip index from WAV stem.

    Parameters
    ----------
    file_stem : str
        Stem of the WAV file name, e.g. "Beach-01-Raw".

    Returns
    -------
    tuple[str, int]
        Tuple containing class name (e.g. "Beach") and clip index (e.g. 1).
        If parsing fails, returns class name as the first part of the stem and -1 as index.
    """
    stem = file_stem.replace("-Raw", "")
    match = CLASS_REGEX.match(stem)
    if match is None:
        class_name = stem.split("-")[0].split(".")[0]
        return class_name, -1
    class_name = match.group(1)
    clip_index = int(match.group(2))
    return class_name, clip_index


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:  # type: ignore[type-arg]
    """
    Resample multichannel audio with polyphase filtering.

    Parameters
    ----------
    audio : np.ndarray
        Multichannel audio array of shape (num_samples, num_channels).
    orig_sr : int
        Original sampling rate of the audio data.
    target_sr : int
        Target sampling rate for resampling.

    Returns
    -------
    np.ndarray
        Resampled audio array of shape (num_samples_resampled, num_channels).
    """
    if orig_sr == target_sr:
        return audio
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Invalid sampling rates orig={orig_sr}, target={target_sr}")

    # Resample each channel along time axis.
    up = target_sr
    down = orig_sr
    resampled = resample_poly(audio, up=up, down=down, axis=0)
    return resampled.astype(np.float32, copy=False)  # type: ignore[no-any-return]


def _sanitise_complex_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Replace NaN/Inf values in complex tensors.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor to sanitise.

    Returns
    -------
    torch.Tensor
        Sanitised complex tensor with NaN/Inf values replaced by 0.
    """
    real = torch.nan_to_num(x.real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = torch.nan_to_num(x.imag, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.complex(real, imag)


def _normalise_split_ratio(split_ratio: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Validate and normalise split ratios.

    Parameters
    ----------
    split_ratio : tuple[float, float, float]
        Train/val/test ratio triplet.

    Returns
    -------
    tuple[float, float, float]
        Normalised ratio triplet summing to 1.
    """
    if len(split_ratio) != SPLIT_PARTS:
        raise ValueError(f"split_ratio must contain {SPLIT_PARTS} values, got {split_ratio}")
    if any(float(value) < 0.0 for value in split_ratio):
        raise ValueError(f"split_ratio must contain non-negative values, got {split_ratio}")
    total = float(sum(split_ratio))
    if total <= 0.0:
        raise ValueError(f"split_ratio must sum to a positive value, got {split_ratio}")
    normalised = tuple(float(value) / total for value in split_ratio)
    return normalised[0], normalised[1], normalised[2]


def _allocate_split_counts(
    total_items: int, split_ratio: tuple[float, float, float]
) -> tuple[int, int, int]:
    """
    Allocate per-split counts from ratios using largest remainder.

    Parameters
    ----------
    total_items : int
        Number of files available for one class after filtering.
    split_ratio : tuple[float, float, float]
        Normalised train/val/test ratios.

    Returns
    -------
    tuple[int, int, int]
        Integer train/val/test counts summing to total_items.
    """
    if total_items <= 0:
        return 0, 0, 0

    raw = [float(total_items) * ratio for ratio in split_ratio]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total_items - sum(counts)

    order = sorted(
        range(SPLIT_PARTS),
        key=lambda idx: ((raw[idx] - counts[idx]), -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


class EigenscapeCSMPairDataset(Dataset[dict[str, Any]]):
    """
    Training dataset for raw EigenScape multichannel audio.

    This dataset returns low-resolution and high-resolution complex CSM pairs
    generated from raw multichannel WAV audio.
    It supports class-stratified train/val/test splitting and 48k -> 24k resampling.

    Attributes
    ----------
    root_path : Path
        Path to EigenScape root. WAV files are searched recursively.
    split : str, optional
        One of "train", "val", "test" (default: "train").
    split_counts : tuple[int, int, int] | None, optional
        Per-class counts for train/val/test split (default: (6, 1, 1)).
        The sum determines how many files are required per class after filtering.
    split_ratio : tuple[float, float, float] | None, optional
        Per-class train/val/test ratios. If set, ratios are applied independently
        to each class after filtering and all available files are assigned to one
        split (default: None). This is useful when classes have different file counts.
    seed : int, optional
        Seed for deterministic class-wise split (default: 42).
    low_channel_indices : tuple[int, int, int, int], optional
        Low-channel indices to use from Eigenmike32 (default: (5, 9, 21, 25)).
    sampling_rate : int, optional
        Target sampling rate for processing (default: 24000).
    nbands : int, optional
        Number of frequency bands for CSM computation (default: 9).
    cache_csm : bool, optional
        Whether to cache computed CSM tensors in memory (default: False).
    max_files : int, optional
        Optional cap on number of files (0 = no cap).
    expected_channels : int | None, optional
        Expected number of channels for Eigenmike raw CSM targets
        (default: 32). Set to None to disable strict channel filtering.
    target_high_channels : int, optional
        Target number of channels for high-resolution CSM targets.
        If a file has fewer channels and fallback is enabled, zero-value channels
        are padded to this count (default: 32).
    allow_channel_fallback : bool, optional
        If True, allow files that do not meet strict channel requirements.
        Missing low-channel indices are clamped to the highest available channel,
        and high-resolution targets are padded/truncated to target_high_channels.
        This is useful for mixed-channel EigenScape files (default: False).
    """

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        root_path: Path,
        split: str = "train",
        split_counts: tuple[int, int, int] | None = (6, 1, 1),
        split_ratio: tuple[float, float, float] | None = None,
        seed: int = 42,
        low_channel_indices: tuple[int, int, int, int] = (5, 9, 21, 25),
        sampling_rate: int = 24000,
        nbands: int = 9,
        cache_csm: bool = False,
        precomputed_csm_root: Path | None = None,
        max_files: int = 0,
        expected_channels: int | None = 32,
        target_high_channels: int = 32,
        allow_channel_fallback: bool = False,
    ) -> None:
        """
        Initialise EigenscapeCSMPairDataset.

        Parameters
        ----------
        root_path : Path
            Path to EigenScape root. WAV files are searched recursively.
        split : str, optional
            One of "train", "val", "test" (default: "train").
        split_counts : tuple[int, int, int] | None, optional
            Per-class counts for train/val/test split (default: (6, 1, 1)).
            The sum determines how many files are required per class after filtering.
        split_ratio : tuple[float, float, float] | None, optional
            Per-class train/val/test ratios. If set, ratios are applied independently
            to each class after filtering and all available files are assigned to one
            split (default: None). This is useful when classes have different file counts.
        seed : int, optional
            Seed for deterministic class-wise split (default: 42).
        low_channel_indices : tuple[int, int, int, int], optional
            Low-channel indices to use from Eigenmike32 (default: (5, 9, 21, 25)).
        sampling_rate : int, optional
            Target sampling rate for processing (default: 24000).
        nbands : int, optional
            Number of frequency bands for CSM computation (default: 9).
        cache_csm : bool, optional
            Whether to cache computed CSM tensors in memory (default: False).
        max_files : int, optional
            Optional cap on number of files (0 = no cap).
        expected_channels : int | None, optional
            Expected number of channels for Eigenmike raw CSM targets
            (default: 32). Set to None to disable strict channel filtering.
        target_high_channels : int, optional
            Target number of channels for high-resolution CSM targets.
            If a file has fewer channels and fallback is enabled, zero-value channels
            are padded to this count (default: 32).
        allow_channel_fallback : bool, optional
            If True, allow files that do not meet strict channel requirements.
            Missing low-channel indices are clamped to the highest available channel,
            and high-resolution targets are padded/truncated to target_high_channels
            (default: False).
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got '{split}'")

        if split_ratio is not None:
            if split_counts is not None:
                logging.info(
                    "EigenscapeCSMPairDataset received both split_ratio and split_counts; "
                    "using split_ratio."
                )
            normalised_split_ratio = _normalise_split_ratio(split_ratio)
            validated_split_counts: tuple[int, int, int] | None = None
        else:
            if split_counts is None:
                raise ValueError("Either split_counts or split_ratio must be provided.")
            if len(split_counts) != SPLIT_PARTS:
                raise ValueError(
                    f"split_counts must contain {SPLIT_PARTS} values, got {split_counts}"
                )
            if any(int(count) < 0 for count in split_counts):
                raise ValueError(f"split_counts must be non-negative, got {split_counts}")
            if sum(split_counts) <= 0:
                raise ValueError(f"split_counts must sum to a positive value, got {split_counts}")
            normalised_split_ratio = None
            counts = [int(count) for count in split_counts]
            validated_split_counts = counts[0], counts[1], counts[2]

        self.root_path = Path(root_path)
        self.split = split
        self.split_counts = validated_split_counts
        self.split_ratio = normalised_split_ratio
        self.low_channel_indices = low_channel_indices
        self.sampling_rate = sampling_rate
        self.nbands = nbands
        self.cache_csm = cache_csm
        self.expected_channels = expected_channels
        self.target_high_channels = int(target_high_channels)
        self.allow_channel_fallback = bool(allow_channel_fallback)
        self._cache: dict[str, dict[str, Any]] = {}
        self._warned_low_channel_fallback = False
        self._warned_high_channel_fallback = False
        extra_tag = f"high{self.target_high_channels}_fallback{int(self.allow_channel_fallback)}"
        self.precomputed_csm_dir = (
            None
            if precomputed_csm_root is None
            else build_precomputed_csm_dir(
                Path(precomputed_csm_root),
                dataset_name="eigenscape",
                dataset_root=self.root_path,
                sampling_rate=self.sampling_rate,
                nbands=self.nbands,
                low_channel_indices=self.low_channel_indices,
                extra_tag=extra_tag,
            )
        )

        if self.target_high_channels <= 0:
            raise ValueError(
                f"target_high_channels must be positive, got {self.target_high_channels}"
            )

        if not self.root_path.exists():
            raise FileNotFoundError(f"EigenScape root path not found: {self.root_path}")

        wavs = sorted(self.root_path.rglob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files found under {self.root_path}")

        # As some eigenscape files have fewer than 32 channels,
        # we need to filter based on channel counts.
        required_low_channels = max(self.low_channel_indices) + 1
        sorted_low = sorted(self.low_channel_indices)
        fallback_required_low_channels = sorted_low[-2] + 1 if len(sorted_low) > 1 else 1
        channel_counts: Counter[int] = Counter()
        valid_wavs: list[tuple[Path, int]] = []
        for wav in wavs:
            info = soundfile.info(wav)
            n_channels = int(info.channels)
            channel_counts[n_channels] += 1
            minimum_required_channels = (
                fallback_required_low_channels
                if self.allow_channel_fallback
                else required_low_channels
            )
            if n_channels < minimum_required_channels:
                continue
            if (
                not self.allow_channel_fallback
                and self.expected_channels is not None
                and n_channels != self.expected_channels
            ):
                continue
            valid_wavs.append((wav, n_channels))

        if not valid_wavs:
            counts_text = ", ".join(
                f"{nch}ch={count}" for nch, count in sorted(channel_counts.items())
            )
            raise ValueError(
                "No EigenScape files match the configured channel requirements. "
                f"Found channel counts: [{counts_text}]. "
                f"Required low-channel indices need at least {required_low_channels} channels "
                f"and expected_channels={self.expected_channels}. "
                "This usually means you downloaded the ambisonic EigenScape release "
                "instead of eigenscape_raw (32-channel Eigenmike). "
                "Run data/eigenscape/download_data.sh to fetch eigenscape_raw."
            )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for wav, n_channels in valid_wavs:
            class_name, clip_index = _parse_class_and_index(wav.stem)
            grouped[class_name].append(
                {
                    "file_id": wav.stem,
                    "wav_path": wav,
                    "class_name": class_name,
                    "clip_index": clip_index,
                    "num_channels": n_channels,
                }
            )

        if self.split_ratio is None:
            required_per_class = sum(self.split_counts or (0, 0, 0))
            insufficient_classes = [
                f"{class_name}={len(entries)}"
                for class_name, entries in sorted(grouped.items())
                if len(entries) < required_per_class
            ]
            if insufficient_classes:
                raise ValueError(
                    "EigenScape class coverage is insufficient after channel filtering. "
                    f"Need {required_per_class} files per class but got: "
                    f"{', '.join(insufficient_classes)}. "
                    "This usually indicates a partial or non-raw EigenScape download."
                )

        rng = np.random.default_rng(seed)
        selected_entries: list[dict[str, Any]] = []

        # Perform class-wise splitting to ensure balanced representation across splits.
        for class_name, entries in grouped.items():  # noqa: B007
            entries_sorted = sorted(entries, key=lambda item: (item["clip_index"], item["file_id"]))
            rng.shuffle(entries_sorted)  # type: ignore[arg-type]

            if self.split_ratio is not None:
                n_train, n_val, n_test = _allocate_split_counts(
                    total_items=len(entries_sorted),
                    split_ratio=self.split_ratio,
                )
            else:
                n_train, n_val, n_test = self.split_counts or (0, 0, 0)

            train_entries = entries_sorted[:n_train]
            val_entries = entries_sorted[n_train : n_train + n_val]
            test_entries = entries_sorted[n_train + n_val : n_train + n_val + n_test]

            if split == "train":
                selected_entries.extend(train_entries)
            elif split == "val":
                selected_entries.extend(val_entries)
            else:
                selected_entries.extend(test_entries)

        self.entries = sorted(selected_entries, key=lambda item: item["file_id"])
        if max_files > 0:
            self.entries = self.entries[:max_files]

        if not self.entries:
            raise ValueError(
                f"No entries for split '{split}'. Check files and split settings "
                f"(split_counts/split_ratio) in {self.root_path}."
            )

    def __len__(self) -> int:
        """
        Return number of files in split.

        Returns
        -------
        int
            Number of files in the current split.
        """
        return len(self.entries)

    def _compute_csm_pair(self, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[type-arg]
        """
        Compute low-resolution and high-resolution CSM tensors from multichannel audio.

        Parameters
        ----------
        audio : np.ndarray
            Multichannel audio array of shape (num_samples, num_channels).
        sample_rate : int
            Sample rate of the audio data.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing low-resolution and high-resolution CSM tensors as PyTorch tensors.
        """
        n_channels = int(audio.shape[1])
        max_low_idx = max(self.low_channel_indices)
        if n_channels <= max_low_idx and not self.allow_channel_fallback:
            raise ValueError(
                f"Audio has {n_channels} channels but needs channel index {max_low_idx}"
            )

        low_indices = tuple(
            min(index, n_channels - 1) if self.allow_channel_fallback else index
            for index in self.low_channel_indices
        )
        if self.allow_channel_fallback and low_indices != self.low_channel_indices:
            if not self._warned_low_channel_fallback:
                logging.warning(
                    "EigenScape low-channel fallback active: requested indices %s, "
                    "using %s for %d-channel file(s).",
                    self.low_channel_indices,
                    low_indices,
                    n_channels,
                )
                self._warned_low_channel_fallback = True
        low_audio = audio[:, list(low_indices)]

        high_audio = audio
        if n_channels != self.target_high_channels:
            if not self.allow_channel_fallback:
                raise ValueError(
                    "EigenScape high-channel count mismatch: "
                    f"file has {n_channels} channels, target_high_channels="
                    f"{self.target_high_channels}. "
                    "Set allow_channel_fallback=True to pad/truncate safely."
                )
            if n_channels < self.target_high_channels:
                pad_channels = self.target_high_channels - n_channels
                high_audio = np.concatenate(
                    [
                        high_audio,
                        np.zeros((high_audio.shape[0], pad_channels), dtype=high_audio.dtype),
                    ],
                    axis=1,
                )
            else:
                high_audio = high_audio[:, : self.target_high_channels]

            if not self._warned_high_channel_fallback:
                logging.warning(
                    "EigenScape high-channel fallback active: file has %d channels, "
                    "target_high_channels=%d. High-resolution targets are padded/truncated.",
                    n_channels,
                    self.target_high_channels,
                )
                self._warned_high_channel_fallback = True

        # Get visibility matrices from LAM implementation,
        # ignoring warnings from numerical issues in CSM computation
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            S_low, _ = get_visibility_matrix(
                low_audio, fs=self.sampling_rate, apgd=False, nbands=self.nbands
            )
            S_high, _ = get_visibility_matrix(
                high_audio, fs=self.sampling_rate, apgd=False, nbands=self.nbands
            )

        # Convert to PyTorch tensors with shape (num_freq_bins, num_frames, num_mics, num_mics)
        S_low_t = (
            torch.from_numpy(np.ascontiguousarray(S_low)).permute(1, 0, 2, 3).to(torch.complex64)
        )
        S_high_t = (
            torch.from_numpy(np.ascontiguousarray(S_high)).permute(1, 0, 2, 3).to(torch.complex64)
        )
        S_low_t = _sanitise_complex_tensor(S_low_t)
        S_high_t = _sanitise_complex_tensor(S_high_t)
        return S_low_t, S_high_t

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return one file sample with low/high CSM tensors.

        This method reads the multichannel audio file at the given index, computes the
        low-resolution and high-resolution CSM tensors.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        dict[str, Any]
            Dictionary containing file_id, scene_id, dataset name, audio path,
            ground truth path (if available), and low/high CSM tensors.
        """
        entry = self.entries[index]
        file_id = str(entry["file_id"])

        if self.cache_csm and file_id in self._cache:
            return self._cache[file_id]

        wav_path = Path(entry["wav_path"])
        cache_path = (
            None
            if self.precomputed_csm_dir is None
            else build_precomputed_csm_path(self.precomputed_csm_dir, file_id)
        )
        cached_pair = None if cache_path is None else load_precomputed_csm_pair(cache_path)
        if cached_pair is not None:
            S_low_t, S_high_t = cached_pair
        else:
            audio, sample_rate = soundfile.read(wav_path, dtype="float32", always_2d=True)
            audio = _resample_audio(audio=audio, orig_sr=sample_rate, target_sr=self.sampling_rate)
            S_low_t, S_high_t = self._compute_csm_pair(audio=audio)
            if cache_path is not None:
                save_precomputed_csm_pair(cache_path, S_low=S_low_t, S_high=S_high_t)

        out = {
            "file_id": file_id,
            "dataset": "eigenscape",
            "class_name": str(entry["class_name"]),
            "audio_path": str(wav_path),
            "ground_truth_path": None,
            "has_dcase_gt": False,
            "S_low": S_low_t,
            "S_high": S_high_t,
        }

        if self.cache_csm:
            self._cache[file_id] = out

        return out
