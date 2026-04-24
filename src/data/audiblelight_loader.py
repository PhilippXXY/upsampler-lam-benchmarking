"""
AudibleLight Eigenmike32 audio-to-CSM pair dataset for upsampler training.

This training dataset returns low-resolution and high-resolution complex CSM
pairs generated from raw multichannel WAV audio.

References
----------
.. [1] AudibleLight Eigenmike32-5 DCASE-STARSS23 Dataset: https://doi.org/10.57967/hf/7810
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas
import soundfile
import torch
from torch.utils.data import Dataset

from data.csm_cache import (
    build_precomputed_csm_dir,
    build_precomputed_csm_path,
    load_precomputed_csm_pair,
    save_precomputed_csm_pair,
)
from lam_min.dataset.gen_dataset.gen_dataset import get_visibility_matrix

SCENE_ID_REGEX = re.compile(r"scene_(\d+)_mic\d+$")
SPLIT_RATIO_TOLERANCE = 1e-6


def _extract_scene_id(file_id: str) -> str:
    """
    Extract scene id from file id.

    The file_id format is expected to be "scene_{scene_id}_mic{mic_id}", e.g. "scene_01_mic01".

    Parameters
    ----------
    file_id : str
        File identifier string from which to extract the scene id.

    Returns
    -------
    str
        Extracted scene id as a string, e.g. "01" from "scene_01_mic01".
    """
    match = SCENE_ID_REGEX.match(file_id)
    if match is None:
        raise ValueError(f"Could not parse scene id from file_id='{file_id}'")
    return match.group(1)


def _split_scene_ids(
    scene_ids: list[str], split_ratio: tuple[float, float, float], seed: int
) -> dict[str, set[str]]:
    """
    Create deterministic train/val/test split at scene level.

    Parameters
    ----------
    scene_ids : list[str]
        List of unique scene identifiers to split.
    split_ratio : tuple[float, float, float]
        Desired train/val/test ratio over scene ids, e.g. (0.8, 0.1, 0.1).
    seed : int
        Seed for random shuffling to ensure reproducibility.

    Returns
    -------
        dict[str, set[str]]
            Dictionary mapping split name to set of scene identifiers in that split.
    """
    if abs(sum(split_ratio) - 1.0) > SPLIT_RATIO_TOLERANCE:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

    ordered = sorted(scene_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val

    train_ids = set(ordered[:n_train])
    val_ids = set(ordered[n_train : n_train + n_val])
    test_ids = set(ordered[n_train + n_val : n_train + n_val + n_test])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


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


class AudibleLightCSMPairDataset(Dataset[dict[str, Any]]):
    """
    Training dataset for AudibleLight Eigenmike32-5 DCASE-STARSS23.

    This dataset returns low-resolution and high-resolution complex CSM pairs
    generated from raw multichannel WAV audio.
    It supports class-stratified train/val/test splitting and 48k -> 24k resampling.

    Attributes
    ----------
    root_path : Path
        Path to the dataset root directory.
    split : str
        One of "train", "val", "test" (default: "train").
    low_channel_indices : tuple[int, int, int, int]
        Low-channel indices to use from Eigenmike32 (default: (5, 9, 21, 25)).
    sampling_rate : int
        Expected sample rate for processing (default: 24000).
    nbands : int
        Number of frequency bands for CSM computation (default: 9).
    cache_csm : bool
        Whether to cache computed CSM tensors in memory (default: False).
    max_files : int
        Optional cap on number of files (0 = no cap).
    """

    def __init__(  # noqa: PLR0913
        self,
        root_path: Path,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        low_channel_indices: tuple[int, int, int, int] = (5, 9, 21, 25),
        sampling_rate: int = 24000,
        nbands: int = 9,
        cache_csm: bool = False,
        precomputed_csm_root: Path | None = None,
        max_files: int = 0,
    ) -> None:
        """
        Initialise AudibleLightCSMPairDataset.

        Parameters
        ----------
        root_path : Path
            Path to the dataset root directory.
        split : str, optional
            One of "train", "val", "test" (default: "train").
        split_ratio : tuple[float, float, float], optional
            Train/val/test ratio over scene ids (default: (0.8, 0.1, 0.1)).
        seed : int, optional
            Seed for deterministic split (default: 42).
        low_channel_indices : tuple[int, int, int, int], optional
            Low-channel indices to use from Eigenmike32 (default: (5, 9, 21, 25)).
        sampling_rate : int, optional
            Expected sample rate for processing (default: 24000).
        nbands : int, optional
            Number of frequency bands for CSM computation (default: 9).
        cache_csm : bool, optional
            Whether to cache computed CSM tensors in memory (default: False).
        max_files : int, optional
            Optional cap on number of files (0 = no cap).
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got '{split}'")

        self.root_path = Path(root_path)
        self.split = split
        self.low_channel_indices = low_channel_indices
        self.sampling_rate = sampling_rate
        self.nbands = nbands
        self.cache_csm = cache_csm
        self._cache: dict[str, dict[str, Any]] = {}
        self.precomputed_csm_dir = (
            None
            if precomputed_csm_root is None
            else build_precomputed_csm_dir(
                Path(precomputed_csm_root),
                dataset_name="audiblelight",
                dataset_root=self.root_path,
                sampling_rate=self.sampling_rate,
                nbands=self.nbands,
                low_channel_indices=self.low_channel_indices,
            )
        )

        if not self.root_path.exists():
            raise FileNotFoundError(f"AudibleLight root path not found: {self.root_path}")

        metadata_csv = self.root_path.joinpath("em32_dev", "metadata.csv")
        wavs_dir = self.root_path.joinpath("em32_dev", "dev-train")
        gt_dir = self.root_path.joinpath("metadata_dev", "dev-train")

        if metadata_csv.exists():
            df = pandas.read_csv(metadata_csv)
            entries: list[dict[str, Any]] = []
            for row in df.itertuples(index=False):
                file_id = str(row.file_id)
                wav_path = self.root_path.joinpath(str(row.file_name))
                gt_path = self.root_path.joinpath(str(row.ground_truth_file_name))
                if wav_path.exists():
                    entries.append(
                        {
                            "file_id": file_id,
                            "scene_id": _extract_scene_id(file_id),
                            "wav_path": wav_path,
                            "ground_truth_path": gt_path if gt_path.exists() else None,
                        }
                    )
        # Fallback to directory scan if metadata CSV is not found
        else:
            wav_files = sorted(wavs_dir.glob("*.wav"))
            entries = []
            for wav_path in wav_files:
                file_id = wav_path.stem
                gt_path = gt_dir.joinpath(f"{file_id}.csv")
                entries.append(
                    {
                        "file_id": file_id,
                        "scene_id": _extract_scene_id(file_id),
                        "wav_path": wav_path,
                        "ground_truth_path": gt_path if gt_path.exists() else None,
                    }
                )

        if not entries:
            audio_dir = str(self.root_path.joinpath("em32_dev", "dev-train"))
            raise FileNotFoundError(f"No AudibleLight files found under {audio_dir}")

        scene_ids = sorted({entry["scene_id"] for entry in entries})
        split_sets = _split_scene_ids(scene_ids, split_ratio=split_ratio, seed=seed)
        selected_scene_ids = split_sets[split]
        self.entries = sorted(
            [entry for entry in entries if entry["scene_id"] in selected_scene_ids],
            key=lambda item: item["file_id"],
        )

        if max_files > 0:
            self.entries = self.entries[:max_files]

        if not self.entries:
            raise ValueError(
                f"No entries for split '{split}'. "
                f"Check split ratio/seed and dataset files in {self.root_path}."
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

    def _compute_csm_pair(
        self,
        audio: np.ndarray,  # type: ignore[type-arg]
        sample_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if sample_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate} Hz, got {sample_rate} Hz"
            )

        max_low_idx = max(self.low_channel_indices)
        if audio.shape[1] <= max_low_idx:
            raise ValueError(
                f"Audio has {audio.shape[1]} channels but needs channel index {max_low_idx}"
            )

        low_audio = audio[:, list(self.low_channel_indices)]
        high_audio = audio

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
        file_id = entry["file_id"]

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
            S_low_t, S_high_t = self._compute_csm_pair(audio=audio, sample_rate=sample_rate)
            if cache_path is not None:
                save_precomputed_csm_pair(cache_path, S_low=S_low_t, S_high=S_high_t)

        out = {
            "file_id": file_id,
            "scene_id": entry["scene_id"],
            "dataset": "audiblelight",
            "audio_path": str(wav_path),
            "ground_truth_path": (
                str(entry["ground_truth_path"]) if entry["ground_truth_path"] is not None else None
            ),
            "has_dcase_gt": entry["ground_truth_path"] is not None,
            "S_low": S_low_t,
            "S_high": S_high_t,
        }

        if self.cache_csm:
            self._cache[file_id] = out

        return out
