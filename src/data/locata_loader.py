"""
LOCATA Eigenmike dataset loaders for Direction of Arrival evaluation.

This module provides:
- LocataGroundTruthLoader: builds per-frame DoA ground truth from LOCATA
  timestamps, positions, and VAD files.
- LocataAudioDataset: full raw-audio record dataset for inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas
import soundfile
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .doa_event import DoaEvent

DEFAULT_LOCATA_TASKS = ("task1", "task2", "task3", "task4")
LOCATA_TO_DCASE_AZIMUTH_OFFSET_DEG = 90.0
ARRAY_WAV_FILENAME = "audio_array_eigenmike.wav"
ARRAY_POSITION_FILENAME = "position_array_eigenmike.txt"
ARRAY_TIMESTAMP_FILENAME = "audio_array_timestamps_eigenmike.txt"
REQUIRED_TIME_FILENAME = "required_time.txt"
ROTATION_COLUMNS = (
    "rotation_11",
    "rotation_12",
    "rotation_13",
    "rotation_21",
    "rotation_22",
    "rotation_23",
    "rotation_31",
    "rotation_32",
    "rotation_33",
)


@dataclass(frozen=True)
class _LocataEntry:
    """
    Internal descriptor for one LOCATA Eigenmike recording.

    Attributes
    ----------
    file_id : str
        Stable file identifier, e.g. "task1_recording1".
    task : str
        Task folder name, e.g. "task1".
    recording : str
        Recording folder name, e.g. "recording1".
    eigenmike_dir : Path
        Path to the `eigenmike` directory.
    wav_path : Path
        Path to `audio_array_eigenmike.wav`.
    """

    file_id: str
    task: str
    recording: str
    eigenmike_dir: Path
    wav_path: Path


def _read_table(path: Path, usecols: list[str] | None = None) -> Any:
    """
    Read a LOCATA tabular text file.

    Parameters
    ----------
    path : Path
        File path to read.
    usecols : list[str] | None, optional
        Optional subset of columns to load.

    Returns
    -------
    pandas.DataFrame
        Parsed dataframe.
    """
    with path.open("r", encoding="utf-8", errors="replace") as file_handle:
        first_line = file_handle.readline().strip()
    if first_line == "version https://git-lfs.github.com/spec/v1":
        raise RuntimeError(
            f"{path} is a Git LFS pointer file, not the real LOCATA table. "
            "Restore the dataset with `git lfs checkout data/locata/eval` "
            "or fetch it with `git lfs pull --include='data/locata/eval/**'`."
        )
    return pandas.read_csv(path, sep=r"\s+", engine="python", usecols=usecols)


def _time_columns_to_seconds(df: Any) -> NDArray[np.float64]:
    """
    Convert LOCATA time columns to scalar seconds.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least `hour`, `minute`, `second`.
        If `day` exists, it is included in the scalar timestamp.

    Returns
    -------
    np.ndarray
        Absolute scalar timestamps in seconds.
    """
    day = (
        np.asarray(df["day"].to_numpy(dtype=np.float64), dtype=np.float64)
        if "day" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )
    hour = np.asarray(df["hour"].to_numpy(dtype=np.float64), dtype=np.float64)
    minute = np.asarray(df["minute"].to_numpy(dtype=np.float64), dtype=np.float64)
    second = np.asarray(df["second"].to_numpy(dtype=np.float64), dtype=np.float64)
    return np.asarray(((day * 24.0 + hour) * 60.0 + minute) * 60.0 + second, dtype=np.float64)


def _nearest_indices(
    reference: NDArray[np.float64], query: NDArray[np.float64]
) -> NDArray[np.int64]:
    """
    Find nearest reference index for each query timestamp.

    Parameters
    ----------
    reference : np.ndarray
        Monotonic reference timestamps.
    query : np.ndarray
        Query timestamps.

    Returns
    -------
    np.ndarray
        Indices into `reference`, one per query.
    """
    insert_idx = np.searchsorted(reference, query, side="left")
    # Handle edge cases where query is before the first reference or after the last reference.
    left = np.clip(insert_idx - 1, 0, max(len(reference) - 1, 0))
    right = np.clip(insert_idx, 0, max(len(reference) - 1, 0))
    # Choose the closest of the two candidates.
    choose_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.asarray(np.where(choose_right, right, left), dtype=np.int64)


def _scan_eigenmike_recordings(path: Path, tasks: tuple[str, ...]) -> list[_LocataEntry]:
    """
    Discover LOCATA Eigenmike recordings for selected tasks.

    It only includes recordings that have the expected `audio_array_eigenmike.wav` file and their
    recording style matches the expected pattern as used by Roman et al..

    Parameters
    ----------
    path : Path
        Root path containing task directories (e.g. `.../locata/dev`).
    tasks : tuple[str, ...]
        Task folder names to include.

    Returns
    -------
    list[_LocataEntry]
        Sorted list of discovered recordings.
    """
    entries: list[_LocataEntry] = []
    for task in tasks:
        task_dir = path / task
        if not task_dir.exists() or not task_dir.is_dir():
            continue
        for recording_dir in sorted(task_dir.iterdir()):
            if not recording_dir.is_dir() or not recording_dir.name.startswith("recording"):
                continue
            eigenmike_dir = recording_dir / "eigenmike"
            wav_path = eigenmike_dir / ARRAY_WAV_FILENAME
            if not wav_path.exists():
                continue
            file_id = f"{task}_{recording_dir.name}"
            entries.append(
                _LocataEntry(
                    file_id=file_id,
                    task=task,
                    recording=recording_dir.name,
                    eigenmike_dir=eigenmike_dir,
                    wav_path=wav_path,
                )
            )
    return sorted(entries, key=lambda item: item.file_id)


def _empty_doa_event() -> DoaEvent:
    """
    Create an empty DoaEvent.

    Can be used as a fallback when no valid DoA information is available for a recording,
    e.g. due to missing files or invalid timestamps.

    Returns
    -------
    DoaEvent
        Empty event with zero-length tensors.
    """
    empty_i64 = torch.empty(0, dtype=torch.int64)
    empty_f32 = torch.empty(0, dtype=torch.float32)
    return DoaEvent(
        frame=empty_i64,
        t_sec=empty_f32,
        active_class_index=empty_i64.clone(),
        source_number_index=empty_i64.clone(),
        azimuth=empty_f32.clone(),
        elevation=empty_f32.clone(),
        distance_cm=empty_f32.clone(),
    )


class LocataGroundTruthLoader:
    """
    Ground truth loader for LOCATA Eigenmike data.

    Notes
    -----
    Ground truth is created per inference frame (default 100 ms) by:
    1. Mapping each frame center to nearest `required_time` index
    2. Computing incoming DoA vectors (source -> array) in global coordinates
    3. Rotating vectors into array-local coordinates via array rotation matrix
    4. Converting LOCATA local axes to the repo's DCASE-style azimuth/elevation/distance
    5. Keeping only frames with active source VAD at frame center
    """

    def __init__(
        self,
        ground_truth_path: Path = Path("data/locata/dev"),
        frame_width_ms: int = 100,
        tasks: tuple[str, ...] = DEFAULT_LOCATA_TASKS,
    ) -> None:
        """
        Initialise LOCATA ground truth loader.

        Parameters
        ----------
        ground_truth_path : Path, optional
            Root LOCATA path containing task directories.
            Defaults to `data/locata/dev`.
        frame_width_ms : int, optional
            Inference frame width in milliseconds.
            Defaults to 100.
        tasks : tuple[str, ...], optional
            Task folders to include. Defaults to task1-task4.
        """
        self.path = Path(ground_truth_path)
        self.frame_width_ms = frame_width_ms
        self.frame_width_resolution = frame_width_ms / 1000.0
        self.tasks = tasks

        if not self.path.exists():
            raise FileNotFoundError(f"Ground truth path not found: {self.path}")

        self._entries = _scan_eigenmike_recordings(self.path, self.tasks)
        if not self._entries:
            raise FileNotFoundError(
                f"No LOCATA Eigenmike recordings found under {self.path} for tasks {self.tasks}"
            )

        # Build a mapping from file_id to entry for quick lookup.
        self._entry_by_id = {entry.file_id: entry for entry in self._entries}
        # Cache for loaded DoaEvent objects to avoid redundant parsing on repeated access.
        self._cache: dict[str, DoaEvent] = {}

    def _find_entry(self, file_id: str) -> _LocataEntry:
        """
        Resolve file_id to recording entry.

        Parameters
        ----------
        file_id : str
            File identifier, e.g. `task1_recording1`.

        Returns
        -------
        _LocataEntry
            Recording descriptor.

        Raises
        ------
        FileNotFoundError
            If the file id is unknown.
        """
        entry = self._entry_by_id.get(file_id)
        if entry is None:
            raise FileNotFoundError(f"Unknown LOCATA file_id '{file_id}' under {self.path}")
        return entry

    def _load_vad_for_source(
        self,
        eigenmike_dir: Path,
        source_name: str,
        frame_centers: NDArray[np.int64],
        n_frames: int,
    ) -> NDArray[np.bool_]:
        """
        Load source VAD and map it to frame-center activity.

        Parameters
        ----------
        eigenmike_dir : Path
            Recording directory containing VAD files.
        source_name : str
            Source identifier suffix, e.g. `loudspeaker1` or `talker5`.
        frame_centers : np.ndarray
            Frame-center sample indices.
        n_frames : int
            Number of frames.

        Returns
        -------
        np.ndarray
            Boolean activity mask of shape (n_frames,).
        """
        primary = eigenmike_dir / f"VAD_eigenmike_{source_name}.txt"
        fallback = eigenmike_dir / f"VAD_source_{source_name}.txt"
        vad_path = primary if primary.exists() else fallback
        if not vad_path.exists():
            return np.ones(n_frames, dtype=bool)

        # The VAD files have one header line, then one line per sample with 0/1 activity.
        vad = np.loadtxt(vad_path, skiprows=1, dtype=np.int8, ndmin=1)
        active = np.zeros(n_frames, dtype=bool)
        # Only mark frames as active if their center sample has valid VAD information.
        valid_centers = frame_centers < len(vad)
        active[valid_centers] = vad[frame_centers[valid_centers]] > 0
        return active

    def _build_event(self, entry: _LocataEntry) -> DoaEvent:  # noqa: PLR0911, PLR0915
        """
        Build DoA event tensors for a single recording.

        We build the event by aligning frame centers to the nearest required time indices,
        then computing the DoA for each source at those frame centers based on the array and source
        positions and the array rotation.
        We also apply VAD and validity masks to only keep frames where the source is active and the
        required time information is valid.

        Parameters
        ----------
        entry : _LocataEntry
            Recording descriptor.

        Returns
        -------
        DoaEvent
            Parsed and frame-aligned DoA event data.
        """
        required_path = entry.eigenmike_dir / REQUIRED_TIME_FILENAME
        array_position_path = entry.eigenmike_dir / ARRAY_POSITION_FILENAME
        array_timestamp_path = entry.eigenmike_dir / ARRAY_TIMESTAMP_FILENAME

        # Load required time, array position, and array timestamp tables.
        required_df = _read_table(required_path)
        array_df = _read_table(
            array_position_path,
            usecols=["day", "hour", "minute", "second", "x", "y", "z", *ROTATION_COLUMNS],
        )
        array_ts_df = _read_table(
            array_timestamp_path,
            usecols=["day", "hour", "minute", "second"],
        )

        # Convert required time and array timestamps to seconds, and extract validity flags.
        required_seconds = _time_columns_to_seconds(required_df)
        required_valid = (
            np.asarray(required_df["valid_flag"].to_numpy(dtype=np.int64), dtype=np.int64) > 0
            if "valid_flag" in required_df.columns
            else np.ones(len(required_df), dtype=bool)
        )

        # Extract array positions and rotations as numpy arrays.
        array_xyz = np.asarray(
            array_df[["x", "y", "z"]].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
        array_rot = np.asarray(
            array_df[list(ROTATION_COLUMNS)].to_numpy(dtype=np.float64),
            dtype=np.float64,
        ).reshape(-1, 3, 3)
        array_seconds = _time_columns_to_seconds(array_ts_df)

        # Determine frame centers based on the array sample rate and desired frame width.
        sample_rate = int(soundfile.info(entry.wav_path).samplerate)
        frame_samples = int(round(sample_rate * self.frame_width_resolution))
        if frame_samples <= 0:
            return _empty_doa_event()

        # Map each frame center to the nearest array timestamp to find the corresponding array
        # position and rotation.
        n_samples = len(array_seconds)
        n_frames = n_samples // frame_samples
        if n_frames <= 0:
            return _empty_doa_event()

        # Compute frame center sample indices and corresponding timestamps.
        frame_numbers = np.arange(n_frames, dtype=np.int64)
        frame_centers = frame_numbers * frame_samples + frame_samples // 2
        frame_centers = np.clip(frame_centers, 0, n_samples - 1)
        frame_center_seconds = array_seconds[frame_centers]

        if len(required_seconds) == 0:
            return _empty_doa_event()

        # For each frame center timestamp, find the nearest required time index to determine which
        # array position/rotation to use and whether the frame has valid required time information.
        required_idx = _nearest_indices(required_seconds, frame_center_seconds)

        # Ensure that we only index into the arrays with valid indices and that all required arrays
        # have a common length to avoid out-of-bounds access.
        common_required_len = min(
            len(required_seconds), len(required_valid), len(array_xyz), len(array_rot)
        )
        if common_required_len <= 0:
            return _empty_doa_event()
        required_idx = np.clip(required_idx, 0, common_required_len - 1)
        frame_has_valid_time = required_valid[required_idx]

        source_paths = sorted(entry.eigenmike_dir.glob("position_source_*.txt"))
        if not source_paths:
            return _empty_doa_event()

        all_frames: list[NDArray[np.int64]] = []
        all_sources: list[NDArray[np.int64]] = []
        all_azimuth: list[NDArray[np.float32]] = []
        all_elevation: list[NDArray[np.float32]] = []
        all_distance_cm: list[NDArray[np.float32]] = []

        # Process each source position file to compute DoA and VAD for each source separately.
        for source_idx, source_path in enumerate(source_paths):
            source_name = source_path.stem.replace("position_source_", "")
            source_df = _read_table(
                source_path,
                usecols=["day", "hour", "minute", "second", "x", "y", "z"],
            )
            source_xyz = np.asarray(
                source_df[["x", "y", "z"]].to_numpy(dtype=np.float64),
                dtype=np.float64,
            )
            if len(source_xyz) == 0:
                continue

            source_common_len = min(common_required_len, len(source_xyz))
            source_idx_required = np.clip(required_idx, 0, source_common_len - 1)

            src = source_xyz[source_idx_required]
            arr = array_xyz[source_idx_required]
            rot = array_rot[source_idx_required]

            # Convert the global source location into the array-local frame.
            # LAM predicts the incoming-wave direction, i.e. source -> array.
            delta_global = arr - src
            delta_local = np.einsum("fji,fj->fi", rot, delta_global)

            # LOCATA's local horizontal reference differs from the DCASE convention used by
            # this repo's metrics.
            # Apply the fixed 90-degree offset after rotation into the array-local frame.
            azimuth = (
                np.degrees(np.arctan2(delta_local[:, 1], delta_local[:, 0]))
                + LOCATA_TO_DCASE_AZIMUTH_OFFSET_DEG
            )
            azimuth = (azimuth + 180.0) % 360.0 - 180.0
            xy_norm = np.linalg.norm(delta_local[:, :2], axis=1)
            elevation = np.degrees(np.arctan2(delta_local[:, 2], xy_norm))
            distance_cm = np.linalg.norm(delta_local, axis=1) * 100.0

            source_active = self._load_vad_for_source(
                entry.eigenmike_dir,
                source_name,
                frame_centers,
                n_frames,
            )

            # Only keep frames where the source is active, the required time is valid,
            # and the DoA values are finite.
            finite_mask = np.isfinite(azimuth) & np.isfinite(elevation) & np.isfinite(distance_cm)
            valid_mask = source_active & frame_has_valid_time & finite_mask
            if not np.any(valid_mask):
                continue

            frames = frame_numbers[valid_mask]
            all_frames.append(frames)
            all_sources.append(np.full(frames.shape, source_idx, dtype=np.int64))
            all_azimuth.append(azimuth[valid_mask].astype(np.float32))
            all_elevation.append(elevation[valid_mask].astype(np.float32))
            all_distance_cm.append(distance_cm[valid_mask].astype(np.float32))

        if not all_frames:
            return _empty_doa_event()

        frame_arr = np.concatenate(all_frames)
        source_arr = np.concatenate(all_sources)
        azimuth_arr = np.concatenate(all_azimuth)
        elevation_arr = np.concatenate(all_elevation)
        distance_arr = np.concatenate(all_distance_cm)
        class_arr = np.zeros_like(frame_arr, dtype=np.int64)
        t_sec_arr = frame_arr.astype(np.float32) * self.frame_width_resolution

        order = np.lexsort((source_arr, frame_arr))

        return DoaEvent(
            frame=torch.from_numpy(frame_arr[order]),
            t_sec=torch.from_numpy(t_sec_arr[order]),
            active_class_index=torch.from_numpy(class_arr[order]),
            source_number_index=torch.from_numpy(source_arr[order]),
            azimuth=torch.from_numpy(azimuth_arr[order]),
            elevation=torch.from_numpy(elevation_arr[order]),
            distance_cm=torch.from_numpy(distance_arr[order]),
        )

    def load(self, file_id: str) -> DoaEvent:
        """
        Load frame-level DoA ground truth for a recording.

        Parameters
        ----------
        file_id : str
            Recording identifier, e.g. `task1_recording1`.

        Returns
        -------
        DoaEvent
            Ground truth DoA events.
        """
        if file_id in self._cache:
            return self._cache[file_id]

        entry = self._find_entry(file_id)
        event = self._build_event(entry)
        self._cache[file_id] = event
        return event


class LocataAudioDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for LOCATA Eigenmike audio and optional ground truth."""

    def __init__(
        self,
        path: Path,
        load_ground_truth: bool = False,
        frame_width_ms: int = 100,
        tasks: tuple[str, ...] = DEFAULT_LOCATA_TASKS,
    ) -> None:
        """
        Initialise LOCATA audio dataset.

        Parameters
        ----------
        path : Path
            Root LOCATA path containing task directories.
        load_ground_truth : bool, optional
            If True, also load per-file ground truth in `__getitem__`.
            Defaults to False.
        frame_width_ms : int, optional
            Frame width used by ground truth loader.
            Defaults to 100.
        tasks : tuple[str, ...], optional
            Task folders to include. Defaults to task1-task4.
        """
        self.path = Path(path)
        self.load_ground_truth = load_ground_truth
        self.frame_width_ms = frame_width_ms
        self.tasks = tasks

        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self.entries = _scan_eigenmike_recordings(self.path, self.tasks)
        if not self.entries:
            raise FileNotFoundError(
                f"No LOCATA Eigenmike recordings found under {self.path} for tasks {self.tasks}"
            )

        # Compatibility attributes used by current inference script.
        self.relevant_dir = [entry.eigenmike_dir for entry in self.entries]
        self.wavs = [entry.wav_path for entry in self.entries]

        if self.load_ground_truth:
            self.ground_truth: LocataGroundTruthLoader | None = LocataGroundTruthLoader(
                ground_truth_path=self.path,
                frame_width_ms=self.frame_width_ms,
                tasks=self.tasks,
            )
        else:
            self.ground_truth = None

    def __len__(self) -> int:
        """
        Return number of recordings.

        Returns
        -------
        int
            Number of LOCATA recordings in this dataset.
        """
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load one LOCATA recording.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        dict[str, Any]
            Sample dictionary containing:
            - audio: torch.Tensor, shape (T, 32)
            - sample_rate: int
            - file_id: str
            - audio_path: str
            - ground_truth: DoaEvent (optional)
        """
        entry = self.entries[index]
        audio_np, sample_rate = soundfile.read(entry.wav_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio_np)

        expected_channels = 32
        if audio.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {audio.shape[1]} for {entry.wav_path}"
            )

        out: dict[str, Any] = {
            "audio": audio,
            "sample_rate": sample_rate,
            "file_id": entry.file_id,
            "audio_path": str(entry.wav_path),
        }

        if self.ground_truth is not None:
            out["ground_truth"] = self.ground_truth.load(entry.file_id)
        return out
