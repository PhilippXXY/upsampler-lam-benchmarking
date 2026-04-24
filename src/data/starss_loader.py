"""
STARSS23 dataset loaders for Direction of Arrival estimation.

This module provides PyTorch dataset classes for loading and preprocessing
audio data and ground truth annotations from the STARSS23 (Spatial Audio
Recording from a Spherical microphone array - 2023) challenge dataset.

The module includes:
    - DoaEvent: Dataclass for Direction of Arrival event metadata
    - StarssGroundTruthLoader: CSV parser for DoA ground truth annotations
    - StarssAudioDataset: full raw-audio record dataset for inference

References
----------
.. [1] STARSS23 Challenge: https://arxiv.org/pdf/2306.09126v1
"""

from pathlib import Path
from typing import Any, Dict

import pandas
import soundfile
import torch
from torch.utils.data import Dataset

from .doa_event import DoaEvent


class StarssGroundTruthLoader:
    """
    Loader for STARSS dataset ground truth Direction of Arrival (DoA) metadata.

    This class provides functionality to load and parse STARSS23 CSV metadata files
    containing Direction of Arrival event information.
    It handles file discovery, validation, and conversion of raw CSV data
    into structured DoaEvent objects.

    Attributes
    ----------
    metadata_dir_path: Path
        Directory path for the STARSS23 dataset metadata CSVs.
    frame_width_ms: int
        Frame width in milliseconds for converting frame numbers to time.
    frame_width_resolution: float
        Frame width in seconds, derived from frame_width_ms.
    """

    def __init__(self, ground_truth_path: Path, frame_width_ms: int = 100) -> None:
        """
        Initialise the StarssGroundTruthLoader.

        Parameters
        ----------
        ground_truth_path: Path
            The directory path containing the STARSS metadata files folder.
        frame_width_ms : int, optional
            Frame width in milliseconds. Defaults to `100`.
        """
        self.metadata_dir_path = Path(ground_truth_path)
        self.frame_width_ms = frame_width_ms
        self.frame_width_resolution = frame_width_ms / 1000.0  # in seconds

    def _find_csv(self, file_id: str) -> Path:
        """
        Locate a CSV file in the metadata directory by file ID.

        Parameters
        ----------
        file_id: str
            The identifier of the file (without `.csv` extension).

        Returns
        -------
        : Path
            The full path to the CSV file.

        Raises
        ------
        FileNotFoundError
            If the metadata directory does not exist or
                              the CSV file with the specified file_id is not found.
        """
        if not self.metadata_dir_path.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir_path}")

        file = self.metadata_dir_path.joinpath(f"{file_id}.csv")
        if file.exists():
            return file
        else:
            raise FileNotFoundError(f"File with name {file_id}.csv does not exist")

    def load(self, file_id: str) -> DoaEvent:
        """
        Load a STARSS dataset file and parse it into a DoaEvent object.

        Parameters
        ----------
        file_id: str
            The identifier of the file to load.

        Returns
        -------
        DoaEvent
            An object containing parsed Direction of Arrival (DoA) event data

        Raises
        ------
        ValueError
            If the CSV file does not contain exactly 6 columns.
        """
        csv_path = self._find_csv(file_id)
        df = pandas.read_csv(csv_path, header=None)

        expected_columns = 6
        if df.shape[1] != expected_columns:
            raise ValueError(
                f"Expected {expected_columns} columns, " f"got {df.shape[1]} in {csv_path}"
            )

        # The column headers are defined in https://arxiv.org/pdf/2306.09126v1
        df.columns = [
            "frame_number",
            "active_class_index",
            "source_number_index",
            "azimuth",
            "elevation",
            "distance",
        ]

        # A frame is a 100 msec
        frame = torch.tensor(df["frame_number"].to_numpy(), dtype=torch.int64)
        t_sec = frame.to(torch.float32) * self.frame_width_resolution
        active_class_index = torch.tensor(df["active_class_index"].to_numpy(), dtype=torch.int64)
        source_number_index = torch.tensor(df["source_number_index"].to_numpy(), dtype=torch.int64)
        azimuth = torch.tensor(df["azimuth"].to_numpy(), dtype=torch.int64)
        elevation = torch.tensor(df["elevation"].to_numpy(), dtype=torch.int64)
        distance_cm = torch.tensor(df["distance"].to_numpy(), dtype=torch.int64)

        doa = DoaEvent(
            frame=frame,
            t_sec=t_sec,
            active_class_index=active_class_index,
            source_number_index=source_number_index,
            azimuth=azimuth,
            elevation=elevation,
            distance_cm=distance_cm,
        )

        return doa


class StarssAudioDataset(Dataset[dict[str, Any]]):
    """
    PyTorch dataset for full STARSS23 audio records.

    This dataset loads multi-channel audio files and optional ground-truth annotations from the
    STARSS23 dataset. Each item is returned as a dictionary containing audio, sample rate,
    file identifier, source path, and optional DoA metadata.

    Attributes
    ----------
    root_path: Path
        Root directory path for the STARSS23 dataset.
    sub_path: Path
        Subdirectory path within root_path.
    split : Path
        Dataset split identifier (e.g., `dev-test-sony`).
    audio_dir : Path
        Constructed full path to the audio directory.
    wavs : list[Path]
        Sorted list of all WAV files found in the audio directory.
    ground_truth : StarssGroundTruthLoader or None
        Ground truth loader instance if load_ground_truth is True, otherwise None.
    """

    def __init__(
        self,
        audio_path: Path,
        ground_truth_path: Path,
        load_ground_truth: bool = True,
        frame_width_ms: int = 100,
    ) -> None:
        """
        Initialise the STARSS audio dataset.

        Parameters
        ----------
        audio_path : Path
            Path to the directory containing the STARSS23 audio files.
        ground_truth_path : Path
            Path to the directory containing ground truth annotations.
        load_ground_truth : bool, optional
            Whether to load ground truth annotations. Default is True.
        frame_width_ms : int, optional
            Frame width in milliseconds. Defaults to `100`.

        Raises
        ------
        FileNotFoundError
            If the specified audio directory does not exist or no WAV files are found.
        """
        self.audio_path = Path(audio_path)
        self.ground_truth_path = Path(ground_truth_path)

        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio dir not found: {self.audio_path}")
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth dir not found: {self.ground_truth_path}")

        # Store all audio files from the directory
        self.wavs = sorted(self.audio_path.glob("*.wav"))
        if not self.wavs:
            raise FileNotFoundError(f"No .wav files found in {self.audio_path}")

        self.frame_width_ms = frame_width_ms
        if load_ground_truth:
            self.ground_truth: StarssGroundTruthLoader | None = StarssGroundTruthLoader(
                self.ground_truth_path, self.frame_width_ms
            )
        else:
            self.ground_truth = None

    def __len__(self) -> int:
        """
        Return the number of audio samples in the dataset.

        Returns
        -------
        length: int
            The total number of `.wav` files in the dataset.
        """
        return len(self.wavs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Load audio data and metadata for a single sample from the dataset.

        Parameters
        ----------
        index: int
            The index of the sample to load.

        Returns
        -------
        A dictionary containing:
            - "audio": Tensor of shape (T, C) where T is the number of frames
            and C is the number of channels (4 or 32).
            - "sample_rate": The sample rate of the audio file.
            - "file_id": The stem (filename without extension) of the audio file.
            - "audio_path": The full path to the audio file as a string.
            - "ground_truth": (optional) Ground truth annotations if available in the dataset.

        Raises
        ------
        ValueError
            If the audio does not have 4 or 32 channels.
        """
        wav_path = self.wavs[index]
        # audio is an array with frames x channels (T x C)
        audio, sample_rate = soundfile.read(wav_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio)

        valid_channel_counts = {4, 32}
        if audio.shape[1] not in valid_channel_counts:
            raise ValueError(f"Expected 4 or 32 channels, got {audio.shape[1]} for {wav_path}")

        file_id = wav_path.stem
        out = {
            "audio": audio,
            "sample_rate": sample_rate,
            "file_id": file_id,
            "audio_path": str(wav_path),
        }

        if self.ground_truth is not None:
            out["ground_truth"] = self.ground_truth.load(file_id=file_id)

        return out
