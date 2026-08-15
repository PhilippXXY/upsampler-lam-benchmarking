"""STAIRS26 audio and acoustic-map metadata loaders."""

import json
from pathlib import Path
from typing import Any

import soundfile
import torch
from torch.utils.data import Dataset

from .doa_event import DoaEvent

SEGMENTATION_COLUMNS = 3


def _peak_pixel(segmentation: list[Any]) -> tuple[float, float]:
    """Return the strongest pixel in a STAIRS26 segmentation mask.

    Parameters
    ----------
    segmentation : list[Any]
        Segmentation polygons containing ``[x, y, amplitude]`` rows.

    Returns
    -------
    tuple[float, float]
        Peak ``x`` and ``y`` coordinates.
    """
    pixels = [
        row for polygon in segmentation for row in polygon if len(row) >= SEGMENTATION_COLUMNS
    ]
    if not pixels:
        raise ValueError("STAIRS26 annotation has no segmentation pixels")
    peak = max(pixels, key=lambda row: float(row[2]))
    return float(peak[0]), float(peak[1])


class Stairs26GroundTruthLoader:
    """Convert STAIRS26 acoustic-map JSON labels to peak DoA events."""

    def __init__(self, metadata_path: Path, frame_width_ms: int = 100) -> None:
        """Initialise the STAIRS26 metadata loader.

        Parameters
        ----------
        metadata_path : Path
            Directory containing matching ``*_std.json`` label files.
        frame_width_ms : int, optional
            Metadata frame width in milliseconds.
        """
        self.metadata_path = Path(metadata_path)
        self.frame_width_s = frame_width_ms / 1000.0
        self._events: dict[str, DoaEvent] = {}

    def _find_json(self, file_id: str) -> Path:
        """Find a recording's metadata file.

        Parameters
        ----------
        file_id : str
            Audio recording stem.

        Returns
        -------
        Path
            Matching metadata path.
        """
        matches = list(self.metadata_path.rglob(f"{file_id}_std.json"))
        if not matches:
            expected = self.metadata_path / f"{file_id}_std.json"
            raise FileNotFoundError(f"STAIRS26 metadata not found: {expected}")
        if len(matches) > 1:
            raise ValueError(f"Multiple STAIRS26 metadata files found for {file_id}: {matches}")
        return matches[0]

    def load(self, file_id: str) -> DoaEvent:
        """Load peak DoA events for one recording.

        Parameters
        ----------
        file_id : str
            Audio recording stem.

        Returns
        -------
        DoaEvent
            Frame-aligned peak directions converted as in RVQ-LAM.
        """
        if file_id not in self._events:
            payload = json.loads(self._find_json(file_id).read_text(encoding="utf-8"))
            rows = []
            for annotation in payload.get("annotations", []):
                x, y = _peak_pixel(annotation.get("segmentation", []))
                rows.append(
                    (
                        int(annotation["metadata_frame_index"]),
                        int(annotation["category_id"]),
                        int(annotation["instance_id"]),
                        (360.0 - x) % 360.0 - 180.0,
                        90.0 - y,
                        float(annotation.get("distance", 0.0)),
                    )
                )
            data = (
                torch.tensor(rows, dtype=torch.float32)
                if rows
                else torch.empty((0, 6), dtype=torch.float32)
            )
            frame = data[:, 0].to(torch.int64)
            self._events[file_id] = DoaEvent(
                frame=frame,
                t_sec=frame.to(torch.float32) * self.frame_width_s,
                active_class_index=data[:, 1].to(torch.int64),
                source_number_index=data[:, 2].to(torch.int64),
                azimuth=data[:, 3],
                elevation=data[:, 4],
                distance_cm=data[:, 5],
            )
        return self._events[file_id]


class Stairs26AudioDataset(Dataset[dict[str, Any]]):
    """Load full STAIRS26 recordings from a flat or extracted archive tree."""

    def __init__(self, audio_path: Path) -> None:
        """Initialise the STAIRS26 audio dataset.

        Parameters
        ----------
        audio_path : Path
            Directory containing STAIRS26 WAV files or an extracted archive tree.

        Raises
        ------
        FileNotFoundError
            If the audio root does not exist or contains no matching WAV files.
        """
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_path}")

        self.wavs = sorted(self.audio_path.rglob("*.wav"))
        if not self.wavs:
            raise FileNotFoundError(f"No .wav files found in {self.audio_path}")

    def __len__(self) -> int:
        """Return the number of recordings.

        Returns
        -------
        int
            Number of discovered WAV files.
        """
        return len(self.wavs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one STAIRS26 recording.

        Parameters
        ----------
        index : int
            Recording index.

        Returns
        -------
        dict[str, Any]
            Time-major audio, sample rate, file ID, and source path.

        Raises
        ------
        ValueError
            If the recording is not an official 4- or 32-channel STAIRS26 file.
        """
        wav_path = self.wavs[index]
        audio, sample_rate = soundfile.read(wav_path, dtype="float32", always_2d=True)
        if audio.shape[1] not in {4, 32}:
            raise ValueError(f"Expected 4 or 32 channels, got {audio.shape[1]} for {wav_path}")
        return {
            "audio": torch.from_numpy(audio),
            "sample_rate": sample_rate,
            "file_id": wav_path.stem,
            "audio_path": str(wav_path),
        }
