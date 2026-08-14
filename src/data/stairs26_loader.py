"""STAIRS26 multi-channel audio loader for inference."""

from pathlib import Path
from typing import Any

import soundfile
import torch
from torch.utils.data import Dataset


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
