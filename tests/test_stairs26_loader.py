"""Tests for the STAIRS26 inference loader."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

from data.stairs26_loader import Stairs26AudioDataset

SAMPLE_RATE = 24_000


def _write_wav(path: Path, channels: int) -> None:
    """Write a short multi-channel test recording.

    Parameters
    ----------
    path : Path
        Output WAV path.
    channels : int
        Number of audio channels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.zeros((24, channels), dtype=np.float32), SAMPLE_RATE)


def test_discovers_nested_wavs(tmp_path: Path) -> None:
    """Discover archive trees recursively."""
    _write_wav(tmp_path / "audio" / "first" / "fold1_room1_mix001.wav", 32)
    _write_wav(tmp_path / "audio" / "second" / "fold2_room1_mix002.wav", 32)

    expected_files = 2
    dataset = Stairs26AudioDataset(tmp_path)
    assert len(dataset) == expected_files
    assert dataset[0]["file_id"] == "fold1_room1_mix001"
    assert dataset[0]["audio"].shape == (24, 32)
    assert dataset[0]["sample_rate"] == SAMPLE_RATE


def test_accepts_official_four_channel_evaluation_audio(tmp_path: Path) -> None:
    """Accept the official four-channel evaluation recording format."""
    _write_wav(tmp_path / "mix001.wav", 4)
    expected_channels = 4
    assert Stairs26AudioDataset(tmp_path)[0]["audio"].shape[1] == expected_channels


def test_rejects_invalid_channel_count(tmp_path: Path) -> None:
    """Reject non-STAIRS channel counts."""
    _write_wav(tmp_path / "invalid.wav", 8)
    with pytest.raises(ValueError, match="Expected 4 or 32 channels"):
        Stairs26AudioDataset(tmp_path)[0]
