"""Tests for variable-microphone inference channel selection."""

from __future__ import annotations

import numpy as np
import pytest

from utils.utils import prepare_audio_for_inference

COUNTS = (4, 8, 16, 24, 32)


def _config(counts: tuple[int, ...] = COUNTS) -> dict[str, object]:
    """Return a minimal variable-SRCNN inference configuration."""
    return {
        "model_name": "VariableSRCNNLAM",
        "sampling_rate": 24000,
        "max_audio_length_sec": 0,
        "variable_input_channel_counts": counts,
    }


@pytest.mark.parametrize("count", COUNTS)
def test_cli_indices_select_every_trained_count(count: int) -> None:
    """Select input channels from CLI identities while retaining the full reference."""
    audio = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    indices = tuple(range(31, 31 - count, -1))
    selected, reference, metadata = prepare_audio_for_inference(
        audio,
        sample_rate=24000,
        inference_config=_config(),
        input_channel_indices=indices,
    )
    assert np.array_equal(selected, audio[:, indices])
    assert np.array_equal(reference, audio)
    assert metadata["selected_channel_indices"] == indices


def test_variable_inference_requires_cli_indices_at_a_trained_count() -> None:
    """Reject missing CLI identities and counts absent from the checkpoint configuration."""
    audio = np.zeros((64, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="--input-channel-indices"):
        prepare_audio_for_inference(audio, 24000, _config())
    with pytest.raises(ValueError, match="trained count"):
        prepare_audio_for_inference(
            audio,
            24000,
            _config((4,)),
            input_channel_indices=tuple(range(8)),
        )
