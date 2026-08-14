"""Tests for masked variable-microphone SRCNN completion."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from upsampler.srcnn import SRCNNUpsampler, VariableSRCNNUpsampler

COUNTS = (4, 8, 16, 24, 32)
CANONICAL_CHANNELS = 32
LEGACY_INPUT_PLANES = 2
VARIABLE_INPUT_PLANES = 3
EXPECTED_RECEPTIVE_FIELD = 65


def _hermitian(batch: int = 1, bands: int = 1) -> torch.Tensor:
    """Create random Hermitian CSM-like tensors."""
    values = torch.randn(batch, bands, 32, 32, dtype=torch.complex64)
    return 0.5 * (values + values.transpose(-1, -2).conj())


def test_variable_srcnn_has_full_grid_receptive_field() -> None:
    """Use the requested three-plane 9–5–5 architecture with a 65-cell receptive field."""
    model = VariableSRCNNUpsampler(feature_channels=4, mapping_channels=3)
    convolutions = [layer for layer in model.srcnn if isinstance(layer, torch.nn.Conv2d)]
    assert [layer.kernel_size for layer in convolutions] == [(9, 9), (5, 5), (5, 5)]
    receptive_field = 1 + sum(
        (layer.kernel_size[0] - 1) * layer.dilation[0] for layer in convolutions
    )
    assert convolutions[0].in_channels == VARIABLE_INPUT_PLANES
    assert receptive_field == EXPECTED_RECEPTIVE_FIELD


@pytest.mark.parametrize("count", COUNTS)
def test_variable_srcnn_preserves_observed_pairs(count: int) -> None:
    """Accept each trained count and preserve measured entries exactly."""
    model = VariableSRCNNUpsampler(feature_channels=4, mapping_channels=3)
    indices = (
        torch.randperm(32, generator=torch.Generator().manual_seed(count))[:count].sort().values
    )
    target = _hermitian()
    sparse = torch.zeros_like(target)
    sparse[..., indices[:, None], indices[None, :]] = target[
        ..., indices[:, None], indices[None, :]
    ]
    output = model(sparse, indices)
    pair_mask = torch.zeros(32, 32, dtype=torch.bool)
    pair_mask[indices[:, None], indices[None, :]] = True
    assert output.shape == target.shape
    assert torch.equal(output[..., pair_mask], sparse[..., pair_mask])
    assert torch.allclose(output, output.transpose(-1, -2).conj())
    if count == CANONICAL_CHANNELS:
        assert torch.equal(output, sparse)


@pytest.mark.parametrize("count", COUNTS[:-1])
def test_missing_entry_loss_backpropagates(count: int) -> None:
    """Optimise missing entries without changing observed-pair loss."""
    model = VariableSRCNNUpsampler(feature_channels=4, mapping_channels=3, loss_name="mse")
    indices = (
        torch.randperm(32, generator=torch.Generator().manual_seed(count))[:count].sort().values
    )
    target = _hermitian()
    sparse = torch.zeros_like(target)
    sparse[..., indices[:, None], indices[None, :]] = target[
        ..., indices[:, None], indices[None, :]
    ]
    prediction = model(sparse, indices)
    loss, _ = model.compute_loss(prediction, target, indices)
    loss.backward()
    assert loss > 0
    assert any(
        parameter.grad is not None and torch.any(parameter.grad) for parameter in model.parameters()
    )


def test_missing_entry_loss_is_normalised_and_32ch_loss_is_zero() -> None:
    """Average only missing complex entries and make full input an exact zero-loss case."""
    model = VariableSRCNNUpsampler(feature_channels=2, mapping_channels=2, loss_name="l1")
    target = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    prediction = torch.full_like(target, 2 + 4j)
    indices = torch.tensor([0, 7, 18, 31])
    prediction[..., indices[:, None], indices[None, :]] = 100 + 100j
    loss, _ = model.compute_loss(prediction, target, indices)
    assert loss == pytest.approx(3.0)
    full_loss, _ = model.compute_loss(prediction, target, torch.arange(32))
    assert full_loss == 0


def test_invalid_runtime_indices_are_rejected() -> None:
    """Reject duplicate, out-of-range, and unsupported microphone subsets."""
    model = VariableSRCNNUpsampler(feature_channels=2, mapping_channels=2)
    sparse = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    with pytest.raises(ValueError, match="trained input count"):
        model(sparse, torch.tensor([0, 0, 1, 2]))
    with pytest.raises(ValueError, match="trained input count"):
        model(sparse, torch.arange(5))
    with pytest.raises(ValueError, match="trained input count"):
        model(sparse, torch.tensor([0, 1, 2, 32]))


def test_variable_srcnn_checkpoint_reload(tmp_path: Path) -> None:
    """Reload a variable SRCNN checkpoint without involving the legacy architecture."""
    model = VariableSRCNNUpsampler(feature_channels=4, mapping_channels=3)
    checkpoint = tmp_path / "variable_srcnn.pth"
    torch.save(model.state_dict(), checkpoint)
    reloaded = VariableSRCNNUpsampler(feature_channels=4, mapping_channels=3)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    sparse = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    indices = torch.tensor([0, 7, 18, 31])
    assert torch.equal(model(sparse, indices), reloaded(sparse, indices))


def test_legacy_srcnn_architecture_and_reload_remain_unchanged() -> None:
    """Keep the fixed SRCNN two-plane architecture checkpoint-compatible."""
    legacy = SRCNNUpsampler(feature_channels=4, mapping_channels=3)
    reloaded = SRCNNUpsampler(feature_channels=4, mapping_channels=3)
    reloaded.load_state_dict(legacy.state_dict())
    low = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    assert legacy.srcnn[0].in_channels == LEGACY_INPUT_PLANES
    assert (
        VariableSRCNNUpsampler(feature_channels=4, mapping_channels=3).srcnn[0].in_channels
        == VARIABLE_INPUT_PLANES
    )
    assert torch.equal(legacy(low), reloaded(low))
