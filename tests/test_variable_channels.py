"""Tests for variable-microphone CSM construction and sampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from data.variable_channels import (
    VariableChannelBatchSampler,
    VariableChannelCSMDataset,
    embed_observed_csm,
    sparse_csm_from_full,
    validate_channel_counts,
    validate_channel_indices,
)
from lam_min.dataset.gen_dataset.gen_dataset import get_visibility_matrix

COUNTS = (4, 8, 16, 24, 32)


class TinyCSMDataset(Dataset[dict[str, Any]]):
    """Return deterministic full-channel positive-semidefinite CSMs."""

    def __len__(self) -> int:
        """Return the sample count."""
        return 12

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one full-channel CSM."""
        generator = torch.Generator().manual_seed(index)
        vectors = torch.randn(2, 1, 32, 3, generator=generator, dtype=torch.complex64)
        return {"S_high": vectors @ vectors.transpose(-1, -2).conj(), "file_id": str(index)}


def _keys(sampler: VariableChannelBatchSampler) -> list[tuple[int, int, tuple[int, ...]]]:
    """Flatten one-file sampler batches."""
    return [batch[0] for batch in sampler]


def test_sampler_balances_counts_and_rotates_each_file() -> None:
    """Balance counts per epoch and rotate each file through all counts."""
    sampler = VariableChannelBatchSampler(12, channel_counts=COUNTS, shuffle=True, seed=7)
    first = _keys(sampler)
    exposures = [sum(key[1] == count for key in first) for count in COUNTS]
    assert max(exposures) - min(exposures) == 1

    observed = []
    for epoch in range(len(COUNTS)):
        sampler.set_epoch(epoch)
        observed.append(next(key[1] for key in _keys(sampler) if key[0] == 0))
    assert tuple(observed) == COUNTS


def test_training_subsets_change_but_validation_is_fixed() -> None:
    """Redraw training subsets while pinning validation to epoch zero."""
    training = VariableChannelBatchSampler(12, channel_counts=COUNTS, shuffle=True, seed=11)
    training.set_epoch(0)
    first = next(key[2] for key in _keys(training) if key[:2] == (0, 4))
    training.set_epoch(5)
    second = next(key[2] for key in _keys(training) if key[:2] == (0, 4))
    validation = VariableChannelBatchSampler(12, channel_counts=COUNTS, shuffle=False, seed=11)
    expected = _keys(validation)
    validation.set_epoch(9)
    assert first != second
    assert _keys(validation) == expected


def test_sampler_seed_is_reproducible() -> None:
    """Reproduce count assignments, file order, and subsets from the same seed."""
    first = VariableChannelBatchSampler(12, channel_counts=COUNTS, shuffle=True, seed=19)
    second = VariableChannelBatchSampler(12, channel_counts=COUNTS, shuffle=True, seed=19)
    first.set_epoch(3)
    second.set_epoch(3)
    assert _keys(first) == _keys(second)


def test_dataset_embeds_only_selected_pairs() -> None:
    """Expose sparse/full CSMs and canonical subset metadata."""
    indices = (1, 5, 12, 30)
    sample = VariableChannelCSMDataset(TinyCSMDataset(), COUNTS)[(0, 4, indices)]
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[list(indices)] = True
    mask = mask & mask.T
    assert sample["S_low"].shape == sample["S_high"].shape
    assert torch.count_nonzero(sample["S_low"][..., ~mask]) == 0
    assert torch.equal(sample["observed_channel_indices"], torch.tensor(indices))


def test_embedding_preserves_arbitrary_index_order() -> None:
    """Map observed CSM axes to their supplied canonical microphone identities."""
    indices = (30, 1, 12, 5)
    observed = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).to(torch.complex64)
    sparse = embed_observed_csm(observed, indices)
    index_tensor = torch.tensor(indices)
    assert torch.equal(sparse[..., index_tensor[:, None], index_tensor[None, :]], observed)


def test_channel_configuration_validation() -> None:
    """Reject unsupported counts and invalid canonical microphone identities."""
    with pytest.raises(ValueError, match="sorted unique subset"):
        validate_channel_counts((4, 5, 32))
    with pytest.raises(ValueError, match="unique values"):
        validate_channel_indices((0, 1, 2, 32))
    with pytest.raises(ValueError, match="trained count"):
        validate_channel_indices(tuple(range(8)), counts=(4,))


def test_cached_full_csm_matches_direct_subset_preprocessing() -> None:
    """Recover the same selected CSM from a normalised full-channel CSM."""
    generator = np.random.default_rng(4)
    audio = generator.normal(size=(4800, 32)).astype(np.float32)
    indices = (1, 5, 12, 30)
    full, _ = get_visibility_matrix(audio, fs=24000, nbands=1)
    observed, _ = get_visibility_matrix(audio[:, indices], fs=24000, nbands=1)
    full_tensor = torch.from_numpy(full)
    sparse = sparse_csm_from_full(full_tensor, indices)
    embedded = embed_observed_csm(torch.from_numpy(observed), indices)
    assert torch.allclose(sparse, embedded, rtol=2e-5, atol=2e-6)
