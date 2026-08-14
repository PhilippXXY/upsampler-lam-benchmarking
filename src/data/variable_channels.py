"""Variable-microphone CSM completion datasets and samplers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence, Sized
from typing import Any, cast

import torch
from torch.utils.data import Dataset, Sampler

DEFAULT_VARIABLE_CHANNEL_COUNTS = (4, 8, 16, 24, 32)
CANONICAL_CHANNELS = 32
VariableChannelKey = tuple[int, int, tuple[int, ...]]


def validate_channel_counts(counts: Sequence[int]) -> tuple[int, ...]:
    """Validate supported canonical microphone counts.

    Parameters
    ----------
    counts : Sequence[int]
        Requested microphone counts.

    Returns
    -------
    tuple[int, ...]
        Sorted, unique microphone counts.
    """
    resolved = tuple(int(count) for count in counts)
    if (
        not resolved
        or tuple(sorted(set(resolved))) != resolved
        or any(count not in DEFAULT_VARIABLE_CHANNEL_COUNTS for count in resolved)
    ):
        raise ValueError("channel counts must be a sorted unique subset of (4, 8, 16, 24, 32)")
    return resolved


def validate_channel_indices(
    indices: Sequence[int] | torch.Tensor,
    counts: Sequence[int] = DEFAULT_VARIABLE_CHANNEL_COUNTS,
) -> torch.Tensor:
    """Validate one canonical microphone subset.

    Parameters
    ----------
    indices : Sequence[int] | torch.Tensor
        Zero-based canonical microphone indices.
    counts : Sequence[int], optional
        Permitted subset sizes.

    Returns
    -------
    torch.Tensor
        One-dimensional long tensor preserving the supplied order.
    """
    resolved = torch.as_tensor(indices, dtype=torch.long)
    valid_counts = validate_channel_counts(counts)
    if (
        resolved.ndim != 1
        or resolved.numel() not in valid_counts
        or resolved.unique().numel() != resolved.numel()
        or bool(((resolved < 0) | (resolved >= CANONICAL_CHANNELS)).any())
    ):
        raise ValueError("channel indices must be unique values in [0, 31] at a trained count")
    return resolved


def normalise_csm(csm: torch.Tensor) -> torch.Tensor:
    """Normalise Hermitian CSMs by their largest non-negative eigenvalue.

    Parameters
    ----------
    csm : torch.Tensor
        Complex CSM tensor with matching final dimensions.

    Returns
    -------
    torch.Tensor
        Hermitian positive-semidefinite CSM with unit largest eigenvalue.
    """
    hermitian = 0.5 * (csm + csm.transpose(-1, -2).conj())
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    eigenvalues = eigenvalues.clamp_min(0)
    scale = eigenvalues.amax(dim=-1, keepdim=True)
    normalised = torch.where(
        scale > 0,
        eigenvalues / scale.clamp_min(torch.finfo(scale.dtype).tiny),
        0,
    )
    return cast(
        torch.Tensor,
        (eigenvectors * normalised.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2).conj(),
    )


def sparse_csm_from_full(
    full_csm: torch.Tensor,
    observed_channel_indices: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Embed a re-normalised observed submatrix in a canonical sparse CSM.

    Parameters
    ----------
    full_csm : torch.Tensor
        Full complex CSM ending in shape ``(32, 32)``.
    observed_channel_indices : Sequence[int] | torch.Tensor
        Zero-based observed microphone indices.

    Returns
    -------
    torch.Tensor
        Sparse complex CSM matching ``full_csm``.
    """
    if full_csm.shape[-2:] != (CANONICAL_CHANNELS, CANONICAL_CHANNELS) or not full_csm.is_complex():
        raise ValueError("full_csm must be complex with final shape (32, 32)")
    indices = validate_channel_indices(observed_channel_indices).to(full_csm.device)
    selected = full_csm.index_select(-2, indices).index_select(-1, indices)
    sparse = torch.zeros_like(full_csm)
    sparse[..., indices[:, None], indices[None, :]] = normalise_csm(selected)
    return sparse


def embed_observed_csm(
    observed_csm: torch.Tensor,
    observed_channel_indices: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Scatter an already normalised observed CSM into the canonical grid.

    Parameters
    ----------
    observed_csm : torch.Tensor
        Complex observed CSM with matching final dimensions.
    observed_channel_indices : Sequence[int] | torch.Tensor
        Canonical indices corresponding to the observed CSM axes.

    Returns
    -------
    torch.Tensor
        Sparse canonical CSM ending in shape ``(32, 32)``.
    """
    indices = validate_channel_indices(observed_channel_indices).to(observed_csm.device)
    if not observed_csm.is_complex() or observed_csm.shape[-2:] != (indices.numel(),) * 2:
        raise ValueError("observed_csm dimensions must match observed_channel_indices")
    sparse = observed_csm.new_zeros(
        *observed_csm.shape[:-2], CANONICAL_CHANNELS, CANONICAL_CHANNELS
    )
    sparse[..., indices[:, None], indices[None, :]] = observed_csm
    return sparse


class VariableChannelCSMDataset(Dataset[dict[str, Any]]):
    """Derive variable sparse inputs from full-channel CSM samples."""

    def __init__(self, dataset: Dataset[dict[str, Any]], channel_counts: Sequence[int]) -> None:
        """Store the source dataset and permitted microphone counts.

        Parameters
        ----------
        dataset : Dataset[dict[str, Any]]
            Dataset returning a full ``S_high`` CSM.
        channel_counts : Sequence[int]
            Permitted input microphone counts.
        """
        self.dataset = dataset
        self.channel_counts = validate_channel_counts(channel_counts)

    def __len__(self) -> int:
        """Return the source dataset length."""
        if not isinstance(self.dataset, Sized):
            raise TypeError("variable-channel source dataset must define __len__")
        return len(self.dataset)

    def __getitem__(self, key: VariableChannelKey) -> dict[str, Any]:
        """Return one deterministically selected sparse CSM sample.

        Parameters
        ----------
        key : VariableChannelKey
            Source index, microphone count, and canonical indices.

        Returns
        -------
        dict[str, Any]
            Source metadata plus sparse/full CSMs and channel metadata.
        """
        index, count, raw_indices = key
        indices = validate_channel_indices(raw_indices, self.channel_counts)
        if indices.numel() != count:
            raise ValueError(f"expected {count} observed channels, got {indices.numel()}")
        sample = dict(self.dataset[index])
        full_csm = sample["S_high"]
        sample.update(
            S_low=sparse_csm_from_full(full_csm, indices),
            observed_channel_indices=indices,
            input_channel_count=torch.tensor(count),
        )
        return sample


class VariableChannelBatchSampler(Sampler[list[VariableChannelKey]]):
    """Yield one count-balanced variable-channel file pass per epoch."""

    def __init__(
        self,
        dataset_size: int,
        *,
        channel_counts: Sequence[int],
        shuffle: bool,
        seed: int,
        sample_weights: Sequence[float] | None = None,
    ) -> None:
        """Initialise deterministic count and subset sampling.

        Parameters
        ----------
        dataset_size : int
            Number of source files.
        channel_counts : Sequence[int]
            Counts assigned across each epoch.
        shuffle : bool
            Whether to redraw order and subsets each epoch.
        seed : int
            Sampling seed.
        sample_weights : Sequence[float] | None, optional
            Optional replacement-sampling weights for balanced datasets.
        """
        self.dataset_size = int(dataset_size)
        self.channel_counts = validate_channel_counts(channel_counts)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.sample_weights = None if sample_weights is None else torch.as_tensor(sample_weights)
        if self.sample_weights is not None and self.sample_weights.numel() != self.dataset_size:
            raise ValueError("sample_weights must match dataset_size")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch controlling training order and subsets."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Return the number of file-level batches."""
        return self.dataset_size

    def __iter__(self) -> Iterator[list[VariableChannelKey]]:
        """Yield deterministic one-file batches."""
        epoch = self.epoch if self.shuffle else 0
        generator = torch.Generator().manual_seed(self.seed + epoch)
        if self.sample_weights is not None:
            order = torch.multinomial(
                self.sample_weights, self.dataset_size, replacement=True, generator=generator
            ).tolist()
        elif self.shuffle:
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            order = list(range(self.dataset_size))
        for position, index in enumerate(order):
            count_offset = position if self.sample_weights is not None else index
            count = self.channel_counts[(count_offset + epoch) % len(self.channel_counts)]
            indices = tuple(
                torch.randperm(CANONICAL_CHANNELS, generator=generator)[:count]
                .sort()
                .values.tolist()
            )
            yield [(index, count, indices)]
