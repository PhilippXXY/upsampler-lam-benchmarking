"""Masked variable-microphone SRCNN CSM completion."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from data.variable_channels import (
    CANONICAL_CHANNELS,
    DEFAULT_VARIABLE_CHANNEL_COUNTS,
    validate_channel_counts,
)
from upsampler.base import StepOptimiser, TrainableUpsampler


class VariableSRCNNUpsampler(TrainableUpsampler):
    """Complete canonical CSMs from masked variable-microphone observations."""

    INPUT_NDIM = 4
    INDEX_NDIM = 2

    def __init__(
        self,
        out_channels: int = CANONICAL_CHANNELS,
        feature_channels: int = 64,
        mapping_channels: int = 32,
        loss_name: str = "l1",
        variable_input_channel_counts: Sequence[int] = DEFAULT_VARIABLE_CHANNEL_COUNTS,
    ) -> None:
        """Initialise the full-grid masked SRCNN.

        Parameters
        ----------
        out_channels : int, optional
            Canonical output channel count, which must be 32.
        feature_channels : int, optional
            Feature extraction width.
        mapping_channels : int, optional
            Mapping layer width.
        loss_name : str, optional
            Missing-entry loss, either ``l1`` or ``mse``.
        variable_input_channel_counts : Sequence[int], optional
            Supported observed microphone counts.
        """
        super().__init__()
        if int(out_channels) != CANONICAL_CHANNELS:
            raise ValueError("VariableSRCNNUpsampler requires out_channels=32")
        self.out_channels = CANONICAL_CHANNELS
        self.variable_input_channel_counts = validate_channel_counts(variable_input_channel_counts)
        self.loss_name = loss_name.strip().lower()
        if self.loss_name not in {"l1", "mse"}:
            raise ValueError("loss_name must be one of: l1, mse")
        self.srcnn = nn.Sequential(
            nn.Conv2d(3, feature_channels, 9, padding=4, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, mapping_channels, 5, padding=8, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mapping_channels, 2, 5, padding=20, dilation=10),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise convolutional weights."""
        convolutions = [layer for layer in self.srcnn if isinstance(layer, nn.Conv2d)]
        for layer in convolutions[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(convolutions[-1].weight)
        if convolutions[-1].bias is not None:
            nn.init.zeros_(convolutions[-1].bias)

    def _pair_mask(
        self, indices: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Build batched observed microphone-pair masks."""
        indices = torch.as_tensor(indices, device=device, dtype=torch.long)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        valid = (
            indices.ndim == self.INDEX_NDIM
            and indices.shape[0] == batch_size
            and indices.shape[1] in self.variable_input_channel_counts
            and bool(((indices >= 0) & (indices < self.out_channels)).all())
            and all(row.unique().numel() == row.numel() for row in indices)
        )
        if not valid:
            raise ValueError("observed_channel_indices do not match a trained input count")
        microphone_mask = torch.zeros(
            batch_size, self.out_channels, dtype=torch.bool, device=device
        )
        microphone_mask.scatter_(1, indices, True)
        return microphone_mask.unsqueeze(-1) & microphone_mask.unsqueeze(-2)

    def _forward_no_metrics(
        self, sparse_csm: torch.Tensor, observed_channel_indices: torch.Tensor
    ) -> torch.Tensor:
        """Run masked CSM completion without runtime measurement."""
        batch_size, bands = sparse_csm.shape[:2]
        pair_mask = self._pair_mask(observed_channel_indices, batch_size, sparse_csm.device)
        mask_plane = pair_mask[:, None].expand(-1, bands, -1, -1)
        model_input = torch.stack(
            (sparse_csm.real, sparse_csm.imag, mask_plane.to(sparse_csm.real.dtype)), dim=2
        ).reshape(batch_size * bands, 3, self.out_channels, self.out_channels)
        raw = self.srcnn(model_input.float()).to(sparse_csm.real.dtype)
        candidate = torch.complex(raw[:, 0], raw[:, 1]).reshape_as(sparse_csm)
        candidate = 0.5 * (candidate + candidate.transpose(-1, -2).conj())
        return torch.where(mask_plane, sparse_csm, candidate)

    def forward(
        self,
        sparse_csm: torch.Tensor,
        observed_channel_indices: torch.Tensor,
        collect_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Complete a sparse canonical CSM.

        Parameters
        ----------
        sparse_csm : torch.Tensor
            Complex tensor shaped ``(batch, bands, 32, 32)``.
        observed_channel_indices : torch.Tensor
            Canonical indices shaped ``(microphones,)`` or ``(batch, microphones)``.
        collect_metrics : bool, optional
            Whether to include runtime statistics.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, dict[str, float]]
            Completed CSM, optionally with runtime statistics.
        """
        if (
            sparse_csm.ndim != self.INPUT_NDIM
            or sparse_csm.shape[-2:] != (32, 32)
            or not sparse_csm.is_complex()
        ):
            raise ValueError("sparse_csm must be complex with shape (batch, bands, 32, 32)")
        if not collect_metrics:
            return self._forward_no_metrics(sparse_csm, observed_channel_indices)
        use_cuda = sparse_csm.device.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(sparse_csm.device)
            torch.cuda.synchronize(sparse_csm.device)
        else:
            tracemalloc.start()
        start = time.perf_counter()
        output = self._forward_no_metrics(sparse_csm, observed_channel_indices)
        if use_cuda:
            torch.cuda.synchronize(sparse_csm.device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if use_cuda:
            memory_mb = torch.cuda.max_memory_allocated(sparse_csm.device) / 1024**2
        else:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024**2
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            self._forward_no_metrics(sparse_csm, observed_channel_indices)
        total_flops = float(counter.get_total_flops())
        frames = int(sparse_csm.shape[0])
        return output, {
            "upsampler_time_ms": elapsed_ms,
            "upsampler_flops": total_flops,
            "upsampler_flops_per_frame": total_flops / max(frames, 1),
            "upsampler_memory_mb": float(memory_mb),
            "num_frames": float(frames),
        }

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        observed_channel_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute mean reconstruction loss over missing microphone pairs.

        Parameters
        ----------
        pred : torch.Tensor
            Completed CSM.
        target : torch.Tensor
            Full-resolution target CSM.
        observed_channel_indices : torch.Tensor
            Canonical observed microphone indices.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and normalised statistics.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
        indices = torch.as_tensor(observed_channel_indices, device=pred.device, dtype=torch.long)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(pred.shape[0], -1)
        pair_mask = self._pair_mask(indices, pred.shape[0], pred.device)
        missing = (~pair_mask).unsqueeze(1).expand_as(pred.real)
        real_error = pred.real - target.real
        imag_error = pred.imag - target.imag
        entry_loss = (
            0.5 * (real_error.abs() + imag_error.abs())
            if self.loss_name == "l1"
            else 0.5 * (real_error.square() + imag_error.square())
        )
        loss = (entry_loss * missing).sum() / missing.sum().clamp_min(1)
        value = float(loss.detach().cpu())
        return loss, self.normalise_step_stats(
            {"loss_total": value, f"loss_{self.loss_name}": value}
        )

    def training_step(
        self,
        S_low: torch.Tensor,
        S_high: torch.Tensor,
        optimiser: StepOptimiser,
        grad_clip_norm: float = 0.0,
        observed_channel_indices: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one variable-channel optimisation step.

        Parameters
        ----------
        S_low : torch.Tensor
            Sparse input CSM.
        S_high : torch.Tensor
            Full target CSM.
        optimiser : StepOptimiser
            Optimiser to update.
        grad_clip_norm : float, optional
            Maximum gradient norm; non-positive values disable clipping.
        observed_channel_indices : torch.Tensor | None, optional
            Canonical observed microphone indices.

        Returns
        -------
        dict[str, float]
            Step losses.
        """
        if observed_channel_indices is None:
            raise ValueError("variable-channel training requires observed_channel_indices")
        if isinstance(optimiser, dict):
            raise TypeError("VariableSRCNNUpsampler requires one optimiser")
        optimiser.zero_grad(set_to_none=True)
        pred = self._forward_no_metrics(S_low, observed_channel_indices)
        loss, stats = self.compute_loss(pred, S_high, observed_channel_indices)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        optimiser.step()
        return stats

    @torch.no_grad()
    def validation_step(
        self,
        S_low: torch.Tensor,
        S_high: torch.Tensor,
        observed_channel_indices: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one variable-channel validation step.

        Parameters
        ----------
        S_low : torch.Tensor
            Sparse input CSM.
        S_high : torch.Tensor
            Full target CSM.
        observed_channel_indices : torch.Tensor | None, optional
            Canonical observed microphone indices.

        Returns
        -------
        dict[str, float]
            Step losses.
        """
        if observed_channel_indices is None:
            raise ValueError("variable-channel validation requires observed_channel_indices")
        pred = self._forward_no_metrics(S_low, observed_channel_indices)
        return self.compute_loss(pred, S_high, observed_channel_indices)[1]
