"""AINN upsampling followed by LAM inference."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
from torch import nn

from lam_min.model.LAM import LAM
from upsampler.ainn import AINNUpsampler


class AINNLAM(nn.Module):
    """
    AINN upsampling followed by LAM inference.

    Parameters
    ----------
    num_bands : int, optional
        Number of frequency bands (default: 16).
    in_channels : int, optional
        Number of low-resolution channels (default: 4).
    out_channels : int, optional
        Number of high-resolution channels (default: 32).
    hidden_channels : int, optional
        Width of hidden AINN MLP layers (default: 64).
    latent_channels : int, optional
        Retained for config compatibility; currently unused in the active AINN implementation
        (default: 64).
    low_channel_indices : Sequence[int], optional
        Zero-based Eigenmike channel indices used for the low-resolution branch.
    loss_name : str, optional
        Reconstruction criterion for the upsampler (default: "mse").
    pde_loss_weight : float, optional
        Weight of Helmholtz residual regularisation in AINN training (default: 0.01).
    pde_freq_min_hz : float, optional
        Minimum PDE-loss frequency used by the wrapped AINN upsampler (default: 100.0).
    pde_freq_max_hz : float, optional
        Maximum PDE-loss frequency used by the wrapped AINN upsampler (default: 4000.0).
    sound_speed : float, optional
        Speed of sound used by the wrapped AINN upsampler (default: 340.0).
    freeze_lam : bool, optional
        If True, freeze LAM parameters so only the upsampler is trainable
        when this wrapper is used in training code (default: True).
    """

    UPSAMPLER_RESULT_WITH_METRICS_LEN = 2

    def __init__(  # noqa: PLR0913
        self,
        num_bands: int = 16,
        in_channels: int = 4,
        out_channels: int = 32,
        hidden_channels: int = 64,
        latent_channels: int = 64,
        low_channel_indices: Sequence[int] = (5, 9, 21, 25),
        loss_name: str = "mse",
        pde_loss_weight: float = 0.01,
        pde_freq_min_hz: float = 100.0,
        pde_freq_max_hz: float = 4000.0,
        sound_speed: float = 340.0,
        freeze_lam: bool = True,
    ) -> None:
        super().__init__()
        self.upsampler = AINNUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            low_channel_indices=low_channel_indices,
            loss_name=loss_name,
            pde_loss_weight=pde_loss_weight,
            pde_freq_min_hz=pde_freq_min_hz,
            pde_freq_max_hz=pde_freq_max_hz,
            sound_speed=sound_speed,
        )
        self.lam = LAM(num_bands=num_bands, Nch=out_channels)
        self._last_upsampler_output: torch.Tensor | None = None
        if freeze_lam:
            for param in self.lam.parameters():
                param.requires_grad = False

    def forward_components(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a forward pass and expose intermediate tensors.

        Parameters
        ----------
        S : torch.Tensor
            Complex low-resolution CSM tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Upsampled CSM, final LAM reconstruction, and latent map.
        """
        S_pred = self.upsampler(S, collect_metrics=False).to(dtype=self.lam.D.dtype)
        self._last_upsampler_output = S_pred.detach()
        out, x, _ = self.lam(S_pred, collect_metrics=False)
        return S_pred, out, x

    def forward(
        self, S, collect_metrics=False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        """
        Forward pass for AINN + LAM.

        Parameters
        ----------
        S : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).
        collect_metrics : bool, optional
            If True, return combined runtime metrics (default: False).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]
            A tuple containing:
            - out: The output tensor from the LAM model.
            - x: The intermediate output from the LAM model.
            - metrics: A dictionary of performance metrics if collect_metrics is True,
              otherwise None.
        """
        self._last_upsampler_output = None
        if collect_metrics:
            metrics = {}

            upsampler_start = time.perf_counter()
            upsampler_result = self.upsampler(S, collect_metrics=True)
            upsampler_end = time.perf_counter()

            if (
                isinstance(upsampler_result, tuple)
                and len(upsampler_result) == self.UPSAMPLER_RESULT_WITH_METRICS_LEN
                and isinstance(upsampler_result[1], dict)
            ):
                S_pred, upsampler_metrics = upsampler_result
            elif isinstance(upsampler_result, tuple):
                S_pred = upsampler_result[0]
                upsampler_metrics = {}
            else:
                S_pred = upsampler_result
                upsampler_metrics = {}

            metrics.update(upsampler_metrics)
            metrics.setdefault("upsampler_time_ms", (upsampler_end - upsampler_start) * 1000.0)
            metrics.setdefault("upsampler_flops", 0.0)
            metrics.setdefault("upsampler_flops_per_frame", 0.0)
            metrics.setdefault("upsampler_memory_mb", 0.0)

            S_pred = S_pred.to(dtype=self.lam.D.dtype)
            self._last_upsampler_output = S_pred.detach()

            out, x, lam_metrics = self.lam(S_pred, collect_metrics=True)
            for key, value in lam_metrics.items():
                metrics[f"lam_{key}"] = value

            metrics["total_time_ms"] = metrics.get("upsampler_time_ms", 0) + metrics.get(
                "lam_total_time_ms", 0
            )
            metrics["total_flops"] = metrics.get("upsampler_flops", 0) + metrics.get("lam_flops", 0)
            metrics["total_memory_mb"] = metrics.get("upsampler_memory_mb", 0) + metrics.get(
                "lam_memory_mb", 0
            )

            num_frames = S.shape[0]
            metrics["num_frames"] = num_frames
            if num_frames > 0:
                metrics["latency_per_frame_ms"] = metrics["total_time_ms"] / num_frames
                metrics["flops_per_frame"] = metrics["total_flops"] / num_frames

            return out, x, metrics

        _, out, x = self.forward_components(S)
        return out, x, None
