"""
Spatially-Adapted Feature Modulation for Efficient Image Super-Resolution (SAFMN) then LAM.

This module defines the SAFMNLAM class, which combines a SAFMN-based upsampler with the LAM model
for super-resolution of low-resolution CSM tensors.
"""

from __future__ import annotations

import torch
from torch import nn

from lam_min.model.LAM import LAM
from upsampler.safmn import SAFMNUpsampler


class SAFMNLAM(nn.Module):
    """
    SAFMN upsampling followed by LAM inference.

    Parameters
    ----------
    num_bands : int, optional
        Number of frequency bands (default: 16).
    in_channels : int, optional
        Number of low-resolution channels (default: 4).
    out_channels : int, optional
        Number of high-resolution channels (default: 32).
    feature_channels : int, optional
        Number of channels in SAFMN feature blocks (default: 36).
    n_blocks : int, optional
        Number of SAFMN feature mixing blocks (default: 8).
    ffn_scale : float, optional
        Expansion ratio in SAFMN CCM blocks (default: 2.0).
    n_levels : int, optional
        Number of SAFM multi-scale levels per block (default: 4).
    loss_name : str, optional
        Reconstruction loss used by the wrapped SAFMN upsampler (default: "l1").
    fft_loss_weight : float, optional
        FFT-domain loss weight used by the wrapped SAFMN upsampler (default: 0.05).
    freeze_lam : bool, optional
        If True, freeze LAM parameters so only the upsampler is trainable
        when this wrapper is used in training code (default: True).
    """

    LAM_OUTPUT_WITH_METRICS_LEN = 3

    def __init__(  # noqa: PLR0913
        self,
        num_bands: int = 16,
        in_channels: int = 4,
        out_channels: int = 32,
        feature_channels: int = 36,
        n_blocks: int = 8,
        ffn_scale: float = 2.0,
        n_levels: int = 4,
        loss_name: str = "l1",
        fft_loss_weight: float = 0.05,
        freeze_lam: bool = True,
    ) -> None:
        super().__init__()
        self.upsampler = SAFMNUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            n_blocks=n_blocks,
            ffn_scale=ffn_scale,
            n_levels=n_levels,
            loss_name=loss_name,
            fft_loss_weight=fft_loss_weight,
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
        Forward pass for SAFMN + LAM.

        Parameters
        ----------
        S : torch.Tensor
            Low-resolution input tensor of shape (batch_size, in_channels, height, width).
        collect_metrics : bool, optional
            If True, collect and return performance metrics during inference (default: False).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]
            A tuple containing:
            - out: The output tensor from the LAM model, with shape (batch_size, num_bands,
              out_channels, out_channels).
            - x: The intermediate output from the LAM model before the final output layer, with
              shape (batch_size, num_bands, out_channels, out_channels).
            - metrics: A dictionary of performance metrics if collect_metrics is True,
              otherwise None.
        """
        self._last_upsampler_output = None
        if collect_metrics:
            metrics = {}

            S_pred, upsampler_metrics = self.upsampler(S, collect_metrics=True)
            S_pred = S_pred.to(dtype=self.lam.D.dtype)
            self._last_upsampler_output = S_pred.detach()
            metrics.update(upsampler_metrics)

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
