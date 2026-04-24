"""
Super-resolution convolutional neural network (SRCNN) upsampling followed by LAM inference.

This module defines the SRCNNLAM class, which combines a SRCNN-based upsampler with the LAM model
for super-resolution of low-resolution CSM tensors.
"""

from __future__ import annotations

import torch
from torch import nn

from lam_min.model.LAM import LAM
from upsampler.srcnn import SRCNNUpsampler


class SRCNNLAM(nn.Module):
    """
    SRCNN upsampling followed by LAM inference.

    Parameters
    ----------
    num_bands : int, optional
        Number of frequency bands (default: 16).
    in_channels : int, optional
        Number of low-resolution channels (default: 4).
    out_channels : int, optional
        Number of high-resolution channels (default: 32).
    feature_channels : int, optional
        Number of channels in the SRCNN feature extraction block (default: 64).
    mapping_channels : int, optional
        Number of channels in the SRCNN mapping block (default: 32).
    loss_name : str, optional
        Reconstruction loss used by the wrapped SRCNN upsampler (default: "l1").
    freeze_lam : bool, optional
        If True, freeze LAM parameters so only the upsampler is trainable
        when this wrapper is used in training code (default: True).
    """

    def __init__(  # noqa: PLR0913
        self,
        num_bands: int = 16,
        in_channels: int = 4,
        out_channels: int = 32,
        feature_channels: int = 64,
        mapping_channels: int = 32,
        loss_name: str = "l1",
        freeze_lam: bool = True,
    ) -> None:
        super().__init__()
        self.upsampler = SRCNNUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            mapping_channels=mapping_channels,
            loss_name=loss_name,
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
        Forward pass for SRCNN + LAM.

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
