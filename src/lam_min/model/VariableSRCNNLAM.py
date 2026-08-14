"""Variable-microphone masked SRCNN completion followed by LAM."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from data.variable_channels import DEFAULT_VARIABLE_CHANNEL_COUNTS
from lam_min.model.LAM import LAM
from upsampler.srcnn import VariableSRCNNUpsampler


class VariableSRCNNLAM(nn.Module):
    """Complete a canonical CSM from arbitrary trained microphone subsets before LAM."""

    def __init__(  # noqa: PLR0913
        self,
        num_bands: int = 16,
        out_channels: int = 32,
        feature_channels: int = 64,
        mapping_channels: int = 32,
        loss_name: str = "l1",
        variable_input_channel_counts: Sequence[int] = DEFAULT_VARIABLE_CHANNEL_COUNTS,
        freeze_lam: bool = True,
    ) -> None:
        """Initialise variable SRCNN and LAM branches.

        Parameters
        ----------
        num_bands : int, optional
            LAM frequency-band count.
        out_channels : int, optional
            Canonical microphone count, which must be 32.
        feature_channels : int, optional
            SRCNN feature width.
        mapping_channels : int, optional
            SRCNN mapping width.
        loss_name : str, optional
            Auxiliary completion loss.
        variable_input_channel_counts : Sequence[int], optional
            Trained microphone counts.
        freeze_lam : bool, optional
            Whether to freeze LAM parameters.
        """
        super().__init__()
        self.upsampler = VariableSRCNNUpsampler(
            out_channels=out_channels,
            feature_channels=feature_channels,
            mapping_channels=mapping_channels,
            loss_name=loss_name,
            variable_input_channel_counts=variable_input_channel_counts,
        )
        self.lam = LAM(num_bands=num_bands, Nch=out_channels)
        self._last_upsampler_output: torch.Tensor | None = None
        if freeze_lam:
            for parameter in self.lam.parameters():
                parameter.requires_grad = False

    def forward_components(
        self, S: torch.Tensor, observed_channel_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return completed CSM, LAM reconstruction, and latent map.

        Parameters
        ----------
        S : torch.Tensor
            Sparse canonical CSM.
        observed_channel_indices : torch.Tensor
            Canonical observed microphone indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Completed CSM, reconstructed CSM, and latent map.
        """
        S_pred = self.upsampler(S, observed_channel_indices).to(dtype=self.lam.D.dtype)
        self._last_upsampler_output = S_pred.detach()
        out, latent, _ = self.lam(S_pred, collect_metrics=False)
        return S_pred, out, latent

    def forward(
        self,
        S: torch.Tensor,
        observed_channel_indices: torch.Tensor,
        collect_metrics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        """Run variable SRCNN completion followed by LAM inference.

        Parameters
        ----------
        S : torch.Tensor
            Sparse canonical CSM.
        observed_channel_indices : torch.Tensor
            Canonical observed microphone indices.
        collect_metrics : bool, optional
            Whether to collect component runtime metrics.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]
            Reconstructed CSM, latent map, and optional metrics.
        """
        self._last_upsampler_output = None
        if not collect_metrics:
            _, out, latent = self.forward_components(S, observed_channel_indices)
            return out, latent, None
        S_pred, metrics = self.upsampler(S, observed_channel_indices, collect_metrics=True)
        S_pred = S_pred.to(dtype=self.lam.D.dtype)
        self._last_upsampler_output = S_pred.detach()
        out, latent, lam_metrics = self.lam(S_pred, collect_metrics=True)
        combined = dict(metrics)
        combined.update({f"lam_{key}": value for key, value in lam_metrics.items()})
        combined["total_time_ms"] = combined.get("upsampler_time_ms", 0) + combined.get(
            "lam_total_time_ms", 0
        )
        combined["total_flops"] = combined.get("upsampler_flops", 0) + combined.get("lam_flops", 0)
        combined["total_memory_mb"] = combined.get("upsampler_memory_mb", 0) + combined.get(
            "lam_memory_mb", 0
        )
        frames = int(S.shape[0])
        combined["num_frames"] = float(frames)
        combined["latency_per_frame_ms"] = combined["total_time_ms"] / max(frames, 1)
        combined["flops_per_frame"] = combined["total_flops"] / max(frames, 1)
        return out, latent, combined
