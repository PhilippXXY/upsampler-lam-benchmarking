"""GAN upsampling followed by LAM inference."""

from __future__ import annotations

import torch
from torch import nn

from lam_min.model.LAM import LAM
from upsampler.gan import GANUpsampler


class GANLAM(nn.Module):
    """GAN-based upsampling followed by LAM inference."""

    def __init__(  # noqa: PLR0913
        self,
        num_bands: int = 16,
        in_channels: int = 4,
        out_channels: int = 32,
        feature_channels: int = 128,
        n_residual_blocks: int = 8,
        loss_name: str = "l1",
        adversarial_weight: float = 0.01,
        content_weight: float = 1.0,
        critic_iters: int = 1,
        discriminator_lr_scale: float = 1.0,
        beta1_adam: float = 0.9,
        beta2_adam: float = 0.999,
        freeze_lam: bool = True,
    ) -> None:
        """
        Initialise GANLAM.

        Parameters
        ----------
        num_bands : int, optional
            Number of frequency bands (default: 16).
        in_channels : int, optional
            Number of low-resolution channels (default: 4).
        out_channels : int, optional
            Number of high-resolution channels (default: 32).
        feature_channels : int, optional
            Number of channels in the GAN feature extraction block (default: 128).
        n_residual_blocks : int, optional
            Number of residual blocks in the GAN feature extraction trunk (default: 8).
        loss_name : str, optional
            Reconstruction loss used inside the wrapped GAN upsampler (default: "l1").
        adversarial_weight : float, optional
            Generator adversarial loss weight for the wrapped GAN upsampler (default: 0.01).
        content_weight : float, optional
            Generator reconstruction loss weight for the wrapped GAN upsampler (default: 1.0).
        critic_iters : int, optional
            Critic updates per generator update in the wrapped GAN upsampler (default: 1).
        discriminator_lr_scale : float, optional
            Discriminator learning-rate multiplier in the wrapped GAN upsampler (default: 1.0).
        beta1_adam : float, optional
            Adam beta1 for the wrapped GAN upsampler (default: 0.9).
        beta2_adam : float, optional
            Adam beta2 for the wrapped GAN upsampler (default: 0.999).
        freeze_lam : bool, optional
            Whether to freeze the LAM parameters (default: True).
        """
        super().__init__()
        self.upsampler = GANUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=feature_channels,
            n_residual_blocks=n_residual_blocks,
            loss_name=loss_name,
            adversarial_weight=adversarial_weight,
            content_weight=content_weight,
            critic_iters=critic_iters,
            discriminator_lr_scale=discriminator_lr_scale,
            beta1_adam=beta1_adam,
            beta2_adam=beta2_adam,
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
        Forward pass for GANUpsampler + LAM.

        Parameters
        ----------
        S : torch.Tensor
            Input low-resolution CSM tensor of shape (num_frames, in_channels, num_bands).
        collect_metrics : bool, optional
            Whether to collect runtime and performance metrics from the upsampler and LAM
            (default: False).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]
            A tuple containing:
            - The upsampled CSM tensor of shape (num_frames, out_channels, num_bands).
            - The LAM output tensor of shape (num_frames, num_bands, out_channels, out_channels).
            - A dictionary of collected metrics if `collect_metrics` is True, otherwise None.
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
