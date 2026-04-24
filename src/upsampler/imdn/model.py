"""
IMDN upsampler for complex CSM super-resolution.

This module adapts IMDN from image super-resolution to complex cross-spectral
matrix (CSM) upsampling:

1. Split complex CSM into real/imaginary channels.
2. Apply IMDN feature distillation blocks and sub-pixel upsampling.
3. Recompose complex output and project to the Hermitian matrix space.

References
----------
.. [1] Z. Hui, X. Gao, Y. Yang, X. Wang,
       "Lightweight Image Super-Resolution with Information Multi-distillation Network,"
       ACM MM, 2019, https://arxiv.org/pdf/1909.11856v1
.. [2] Official implementation:
       https://github.com/Zheng222/IMDN
"""

from __future__ import annotations

import time
import tracemalloc
from typing import cast

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from upsampler.base import TrainableUpsampler
from upsampler.imdn import layers as imdn_layers


class IMDNUpsampler(TrainableUpsampler):  # type: ignore[no-any-unimported]
    """
    IMDN-based upsampler for complex CSM tensors.

    Parameters
    ----------
    in_channels : int, optional
        Number of low-resolution microphone channels per CSM axis (default: 4).
    out_channels : int, optional
        Number of high-resolution microphone channels per CSM axis (default: 32).
    feature_channels : int, optional
        Number of feature channels in the IMDN trunk (default: 64).
    mapping_channels : int, optional
        Channel width in the 1x1 fusion bottleneck (default: 32).
    loss_name : str, optional
        Loss function name, either "l1" or "mse" (default: "l1").
    """

    EXPECTED_INPUT_NDIM = 4
    IMDN_BLOCKS = 6
    COMPLEX_CHANNELS = 2

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 4,
        out_channels: int = 32,
        feature_channels: int = 64,
        mapping_channels: int = 32,
        loss_name: str = "l1",
    ) -> None:
        """
        Initialise the IMDN upsampler.

        Parameters
        ----------
        in_channels : int, optional
            Number of low-resolution microphone channels (default: 4).
        out_channels : int, optional
            Number of high-resolution microphone channels (default: 32).
        feature_channels : int, optional
            Number of channels in IMDN feature extraction blocks (default: 64).
        mapping_channels : int, optional
            Number of channels in the fusion bottleneck block (default: 32).
        loss_name : str, optional
            Loss function name, either "l1" or "mse" (default: "l1").
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_channels = int(feature_channels)
        self.mapping_channels = int(mapping_channels)
        self.loss_name = loss_name.strip().lower()

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be > 0.")
        if self.mapping_channels <= 0:
            raise ValueError("mapping_channels must be > 0.")
        if self.out_channels % self.in_channels != 0:
            raise ValueError(
                "out_channels must be an integer multiple of in_channels "
                f"for pixel shuffle. Got in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}."
            )
        self.upscale_factor = self.out_channels // self.in_channels

        self.fea_conv = nn.Conv2d(
            in_channels=self.COMPLEX_CHANNELS,
            out_channels=self.feature_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dtype=torch.float32,
        )

        # Keep original IMDN naming to retain intuitive checkpoint keys.
        self.IMDB1 = imdn_layers.IMDModule(in_channels=self.feature_channels)
        self.IMDB2 = imdn_layers.IMDModule(in_channels=self.feature_channels)
        self.IMDB3 = imdn_layers.IMDModule(in_channels=self.feature_channels)
        self.IMDB4 = imdn_layers.IMDModule(in_channels=self.feature_channels)
        self.IMDB5 = imdn_layers.IMDModule(in_channels=self.feature_channels)
        self.IMDB6 = imdn_layers.IMDModule(in_channels=self.feature_channels)

        self.c = imdn_layers.conv_block(
            self.feature_channels * self.IMDN_BLOCKS,
            self.mapping_channels,
            kernel_size=1,
            act_type="lrelu",
        )

        self.LR_conv = nn.Conv2d(
            in_channels=self.mapping_channels,
            out_channels=self.feature_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dtype=torch.float32,
        )

        self.upsampler: nn.Conv2d | nn.Sequential
        if self.upscale_factor == 1:
            self.upsampler = nn.Conv2d(
                in_channels=self.feature_channels,
                out_channels=self.COMPLEX_CHANNELS,
                kernel_size=3,
                padding=1,
                bias=True,
                dtype=torch.float32,
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.feature_channels,
                    out_channels=self.COMPLEX_CHANNELS * (self.upscale_factor**2),
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    dtype=torch.float32,
                ),
                nn.PixelShuffle(self.upscale_factor),
            )

        self.loss_fn = self._build_loss(self.loss_name)
        self._init_weights()

    def _build_loss(self, loss_name: str) -> nn.Module:
        """
        Build standard PyTorch loss function used for training.

        Parameters
        ----------
        loss_name : str
            Name of the loss function to use.

        Returns
        -------
        nn.Module
            PyTorch loss module.
        """
        if loss_name == "l1":
            return nn.L1Loss()
        if loss_name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss_name '{loss_name}'. Use one of: l1, mse.")

    def _init_weights(self) -> None:
        """
        Initialise convolution kernels.

        Kaiming normal is used for hidden layers and Xavier normal for the
        final reconstruction convolution.
        """
        final_conv = (
            self.upsampler[0] if isinstance(self.upsampler, nn.Sequential) else self.upsampler
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is final_conv:
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(
                        module.weight,
                        a=0.05,
                        mode="fan_in",
                        nonlinearity="leaky_relu",
                    )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _imdn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run IMDN trunk and reconstruction head on real-valued 2-channel input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, 2, in_channels, in_channels].

        Returns
        -------
        torch.Tensor
            Real-valued output tensor with shape [batch, 2, out_channels, out_channels].
        """
        out_fea = self.fea_conv(x)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        return self.upsampler(out_lr)  # type: ignore[no-any-return]

    def forward(
        self, S_low: torch.Tensor, collect_metrics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Upsample complex CSM tensors from [B, F, in_channels, in_channels].

        Parameters
        ----------
        S_low : torch.Tensor
            Complex tensor with shape [batch, num_bands, in_channels, in_channels].
        collect_metrics : bool, optional
            Whether to return runtime metrics (default: False).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, dict[str, float]]
            Upsampled complex tensor, or tensor and metrics dictionary.
        """
        if S_low.ndim != self.EXPECTED_INPUT_NDIM:
            raise ValueError(f"S_low must be 4D [B, F, C, C], got shape={tuple(S_low.shape)}")
        if not S_low.is_complex():
            raise ValueError("S_low must be a complex tensor.")
        if S_low.shape[-2:] != (self.in_channels, self.in_channels):
            raise ValueError(
                "S_low must have shape [B, F, "
                f"{self.in_channels}, {self.in_channels}] on the last two dimensions, "
                f"got {tuple(S_low.shape)}"
            )

        use_cuda = S_low.device.type == "cuda"
        start = 0.0
        if collect_metrics:
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(S_low.device)
                torch.cuda.synchronize(S_low.device)
            else:
                tracemalloc.start()
            start = time.perf_counter()

        batch_size = int(S_low.shape[0])
        S_up = self._forward_no_metrics(S_low)

        if not collect_metrics:
            return S_up

        if use_cuda:
            torch.cuda.synchronize(S_low.device)
        end = time.perf_counter()

        if use_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(S_low.device) / (1024.0 * 1024.0)
        else:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory / (1024.0 * 1024.0)

        flop_counter = FlopCounterMode(display=False)
        with torch.no_grad():
            with flop_counter:
                self._forward_no_metrics(S_low)
        total_flops = float(flop_counter.get_total_flops())
        flops_per_frame = total_flops / batch_size if batch_size > 0 else 0.0

        metrics: dict[str, float] = {
            "upsampler_time_ms": (end - start) * 1000.0,
            "upsampler_flops": total_flops,
            "upsampler_flops_per_frame": flops_per_frame,
            "upsampler_memory_mb": float(peak_memory_mb),
            "num_frames": float(batch_size),
        }
        return S_up, metrics

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute complex reconstruction loss.

        Complex tensors are split into real/imag parts and loss is averaged.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted complex tensor with shape [B, F, out_channels, out_channels].
        target : torch.Tensor
            Target complex tensor with shape [B, F, out_channels, out_channels].

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Total loss tensor and scalar loss statistics.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

        loss_real = self.loss_fn(pred.real, target.real)
        loss_imag = self.loss_fn(pred.imag, target.imag)
        total = 0.5 * (loss_real + loss_imag)

        total_value = float(total.detach().cpu().item())
        stats: dict[str, float] = {"loss_total": total_value}
        if self.loss_name == "l1":
            stats["loss_l1"] = total_value
        elif self.loss_name == "mse":
            stats["loss_mse"] = total_value
        return total, self.normalise_step_stats(stats)

    def _forward_no_metrics(self, S_low: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without timing/FLOP/memory collection.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex tensor with shape [batch, num_bands, in_channels, in_channels].

        Returns
        -------
        torch.Tensor
            Upsampled complex tensor with shape
            [batch, num_bands, out_channels, out_channels].
        """
        batch_size, num_bands, _, _ = S_low.shape

        real = S_low.real.reshape(batch_size * num_bands, 1, self.in_channels, self.in_channels)
        imag = S_low.imag.reshape(batch_size * num_bands, 1, self.in_channels, self.in_channels)
        imdn_input = torch.cat((real, imag), dim=1)

        imdn_output = self._imdn_forward(imdn_input.to(torch.float32)).to(imdn_input.dtype)
        if imdn_output.shape[1] != self.COMPLEX_CHANNELS:
            raise RuntimeError(
                "IMDN output channel mismatch. Expected "
                f"{self.COMPLEX_CHANNELS}, got {imdn_output.shape[1]}."
            )
        if imdn_output.shape[-2:] != (self.out_channels, self.out_channels):
            raise RuntimeError(
                "IMDN output spatial shape mismatch. Expected "
                f"({self.out_channels}, {self.out_channels}), got {tuple(imdn_output.shape[-2:])}."
            )

        S_up = torch.complex(imdn_output[:, 0], imdn_output[:, 1]).reshape(
            batch_size, num_bands, self.out_channels, self.out_channels
        )

        return 0.5 * (S_up + S_up.transpose(-1, -2).conj())

    def training_step(
        self,
        S_low: torch.Tensor,
        S_high: torch.Tensor,
        optimiser: torch.optim.Optimizer,
        grad_clip_norm: float = 0.0,
    ) -> dict[str, float]:
        """
        Run one optimisation step (forward + loss + backward + update).

        Parameters
        ----------
        S_low : torch.Tensor
            Input complex tensor with shape [B, F, in_channels, in_channels].
        S_high : torch.Tensor
            Target complex tensor with shape [B, F, out_channels, out_channels].
        optimiser : torch.optim.Optimizer
            Optimiser for parameter updates.
        grad_clip_norm : float, optional
            Maximum norm for gradient clipping (default: 0.0, no clipping).

        Returns
        -------
        dict[str, float]
            Scalar training statistics.
        """
        optimiser.zero_grad(set_to_none=True)
        pred = cast(torch.Tensor, self.forward(S_low, collect_metrics=False))
        loss, stats = self.compute_loss(pred=pred, target=S_high)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_norm)
        optimiser.step()
        return stats

    @torch.no_grad()
    def validation_step(self, S_low: torch.Tensor, S_high: torch.Tensor) -> dict[str, float]:
        """
        Run one validation step (forward + loss only).

        Parameters
        ----------
        S_low : torch.Tensor
            Input complex tensor with shape [B, F, in_channels, in_channels].
        S_high : torch.Tensor
            Target complex tensor with shape [B, F, out_channels, out_channels].

        Returns
        -------
        dict[str, float]
            Scalar validation statistics.
        """
        pred = cast(torch.Tensor, self.forward(S_low, collect_metrics=False))
        _, stats = self.compute_loss(pred=pred, target=S_high)
        return self.normalise_step_stats(stats)  # type: ignore[no-any-return]
