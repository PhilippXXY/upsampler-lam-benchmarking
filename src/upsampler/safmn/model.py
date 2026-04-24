"""
SAFMN upsampler for complex CSM super-resolution.

This module adapts SAFMN from image super-resolution to complex cross-spectral
matrix (CSM) upsampling:

1. Split complex CSM into real/imaginary channels.
2. Apply SAFMN feature mixing blocks (SAFM + CCM).
3. Recompose complex output and project to the Hermitian matrix space.

References
----------
.. [1] L. Sun, J. Dong, J. Tang, J. Pan,
       "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution",
       ICCV 2023,
       https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.pdf
.. [2] Official implementation:
       https://github.com/sunny2109/SAFMN
"""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import cast

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from upsampler.base import TrainableUpsampler
from upsampler.safmn.layers import AttBlock


class SAFMNUpsampler(TrainableUpsampler):  # type: ignore[no-any-unimported]
    """
    SAFMN-based upsampler for complex CSM tensors.

    Parameters
    ----------
    in_channels : int, optional
        Number of low-resolution microphone channels per CSM axis (default: 4).
    out_channels : int, optional
        Number of high-resolution microphone channels per CSM axis (default: 32).
    feature_channels : int, optional
        Number of feature channels in the SAFMN trunk (default: 36).
    n_blocks : int, optional
        Number of SAFMN feature mixing blocks (default: 8).
    ffn_scale : float, optional
        Expansion ratio used in each CCM block (default: 2.0).
    n_levels : int, optional
        Number of multi-scale levels in each SAFM block (default: 4).
    loss_name : str, optional
        Loss function name, one of "l1", "mse" (default: "l1").
    fft_loss_weight : float, optional
        Weight for the FFT-domain term in the paper-style composite loss
        (default: 0.05).
    """

    EXPECTED_INPUT_NDIM = 4
    COMPLEX_CHANNELS = 2

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 4,
        out_channels: int = 32,
        feature_channels: int = 36,
        n_blocks: int = 8,
        ffn_scale: float = 2.0,
        n_levels: int = 4,
        loss_name: str = "l1",
        fft_loss_weight: float = 0.05,
    ) -> None:
        """
        Initialise the SAFMN upsampler.

        Parameters
        ----------
        in_channels : int, optional
            Number of low-resolution microphone channels (default: 4).
        out_channels : int, optional
            Number of high-resolution microphone channels (default: 32).
        feature_channels : int, optional
            Number of channels in SAFMN feature extraction blocks (default: 36).
        n_blocks : int, optional
            Number of SAFMN feature mixing blocks (default: 8).
        ffn_scale : float, optional
            Expansion ratio used by CCM in each block (default: 2.0).
        n_levels : int, optional
            Number of SAFM scales in each block (default: 4).
        loss_name : str, optional
            Loss function name, one of "l1", "mse"
            (default: "l1").
        fft_loss_weight : float, optional
            Weight for the FFT-domain term in the paper-style
            composite loss (default: 0.05).
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_channels = int(feature_channels)
        self.n_blocks = int(n_blocks)
        self.ffn_scale = float(ffn_scale)
        self.n_levels = int(n_levels)
        self.loss_name = loss_name.strip().lower()
        self.fft_loss_weight = float(fft_loss_weight)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be > 0.")
        if self.n_blocks <= 0:
            raise ValueError("n_blocks must be > 0.")
        if self.ffn_scale <= 0.0:
            raise ValueError("ffn_scale must be > 0.")
        if self.n_levels <= 0:
            raise ValueError("n_levels must be > 0.")
        if self.fft_loss_weight < 0.0:
            raise ValueError("fft_loss_weight must be >= 0.")
        if self.out_channels % self.in_channels != 0:
            raise ValueError(
                "out_channels must be an integer multiple of in_channels. "
                f"Got in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}."
            )
        self.upscale_factor = self.out_channels // self.in_channels

        self.to_feat = nn.Conv2d(
            in_channels=self.COMPLEX_CHANNELS,
            out_channels=self.feature_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dtype=torch.float32,
        )

        self.feats = nn.Sequential(
            *[
                AttBlock(
                    dim=self.feature_channels,
                    ffn_scale=self.ffn_scale,
                    n_levels=self.n_levels,
                )
                for _ in range(self.n_blocks)
            ]
        )

        # If upscale_factor == 1, we can use a single convolution for the final projection.
        self.to_img: nn.Conv2d | nn.Sequential
        if self.upscale_factor == 1:
            self.to_img = nn.Conv2d(
                in_channels=self.feature_channels,
                out_channels=self.COMPLEX_CHANNELS,
                kernel_size=3,
                padding=1,
                bias=True,
                dtype=torch.float32,
            )
        else:
            self.to_img = nn.Sequential(
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
        if loss_name in {"l1"}:
            return nn.L1Loss()
        if loss_name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss_name '{loss_name}'. Use one of: l1, mse")

    def _fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT-domain reconstruction loss.

        The implementation mirrors the official SAFMN code:
        - `torch.fft.rfft2` on prediction and target
        - stack real/imaginary parts
        - base criterion distance

        Parameters
        ----------
        pred : torch.Tensor
            Real-valued prediction tensor with shape [N, C, H, W].
        target : torch.Tensor
            Real-valued target tensor with shape [N, C, H, W].

        Returns
        -------
        torch.Tensor
            Scalar FFT-domain loss using the selected base criterion.
        """
        # MPS currently emits a noisy internal resize warning for rfft2.
        # We cannot use `out=` here because FFT out-variants do not support autograd.
        if pred.device.type == "mps" or target.device.type == "mps":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="An output with one or more elements was resized since it had shape.*",
                    category=UserWarning,
                )
                pred_fft = torch.fft.rfft2(pred)
                target_fft = torch.fft.rfft2(target)
        else:
            pred_fft = torch.fft.rfft2(pred)
            target_fft = torch.fft.rfft2(target)
        pred_fft_ri = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
        target_fft_ri = torch.stack((target_fft.real, target_fft.imag), dim=-1)
        return self.loss_fn(pred_fft_ri, target_fft_ri)  # type: ignore[no-any-return]

    def _init_weights(self) -> None:
        """
        Initialise convolution kernels.

        Kaiming normal is used for hidden layers and Xavier normal for the
        final reconstruction convolution.
        """
        final_conv = self.to_img[0] if isinstance(self.to_img, nn.Sequential) else self.to_img
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is final_conv:
                    nn.init.xavier_normal_(module.weight)
                else:
                    # gelu is used in SAFM blocks, but pyTorch's kaiming initialisation does not
                    # have a specific mode for gelu
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _safmn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run SAFMN trunk and reconstruction head on real-valued 2-channel input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, 2, in_channels, in_channels].

        Returns
        -------
        torch.Tensor
            Real-valued output tensor with shape [batch, 2, out_channels, out_channels].
        """
        feat = self.to_feat(x)
        mixed = self.feats(feat) + feat
        return self.to_img(mixed)  # type: ignore[no-any-return]

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
        Compute composite SAFMN reconstruction loss.

        For both `l1` and `mse`, this uses the paper-style form:
        base-domain loss plus weighted FFT-domain loss with the same base criterion.

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

        pred_ri = torch.cat((pred.real.unsqueeze(2), pred.imag.unsqueeze(2)), dim=2)
        target_ri = torch.cat((target.real.unsqueeze(2), target.imag.unsqueeze(2)), dim=2)
        pred_ri = pred_ri.reshape(-1, self.COMPLEX_CHANNELS, self.out_channels, self.out_channels)
        target_ri = target_ri.reshape(
            -1, self.COMPLEX_CHANNELS, self.out_channels, self.out_channels
        )

        loss_base = self.loss_fn(pred_ri, target_ri)
        loss_fft = self._fft_loss(pred_ri, target_ri)
        total = loss_base + self.fft_loss_weight * loss_fft

        total_value = float(total.detach().cpu().item())
        loss_base_value = float(loss_base.detach().cpu().item())
        loss_fft_value = float(loss_fft.detach().cpu().item())
        stats: dict[str, float] = {
            "loss_total": total_value,
            "loss_fft_weight": self.fft_loss_weight,
        }
        if self.loss_name in {"l1"}:
            stats["loss_l1"] = loss_base_value
            stats["loss_fft_l1"] = loss_fft_value
        elif self.loss_name == "mse":
            stats["loss_mse"] = loss_base_value
            stats["loss_fft_mse"] = loss_fft_value
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
        safmn_input = torch.cat((real, imag), dim=1)

        safmn_output = self._safmn_forward(safmn_input.to(torch.float32)).to(safmn_input.dtype)
        if safmn_output.shape[1] != self.COMPLEX_CHANNELS:
            raise RuntimeError(
                "SAFMN output channel mismatch. Expected "
                f"{self.COMPLEX_CHANNELS}, got {safmn_output.shape[1]}."
            )
        if safmn_output.shape[-2:] != (self.out_channels, self.out_channels):
            raise RuntimeError(
                "SAFMN output spatial shape mismatch. Expected "
                f"({self.out_channels}, {self.out_channels}), got "
                f"{tuple(safmn_output.shape[-2:])}."
            )

        S_up = torch.complex(safmn_output[:, 0], safmn_output[:, 1]).reshape(
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
