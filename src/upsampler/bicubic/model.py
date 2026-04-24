"""
Bicubic interpolation baseline for complex CSM upsampling.

This module implements a 4x4 -> 32x32 upsampler by applying bicubic
interpolation to real and imaginary parts independently. The final
output is projected onto the Hermitian matrix space.

References
----------
.. [1] PyTorch interpolate API:
       https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
"""

from __future__ import annotations

import time
import tracemalloc
from typing import cast

import torch
import torch.nn.functional as torch_f
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from upsampler.base import TrainableUpsampler


class BicubicUpsampler(TrainableUpsampler):  # type: ignore[no-any-unimported]
    """
    Bicubic upsampler using interpolation only.

    Attributes
    ----------
    in_channels : int, optional
        Number of input microphone channels (default: 4).
    out_channels : int, optional
        Number of output microphone channels (default: 32).
    loss_name : str, optional
        Name of the loss function used by ``compute_loss`` for training
        scripts and evaluation (default: "l1").
    """

    EXPECTED_INPUT_NDIM = 4
    # Cubic interpolation uses a 4x4 neighbourhood (16 samples) per output sample.
    BICUBIC_NEIGHBOURHOOD = 16

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 32,
        loss_name: str = "l1",
    ) -> None:
        """
        Initialise the bicubic upsampler model.

        Parameters
        ----------
        in_channels : int, optional
            Number of input microphone channels (default: 4).
        out_channels : int, optional
            Number of output microphone channels (default: 32).
        loss_name : str, optional
            Name of the loss function used by ``compute_loss`` (default: "l1").
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_name = loss_name.strip().lower()
        self.loss_fn = self._build_loss(self.loss_name)

    def _build_loss(self, loss_name: str) -> nn.Module:
        """
        Build standard PyTorch loss function used for training.

        Supported loss names: "l1", "mse".

        Parameters
        ----------
        loss_name : str
            Name of the loss function to use.

        Returns
        -------
        nn.Module
            PyTorch loss function module.
        """
        if loss_name == "l1":
            return nn.L1Loss()
        if loss_name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss_name '{loss_name}'. Use one of: l1, mse.")

    def forward(
        self, S_low: torch.Tensor, collect_metrics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Upsample complex CSM tensors from [B, F, 4, 4] to [B, F, 32, 32].

        With B= batch size, F=number of frequency bands, and 4/32 = number of microphone channels.

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

        batch_size, num_bands, _, _ = S_low.shape

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
        measured_flops = float(flop_counter.get_total_flops())
        # Estimate bicubic interpolation FLOPs and add to measured FLOPs for total count.
        interpolation_flops = float(self._estimate_bicubic_interp_flops(batch_size, num_bands))
        total_flops = measured_flops + interpolation_flops
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
        Compute reconstruction loss with a standard PyTorch criterion.

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
            Total loss tensor and dictionary of loss components for logging.
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

        # Bicubic interpolation is applied to real and imaginary parts separately.
        real = S_low.real.reshape(batch_size * num_bands, 1, self.in_channels, self.in_channels)
        imag = S_low.imag.reshape(batch_size * num_bands, 1, self.in_channels, self.in_channels)

        real_up = torch_f.interpolate(
            real,
            size=(self.out_channels, self.out_channels),
            mode="bicubic",
            align_corners=False,
        )
        imag_up = torch_f.interpolate(
            imag,
            size=(self.out_channels, self.out_channels),
            mode="bicubic",
            align_corners=False,
        )

        S_up = torch.complex(real_up[:, 0], imag_up[:, 0]).reshape(
            batch_size, num_bands, self.out_channels, self.out_channels
        )
        return 0.5 * (S_up + S_up.transpose(-1, -2).conj())

    def _estimate_bicubic_interp_flops(self, batch_size: int, num_bands: int) -> int:
        """
        Estimate bicubic interpolation FLOPs for real+imaginary branches.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        num_bands : int
            Number of frequency bands.

        Returns
        -------
        int
        Estimated number of FLOPs for bicubic interpolation.

        Notes
        -----
        Why estimate instead of relying only on `FlopCounterMode`:
        PyTorch FLOP counting only reports operators with registered formulas in
        `torch.utils.flop_counter.flop_registry`. Bicubic interpolation is not
        listed there in torch v2.10.0, so it can be under-counted if we do not
        add an explicit estimate.

        Source links
        ------------
        - FLOP registry in PyTorch v2.10.0:
          https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L580-L602
        - FLOP counting dispatch check (`if func_packet in self.flop_registry`):
          https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L778-L785
        - PyTorch guide for adding missing FLOP formulas:
          https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html

        Estimation model
        ----------------
        Uses 16 multiply-adds per output sample per branch (4x4 neighbourhood),
        i.e. 32 FLOPs per sample per branch, for both real and imaginary paths.
        """
        output_samples = batch_size * num_bands * self.out_channels * self.out_channels
        branches = 2  # real + imaginary interpolation
        flops_per_sample_per_branch = self.BICUBIC_NEIGHBOURHOOD * 2
        return int(output_samples * branches * flops_per_sample_per_branch)

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
            PyTorch optimiser for updating model parameters.
        grad_clip_norm : float, optional
            Maximum norm for gradient clipping (default: 0.0, no clipping).

        Returns
        -------
        dict[str, float]
            Dictionary of training statistics, including loss components.
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
            Dictionary of validation statistics, including loss components.
        """
        pred = cast(torch.Tensor, self.forward(S_low, collect_metrics=False))
        _, stats = self.compute_loss(pred=pred, target=S_high)
        return self.normalise_step_stats(stats)  # type: ignore[no-any-return]
