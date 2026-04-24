"""Shared training interface for trainable upsamplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
from torch import nn

StepOptimiser: TypeAlias = torch.optim.Optimizer | dict[str, torch.optim.Optimizer]


class TrainableUpsampler(nn.Module, ABC):
    """
    Base interface for model-owned optimisation logic.

    Makes sure that all trainable upsampler models implement the same training and validation step
    signatures, and that they return a consistent dictionary of loss statistics for logging
    and checkpointing.
    """

    REQUIRED_LOSS_KEYS = ("loss_total",)

    @staticmethod
    def normalise_step_stats(stats: dict[str, float]) -> dict[str, float]:
        """
        Normalise per-step statistics to the trainer schema.

        This method ensures that the returned dictionary includes all required loss keys.

        Parameters
        ----------
        stats : dict[str, float]
            Dictionary of loss statistics from the training/validation step.

        Returns
        -------
        dict[str, float]
            Normalised dictionary of loss statistics, including all required keys.

        Raises
        ------
        ValueError
            If any required loss key is missing from the input statistics.
        """
        if "loss_total" not in stats:
            raise ValueError("Step statistics must include 'loss_total'.")

        out = {key: float(stats.get(key, 0.0)) for key in TrainableUpsampler.REQUIRED_LOSS_KEYS}
        for key, value in stats.items():
            if key not in out:
                out[key] = float(value)
        return out

    @abstractmethod
    def training_step(
        self,
        S_low: torch.Tensor,
        S_high: torch.Tensor,
        optimiser: StepOptimiser,
        grad_clip_norm: float = 0.0,
    ) -> dict[str, float]:
        """
        Run one optimisation step and return scalar loss statistics.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).
        S_high : torch.Tensor
            Complex high-resolution CSM tensor with shape
            (batch, num_bands, out_channels, out_channels).
        optimiser : StepOptimiser
            Optimiser input for the training step. This can be a single optimiser
            (for non-adversarial models) or a dictionary of optimisers for models
            with multiple parameter groups (e.g., GAN generator/discriminator).
        grad_clip_norm : float, optional
            If > 0, clip gradients to this maximum norm (default: 0.0, no clipping).

        Returns
        -------
        dict[str, float]
            Dictionary of scalar loss statistics for this training step.

        Raises
        ------
        NotImplementedError
            If the training step is not implemented by the subclass.
        """
        raise NotImplementedError("Training step not implemented for base class.")

    @abstractmethod
    def validation_step(self, S_low: torch.Tensor, S_high: torch.Tensor) -> dict[str, float]:
        """
        Run one validation step and return scalar loss statistics.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).
        S_high : torch.Tensor
            Complex high-resolution CSM tensor with shape
            (batch, num_bands, out_channels, out_channels).

        Returns
        -------
        dict[str, float]
            Dictionary of scalar loss statistics for this validation step.

        Raises
        ------
        NotImplementedError
            If the validation step is not implemented by the subclass.
        """
        raise NotImplementedError("Validation step not implemented for base class.")
