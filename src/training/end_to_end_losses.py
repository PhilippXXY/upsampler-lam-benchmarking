"""Shared loss helpers for end-to-end wrapper training."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, cast

import torch
from torch import nn

from lam_min.model.cdbpn import Net as CDBPN
from upsampler.ainn import AINNUpsampler
from upsampler.bicubic import BicubicUpsampler
from upsampler.gan import GANUpsampler
from upsampler.imdn import IMDNUpsampler
from upsampler.safmn import SAFMNUpsampler
from upsampler.srcnn import SRCNNUpsampler


class EndToEndLossModel(Protocol):
    """Structural interface for end-to-end wrapper loss helpers."""

    upsampler: nn.Module
    lam: nn.Module

    def forward_components(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return upsampled CSM, reconstructed CSM, and latent map."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return trainable parameters."""


def _generic_complex_mse(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute a generic complex MSE auxiliary loss.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted complex CSM tensor.
    target : torch.Tensor
        Target complex CSM tensor.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        Raw auxiliary loss and associated statistics.
    """
    loss_real = torch.nn.functional.mse_loss(pred.real, target.real)
    loss_imag = torch.nn.functional.mse_loss(pred.imag, target.imag)
    total = 0.5 * (loss_real + loss_imag)
    return total, {"loss_aux_mse": float(total.detach().cpu().item())}


def compute_auxiliary_loss(
    model: EndToEndLossModel,
    *,
    S_pred: torch.Tensor,
    S_high: torch.Tensor,
    use_model_specific_aux_loss: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the auxiliary reconstruction loss.

    Parameters
    ----------
    model : EndToEndLossModel
        Wrapper model.
    S_pred : torch.Tensor
        Upsampled CSM prediction from the wrapper.
    S_high : torch.Tensor
        Ground-truth high-resolution CSM.
    use_model_specific_aux_loss : bool
        Whether to reuse each upsampler's own reconstruction helper.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        Raw auxiliary loss and its statistics.
    """
    target = S_high.to(dtype=S_pred.dtype)

    if not use_model_specific_aux_loss:
        return _generic_complex_mse(S_pred, target)

    upsampler = model.upsampler
    if isinstance(upsampler, CDBPN):
        return _generic_complex_mse(S_pred, target)
    if isinstance(upsampler, AINNUpsampler):
        compute_loss = cast(
            Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor | None],
                tuple[torch.Tensor, dict[str, float]],
            ],
            upsampler.compute_loss,
        )
        return compute_loss(S_pred, target, None)
    if isinstance(upsampler, GANUpsampler):
        compute_reconstruction_loss = cast(
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float]]],
            upsampler._compute_reconstruction_loss,
        )
        return compute_reconstruction_loss(S_pred, target)
    if isinstance(
        upsampler,
        (BicubicUpsampler, SRCNNUpsampler, IMDNUpsampler, SAFMNUpsampler),
    ):
        standard_compute_loss = cast(
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float]]],
            upsampler.compute_loss,
        )
        return standard_compute_loss(S_pred, target)
    raise TypeError(f"Unsupported upsampler type '{type(upsampler).__name__}' for auxiliary loss.")


def _extract_original_lam_latent(latent_x: torch.Tensor) -> torch.Tensor:
    """
    Extract the latent tensor using the original trainer convention.

    Parameters
    ----------
    latent_x : torch.Tensor
        Latent map returned by the LAM wrapper.

    Returns
    -------
    torch.Tensor
        Latent tensor passed into the original-method loss.
    """
    if latent_x.ndim == 1:
        return torch.abs(latent_x)
    return torch.abs(latent_x[0])


def compute_lam_loss_terms(
    model: EndToEndLossModel,
    *,
    S_low: torch.Tensor,
    S_high: torch.Tensor,
    lam_loss: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Run the wrapper forward pass and compute the original-method LAM losses.

    Parameters
    ----------
    model : EndToEndLossModel
        Wrapper model.
    S_low : torch.Tensor
        Low-resolution complex CSM batch.
    S_high : torch.Tensor
        High-resolution complex CSM batch.
    lam_loss : nn.Module
        Original-method LAM loss.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
        Upsampled CSM, reconstructed CSM, total LAM loss, reconstruction term,
        TV term, and the scalar TV weight.
    """
    S_pred, S_out, latent_x = model.forward_components(S_low)
    original_latent = _extract_original_lam_latent(latent_x)
    lam_target = S_high.to(dtype=S_out.dtype)
    lam_loss_fn = cast(
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        lam_loss,
    )
    lam_total, lam_reconstruction, lam_tv = lam_loss_fn(S_out, lam_target, original_latent)
    lam_l1_weight = getattr(lam_loss, "l1_weight", None)
    if not isinstance(lam_l1_weight, (float, int)):
        raise TypeError("LAM loss module must expose numeric l1_weight.")

    return (
        S_pred,
        S_out,
        lam_total,
        lam_reconstruction,
        lam_tv,
        float(lam_l1_weight),
    )
