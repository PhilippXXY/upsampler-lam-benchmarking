"""GAN-specific helpers for end-to-end wrapper training."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn

from lam_min.model.GANLAM import GANLAM
from training.end_to_end_losses import compute_auxiliary_loss, compute_lam_loss_terms

OptimiserInput: TypeAlias = torch.optim.Optimizer | dict[str, torch.optim.Optimizer]


def _gan_generator_and_lam_parameters(model: GANLAM) -> list[nn.Parameter]:  # type: ignore[no-any-unimported]
    """
    Return the trainable generator and LAM parameters for GAN end-to-end training.

    Parameters
    ----------
    model : GANLAM
        GAN wrapper model.

    Returns
    -------
    list[nn.Parameter]
        Trainable parameters belonging to the GAN generator and the LAM branch.
    """
    parameters = list(model.upsampler.generator.parameters()) + list(model.lam.parameters())
    return [parameter for parameter in parameters if parameter.requires_grad]


def _extract_gan_optimisers(
    optimiser: OptimiserInput,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Extract generator/LAM and discriminator optimisers for GAN end-to-end training.

    Parameters
    ----------
    optimiser : OptimiserInput
        End-to-end optimiser input.

    Returns
    -------
    tuple[torch.optim.Optimizer, torch.optim.Optimizer]
        Generator/LAM optimiser and discriminator optimiser.
    """
    if not isinstance(optimiser, dict):
        raise ValueError(
            "GANLAM end-to-end training requires an optimiser dict with keys "
            "'generator' and 'discriminator'."
        )

    generator_optimiser = optimiser.get("generator")
    discriminator_optimiser = optimiser.get("discriminator")
    if generator_optimiser is None or discriminator_optimiser is None:
        raise ValueError(
            "GANLAM end-to-end training requires optimiser dict keys "
            "'generator' and 'discriminator'."
        )
    return generator_optimiser, discriminator_optimiser


def build_gan_optimiser(  # type: ignore[no-any-unimported]
    model: GANLAM,
    *,
    learning_rate: float,
    weight_decay: float,
) -> OptimiserInput | None:
    """
    Build the GAN-specific optimiser split for end-to-end training.

    Parameters
    ----------
    model : GANLAM
        GAN wrapper model.
    learning_rate : float
        Base stage learning rate.
    weight_decay : float
        Optimiser weight decay.

    Returns
    -------
    OptimiserInput | None
        Generator/LAM and discriminator optimiser dict, or ``None`` when there
        are no trainable parameters.
    """
    generator_parameters = _gan_generator_and_lam_parameters(model)
    discriminator_parameters = [
        parameter
        for parameter in model.upsampler.discriminator.parameters()
        if parameter.requires_grad
    ]
    if not generator_parameters and not discriminator_parameters:
        return None
    if not generator_parameters:
        raise ValueError("GANLAM end-to-end training requires trainable generator/LAM weights.")
    if not discriminator_parameters:
        raise ValueError("GANLAM end-to-end training requires trainable discriminator weights.")

    beta1 = float(model.upsampler.beta1)
    beta2 = float(model.upsampler.beta2)
    return {
        "generator": torch.optim.Adam(
            generator_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        ),
        "discriminator": torch.optim.Adam(
            discriminator_parameters,
            lr=learning_rate * float(model.upsampler.discriminator_lr_scale),
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        ),
    }


def run_gan_end_to_end_step(  # type: ignore[no-any-unimported]  # noqa: C901, PLR0913, PLR0915
    model: GANLAM,
    *,
    S_low: torch.Tensor,
    S_high: torch.Tensor,
    lam_loss: nn.Module,
    aux_enabled: bool,
    aux_weight_config: float,
    effective_aux_weight: float,
    aux_baseline_ratio: float,
    use_model_specific_aux_loss: bool,
    optimiser: OptimiserInput | None = None,
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    """
    Run one GAN end-to-end train or validation step.

    Parameters
    ----------
    model : GANLAM
        GAN wrapper model.
    S_low : torch.Tensor
        Low-resolution complex CSM batch.
    S_high : torch.Tensor
        High-resolution complex CSM batch.
    lam_loss : nn.Module
        Original-method LAM loss.
    aux_enabled : bool
        Whether the extra auxiliary reconstruction loss is enabled.
    aux_weight_config : float
        Auxiliary weight configured in the training file.
    effective_aux_weight : float
        Effective auxiliary multiplier applied during optimisation.
    aux_baseline_ratio : float
        Fixed baseline ratio used to derive ``effective_aux_weight``.
    use_model_specific_aux_loss : bool
        Whether to reuse model-specific auxiliary losses.
    optimiser : OptimiserInput | None, optional
        Optimiser input for training mode. ``None`` means validation mode.
    grad_clip_norm : float, optional
        Maximum gradient norm. Non-positive values disable clipping.

    Returns
    -------
    dict[str, float]
        Scalar GAN and LAM loss statistics for the step.
    """
    upsampler = model.upsampler
    discriminator_optimiser: torch.optim.Optimizer | None = None
    generator_optimiser: torch.optim.Optimizer | None = None
    if optimiser is not None:
        generator_optimiser, discriminator_optimiser = _extract_gan_optimisers(optimiser)

    fake_csm_discriminator = upsampler._forward_no_metrics(S_low)
    real_ri = upsampler._reshape_csm_to_ri(
        S_high.to(dtype=fake_csm_discriminator.dtype),
        channels=upsampler.out_channels,
    ).to(torch.float32)
    fake_ri = upsampler._reshape_csm_to_ri(
        fake_csm_discriminator,
        channels=upsampler.out_channels,
    ).to(torch.float32)
    label_real = torch.ones((real_ri.shape[0], 1), dtype=real_ri.dtype, device=real_ri.device)
    label_fake = torch.zeros((real_ri.shape[0], 1), dtype=real_ri.dtype, device=real_ri.device)

    if discriminator_optimiser is not None:
        discriminator_optimiser.zero_grad(set_to_none=True)

    d_real = upsampler.discriminator(real_ri)
    d_fake = upsampler.discriminator(fake_ri.detach())
    loss_d_real = upsampler.adversarial_criterion(d_real, label_real)
    loss_d_fake = upsampler.adversarial_criterion(d_fake, label_fake)
    loss_d = loss_d_real + loss_d_fake

    if discriminator_optimiser is not None:
        loss_d.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                upsampler.discriminator.parameters(),
                max_norm=grad_clip_norm,
            )
        discriminator_optimiser.step()
        upsampler._step_counter += 1

    should_update_generator = (
        discriminator_optimiser is not None
        and (upsampler._step_counter % upsampler.critic_iters) == 0
    )

    if generator_optimiser is not None:
        generator_optimiser.zero_grad(set_to_none=True)

    (
        S_pred,
        _S_out,
        lam_total,
        lam_reconstruction,
        lam_tv,
        lam_l1_weight,
    ) = compute_lam_loss_terms(
        model,
        S_low=S_low,
        S_high=S_high,
        lam_loss=lam_loss,
    )

    gan_target = S_high.to(dtype=S_pred.dtype)
    content_raw, _content_stats = upsampler._compute_reconstruction_loss(S_pred, gan_target)
    fake_ri_for_g = upsampler._reshape_csm_to_ri(
        S_pred,
        channels=upsampler.out_channels,
    ).to(torch.float32)
    adv_loss = upsampler.adversarial_criterion(upsampler.discriminator(fake_ri_for_g), label_real)

    loss_g_content = upsampler.content_weight * content_raw
    loss_g_adv = upsampler.adversarial_weight * adv_loss
    loss_g_total = loss_g_content + loss_g_adv

    aux_raw = lam_total.new_zeros(())
    aux_stats: dict[str, float] = {}
    if aux_enabled:
        aux_raw, aux_stats = compute_auxiliary_loss(
            model,
            S_pred=S_pred,
            S_high=S_high,
            use_model_specific_aux_loss=use_model_specific_aux_loss,
        )
    applied_aux_weight = effective_aux_weight if aux_enabled else 0.0
    applied_aux_ratio = aux_baseline_ratio if aux_enabled else 0.0
    aux_weighted = applied_aux_weight * aux_raw
    total = lam_total + loss_g_total + aux_weighted

    if generator_optimiser is not None:
        total.backward()
        if not should_update_generator:
            for parameter in upsampler.generator.parameters():
                parameter.grad = None
        generator_and_lam_parameters = _gan_generator_and_lam_parameters(model)
        if grad_clip_norm > 0.0 and generator_and_lam_parameters:
            torch.nn.utils.clip_grad_norm_(
                generator_and_lam_parameters,
                max_norm=grad_clip_norm,
            )
        generator_optimiser.step()

    stats: dict[str, float] = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_lam_total": float(lam_total.detach().cpu().item()),
        "loss_lam_reconstruction": float(lam_reconstruction.detach().cpu().item()),
        "loss_lam_tv": float(lam_tv.detach().cpu().item()),
        "loss_lam_tv_weight": float(lam_l1_weight),
        "loss_g_total": float(loss_g_total.detach().cpu().item()),
        "loss_g_content": float(loss_g_content.detach().cpu().item()),
        "loss_g_adv": float(loss_g_adv.detach().cpu().item()),
        "loss_d": float(loss_d.detach().cpu().item()),
        "loss_d_real": float(loss_d_real.detach().cpu().item()),
        "loss_d_fake": float(loss_d_fake.detach().cpu().item()),
        "loss_aux_raw": float(aux_raw.detach().cpu().item()),
        "loss_aux": float(aux_weighted.detach().cpu().item()),
        "loss_aux_weight_config": float(aux_weight_config),
        "loss_aux_baseline_ratio": float(applied_aux_ratio),
        "loss_aux_weight": float(applied_aux_weight),
        "generator_updated": 1.0 if should_update_generator else 0.0,
        "critic_iters": float(upsampler.critic_iters),
        "num_frames": float(S_low.shape[0]),
        "num_bands": float(S_low.shape[1]),
    }
    for key, value in aux_stats.items():
        stats[f"aux_{key}"] = float(value)
    return stats
