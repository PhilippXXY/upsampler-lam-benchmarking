"""End-to-end upsampler + LAM training helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lam_min.model.AINNLAM import AINNLAM
from lam_min.model.BicubicLAM import BicubicLAM
from lam_min.model.GANLAM import GANLAM
from lam_min.model.IMDNLAM import IMDNLAM
from lam_min.model.SAFMNLAM import SAFMNLAM
from lam_min.model.SRCNNLAM import SRCNNLAM
from lam_min.model.UpLAM import UpLAM
from lam_min.training_loss import MSETVLoss
from lam_min.util.utils import (
    load_ainn_lam_state,
    load_bicubic_lam_state,
    load_gan_lam_state,
    load_imdn_lam_state,
    load_safmn_lam_state,
    load_srcnn_lam_state,
    load_uplam_lam_state,
)
from training.end_to_end_gan import (
    OptimiserInput,
    build_gan_optimiser,
    run_gan_end_to_end_step,
)
from training.end_to_end_losses import compute_auxiliary_loss, compute_lam_loss_terms
from utils.training_utils import load_state_dict_checkpoint

SUPPORTED_END_TO_END_MODELS = (
    "UpLAM",
    "BicubicLAM",
    "SRCNNLAM",
    "IMDNLAM",
    "SAFMNLAM",
    "GANLAM",
    "AINNLAM",
)
ORIGINAL_LAM_METHOD = "original_msetv"


@dataclass(frozen=True)
class AuxWeightCalibration:
    """Fixed auxiliary-weight calibration derived from one random-init probe."""

    initial_lam_total: float
    initial_aux_raw: float
    baseline_ratio: float
    effective_aux_weight: float


class EndToEndModel(Protocol):
    """Protocol for end-to-end wrapper models used in training."""

    upsampler: nn.Module
    lam: nn.Module

    def forward_components(
        self, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return upsampled CSM, reconstructed CSM, and latent map."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return trainable parameters."""

    def train(self, mode: bool = True) -> nn.Module:
        """Switch the wrapper between train/eval mode."""


def build_end_to_end_model(  # noqa: PLR0911
    model_cfg: dict[str, Any],
    *,
    num_bands: int,
) -> nn.Module:
    """
    Build an end-to-end wrapper model.

    Parameters
    ----------
    model_cfg : dict[str, Any]
        Model configuration dictionary.
    num_bands : int
        Number of frequency bands in the training data.

    Returns
    -------
    nn.Module
        Instantiated wrapper model.
    """
    model_name = str(model_cfg["name"])
    common_kwargs = {
        "num_bands": num_bands,
        "in_channels": int(model_cfg.get("in_channels", 4)),
        "out_channels": int(model_cfg.get("out_channels", 32)),
        "freeze_lam": False,
    }

    if model_name == "BicubicLAM":
        return cast(nn.Module, BicubicLAM(**common_kwargs))
    if model_name == "UpLAM":
        in_channels = int(model_cfg.get("in_channels", 4))
        out_channels = int(model_cfg.get("out_channels", 32))
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"UpLAM requires positive in_channels/out_channels, "
                f"got {in_channels}/{out_channels}."
            )
        scale_factor = out_channels // in_channels
        if scale_factor * in_channels != out_channels:
            raise ValueError(
                "UpLAM requires out_channels to be an integer multiple of in_channels. "
                f"Got {in_channels} -> {out_channels}."
            )
        return cast(
            nn.Module,
            UpLAM(
                num_bands=num_bands,
                base_filter=int(model_cfg.get("base_filter", 32)),
                feat=int(model_cfg.get("feature_channels", 128)),
                num_stages=int(model_cfg.get("num_stages", 10)),
                scale_factor=scale_factor,
                freeze_lam=False,
            ),
        )
    if model_name == "SRCNNLAM":
        return cast(
            nn.Module,
            SRCNNLAM(
                **common_kwargs,
                feature_channels=int(model_cfg.get("feature_channels", 64)),
                mapping_channels=int(model_cfg.get("mapping_channels", 32)),
                loss_name=str(model_cfg.get("loss_name", "l1")),
            ),
        )
    if model_name == "IMDNLAM":
        return cast(
            nn.Module,
            IMDNLAM(
                **common_kwargs,
                feature_channels=int(model_cfg.get("feature_channels", 64)),
                mapping_channels=int(model_cfg.get("mapping_channels", 32)),
                loss_name=str(model_cfg.get("loss_name", "l1")),
            ),
        )
    if model_name == "SAFMNLAM":
        return cast(
            nn.Module,
            SAFMNLAM(
                **common_kwargs,
                feature_channels=int(model_cfg.get("feature_channels", 36)),
                n_blocks=int(model_cfg.get("n_blocks", 8)),
                ffn_scale=float(model_cfg.get("ffn_scale", 2.0)),
                n_levels=int(model_cfg.get("n_levels", 4)),
                loss_name=str(model_cfg.get("loss_name", "l1")),
                fft_loss_weight=float(model_cfg.get("fft_loss_weight", 0.05)),
            ),
        )
    if model_name == "GANLAM":
        return cast(
            nn.Module,
            GANLAM(
                **common_kwargs,
                feature_channels=int(model_cfg.get("feature_channels", 128)),
                n_residual_blocks=int(
                    model_cfg.get("n_residual_blocks", model_cfg.get("n_blocks", 8))
                ),
                loss_name=str(model_cfg.get("loss_name", "l1")),
                adversarial_weight=float(model_cfg.get("adversarial_weight", 0.01)),
                content_weight=float(model_cfg.get("content_weight", 1.0)),
                critic_iters=int(model_cfg.get("critic_iters", 1)),
                discriminator_lr_scale=float(model_cfg.get("discriminator_lr_scale", 1.0)),
                beta1_adam=float(model_cfg.get("beta1_adam", 0.9)),
                beta2_adam=float(model_cfg.get("beta2_adam", 0.999)),
            ),
        )
    if model_name == "AINNLAM":
        return cast(
            nn.Module,
            AINNLAM(
                **common_kwargs,
                hidden_channels=int(model_cfg.get("hidden_channels", 64)),
                latent_channels=int(model_cfg.get("latent_channels", 64)),
                low_channel_indices=tuple(
                    int(index) for index in model_cfg.get("low_channel_indices", [5, 9, 21, 25])
                ),
                loss_name=str(model_cfg.get("loss_name", "mse")),
                pde_loss_weight=float(model_cfg.get("pde_loss_weight", 0.01)),
                pde_freq_min_hz=float(model_cfg.get("pde_freq_min_hz", 100.0)),
                pde_freq_max_hz=float(model_cfg.get("pde_freq_max_hz", 4000.0)),
                sound_speed=float(model_cfg.get("sound_speed", 340.0)),
            ),
        )

    raise ValueError(
        f"Unsupported end-to-end model '{model_name}'. "
        f"Use one of: {', '.join(SUPPORTED_END_TO_END_MODELS)}"
    )


def build_lam_loss(loss_cfg: dict[str, Any], *, device: torch.device) -> nn.Module:
    """
    Build the original-method LAM loss.

    Parameters
    ----------
    loss_cfg : dict[str, Any]
        Loss configuration dictionary.
    device : torch.device
        Training device, retained for parity with the original interface.

    Returns
    -------
    nn.Module
        Original-method LAM loss module.
    """
    lam_method = str(loss_cfg.get("lam_method", ORIGINAL_LAM_METHOD)).strip().lower()
    if lam_method != ORIGINAL_LAM_METHOD:
        raise ValueError(
            f"Unsupported lam_method '{lam_method}'. Only '{ORIGINAL_LAM_METHOD}' is supported."
        )

    return cast(
        nn.Module,
        MSETVLoss(
            l1_weight=float(loss_cfg.get("lam_tv_weight", 1.0e-5)),
            device=str(device),
        ),
    )


def initialise_model(  # noqa: C901
    model: nn.Module,
    *,
    model_name: str,
    initialisation_cfg: dict[str, Any],
    resume_strict: bool,
    device: torch.device,
) -> None:
    """
    Initialise a wrapper model from separate or combined checkpoints.

    Parameters
    ----------
    model : nn.Module
        Wrapper model to initialise.
    model_name : str
        Wrapper model name.
    initialisation_cfg : dict[str, Any]
        Initialisation configuration dictionary.
    resume_strict : bool
        Strictness used when resuming a full wrapper checkpoint.
    device : torch.device
        Device used for checkpoint loading.
    """
    resume_checkpoint = str(initialisation_cfg.get("resume_checkpoint", "")).strip()
    if resume_checkpoint:
        load_state_dict_checkpoint(model=model, path=Path(resume_checkpoint), strict=resume_strict)
        logging.info("Resumed combined end-to-end checkpoint for %s.", model_name)
        return

    upsampler_checkpoint = str(initialisation_cfg.get("upsampler_checkpoint", "")).strip()
    lam_checkpoint = str(initialisation_cfg.get("lam_checkpoint", "")).strip() or None

    if model_name not in {"BicubicLAM", "UpLAM"} and not upsampler_checkpoint:
        raise ValueError(
            f"initialisation.upsampler_checkpoint is required for {model_name} when not resuming."
        )
    if model_name == "UpLAM" and not upsampler_checkpoint and lam_checkpoint is None:
        raise ValueError(
            "UpLAM requires either initialisation.upsampler_checkpoint or initialisation."
            "lam_checkpoint when not resuming."
        )

    if model_name == "BicubicLAM":
        load_bicubic_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    elif model_name == "UpLAM":
        load_uplam_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    elif model_name == "SRCNNLAM":
        load_srcnn_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    elif model_name == "IMDNLAM":
        load_imdn_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    elif model_name == "SAFMNLAM":
        load_safmn_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    elif model_name == "GANLAM":
        load_gan_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    elif model_name == "AINNLAM":
        load_ainn_lam_state(
            model,
            upsampler_checkpoint,
            device,
            lam_checkpoint=lam_checkpoint,
        )
    else:
        raise ValueError(f"Unsupported end-to-end model '{model_name}'.")

    logging.info(
        "Initialised %s from upsampler checkpoint '%s' and LAM checkpoint '%s'.",
        model_name,
        upsampler_checkpoint or "<none>",
        lam_checkpoint or "<none>",
    )


def build_optimiser(  # type: ignore[no-any-unimported]
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    freeze_upsampler: bool = False,
) -> OptimiserInput | None:
    """
    Build the optimiser for end-to-end training.

    Parameters
    ----------
    model : nn.Module
        Wrapper model.
    learning_rate : float
        Optimiser learning rate.
    weight_decay : float
        Optimiser weight decay.

    Returns
    -------
    OptimiserInput | None
        AdamW optimiser for the standard wrappers, a generator/discriminator
        optimiser dict for ``GANLAM``, or ``None`` when there are no trainable
        parameters.
    """
    if isinstance(model, GANLAM) and not freeze_upsampler:
        return build_gan_optimiser(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    trainable_params = (
        [parameter for parameter in model.lam.parameters() if parameter.requires_grad]  # type: ignore[union-attr]
        if freeze_upsampler
        else [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    if not trainable_params:
        return None
    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def freeze_upsampler_parameters(model: EndToEndModel) -> None:
    """
    Freeze the wrapper upsampler branch in-place.

    Parameters
    ----------
    model : EndToEndModel
        Wrapper model whose upsampler branch should be fixed.
    """
    for parameter in model.upsampler.parameters():
        parameter.requires_grad = False


def calibrate_aux_weight(  # noqa: PLR0913
    *,
    model_cfg: dict[str, Any],
    num_bands: int,
    device: torch.device,
    S_low: torch.Tensor,
    S_high: torch.Tensor,
    lam_loss: nn.Module,
    aux_weight: float,
    use_model_specific_aux_loss: bool,
    random_init_generator: torch.Generator | None = None,
) -> AuxWeightCalibration:
    """
    Calibrate the effective auxiliary weight from one random-init probe pass.

    Parameters
    ----------
    model_cfg : dict[str, Any]
        Wrapper architecture configuration.
    num_bands : int
        Number of frequency bands used to build the wrapper.
    device : torch.device
        Device used for the temporary calibration model.
    S_low : torch.Tensor
        Probe low-resolution chunk.
    S_high : torch.Tensor
        Probe high-resolution chunk.
    lam_loss : nn.Module
        LAM loss used for the real training run.
    aux_weight : float
        Configured auxiliary weight before calibration.
    use_model_specific_aux_loss : bool
        Whether to reuse model-specific auxiliary losses.
    random_init_generator : torch.Generator | None, optional
        Optional torch RNG state used to initialise the temporary calibration
        model without perturbing the global training RNG.

    Returns
    -------
    AuxWeightCalibration
        Baseline losses and the derived effective auxiliary weight.
    """
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_index = 0 if device.index is None else int(device.index)
        cuda_devices = [cuda_index]
    rng_context = (
        torch.random.fork_rng(devices=cuda_devices)
        if random_init_generator is not None
        else nullcontext()
    )
    with rng_context:
        if random_init_generator is not None:
            torch.set_rng_state(random_init_generator.get_state())
        calibration_model = build_end_to_end_model(model_cfg=model_cfg, num_bands=num_bands).to(
            device
        )
        calibration_model.train()

        with torch.no_grad():
            S_low_probe = S_low.to(device)
            S_high_probe = S_high.to(device)
            S_pred, _S_out, lam_total, _lam_reconstruction, _lam_tv, _lam_l1_weight = (
                compute_lam_loss_terms(
                    calibration_model,
                    S_low=S_low_probe,
                    S_high=S_high_probe,
                    lam_loss=lam_loss,
                )
            )
            aux_raw, _aux_stats = compute_auxiliary_loss(
                calibration_model,
                S_pred=S_pred,
                S_high=S_high_probe,
                use_model_specific_aux_loss=use_model_specific_aux_loss,
            )

    initial_lam_total = float(lam_total.detach().cpu().item())
    initial_aux_raw = float(aux_raw.detach().cpu().item())
    baseline_ratio = initial_lam_total / initial_aux_raw if initial_aux_raw != 0.0 else float("inf")
    effective_aux_weight = float(aux_weight) * baseline_ratio

    if initial_aux_raw <= 0.0:
        raise ValueError(
            "AuxEn calibration requires a strictly positive initial auxiliary loss, "
            f"got {initial_aux_raw:.6f}."
        )
    if not all(
        torch.isfinite(torch.tensor(value, dtype=torch.float64))
        for value in (initial_lam_total, initial_aux_raw, baseline_ratio, effective_aux_weight)
    ):
        raise ValueError(
            "AuxEn calibration produced non-finite values "
            f"(lam={initial_lam_total}, aux={initial_aux_raw}, "
            f"ratio={baseline_ratio}, weight={effective_aux_weight})."
        )

    return AuxWeightCalibration(
        initial_lam_total=initial_lam_total,
        initial_aux_raw=initial_aux_raw,
        baseline_ratio=baseline_ratio,
        effective_aux_weight=effective_aux_weight,
    )


def _run_lam_with_frozen_upsampler(
    model: EndToEndModel,
    *,
    S_low: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Skip autograd through the frozen upsampler branch.

    When only the LAM branch is trainable, the upsampler output can be treated
    as a constant input to LAM. This avoids building a large backward graph for
    the wrapper's frozen front-end.
    """
    with torch.no_grad():
        if isinstance(model, UpLAM):
            x_rel, x_imag = model._prepare_cdbpn_input(S_low)
            S_pred = model.upsampler(x_rel, x_imag, collect_metrics=False)
        else:
            S_pred = model.upsampler(S_low, collect_metrics=False)
        S_pred = S_pred.to(dtype=model.lam.D.dtype)

    S_pred = S_pred.detach()
    S_out, latent_x, _ = model.lam(S_pred, collect_metrics=False)
    return S_pred, S_out, latent_x


def run_end_to_end_step(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    model: EndToEndModel,
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
    freeze_upsampler: bool = False,
) -> dict[str, float]:
    """
    Run a single end-to-end train or validation step.

    Parameters
    ----------
    model : EndToEndModel
        Wrapper model.
    S_low : torch.Tensor
        Low-resolution complex CSM batch.
    S_high : torch.Tensor
        High-resolution complex CSM batch.
    lam_loss : nn.Module
        Original-method LAM loss.
    aux_enabled : bool
        Whether the auxiliary reconstruction loss is enabled.
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
        Scalar loss statistics for the step.
    """
    if isinstance(model, GANLAM) and not freeze_upsampler:
        return run_gan_end_to_end_step(  # type: ignore[no-any-return]
            model,
            S_low=S_low,
            S_high=S_high,
            lam_loss=lam_loss,
            aux_enabled=aux_enabled,
            aux_weight_config=aux_weight_config,
            effective_aux_weight=effective_aux_weight,
            aux_baseline_ratio=aux_baseline_ratio,
            use_model_specific_aux_loss=use_model_specific_aux_loss,
            optimiser=optimiser,
            grad_clip_norm=grad_clip_norm,
        )

    if optimiser is not None:
        optimiser.zero_grad(set_to_none=True)

    if freeze_upsampler:
        S_pred, S_out, latent_x = _run_lam_with_frozen_upsampler(model, S_low=S_low)
        original_latent = torch.abs(latent_x) if latent_x.ndim == 1 else torch.abs(latent_x[0])
        lam_target = S_high.to(dtype=S_out.dtype)
        lam_total, lam_reconstruction, lam_tv = cast(
            Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ],
            lam_loss,
        )(S_out, lam_target, original_latent)
        lam_l1_weight = getattr(lam_loss, "l1_weight", None)
        if not isinstance(lam_l1_weight, (float, int)):
            raise TypeError("LAM loss module must expose numeric l1_weight.")
    else:
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
    total = lam_total + aux_weighted

    if optimiser is not None:
        total.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimiser.step()

    stats: dict[str, float] = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_lam_total": float(lam_total.detach().cpu().item()),
        "loss_lam_reconstruction": float(lam_reconstruction.detach().cpu().item()),
        "loss_lam_tv": float(lam_tv.detach().cpu().item()),
        "loss_lam_tv_weight": float(lam_l1_weight),
        "loss_aux_raw": float(aux_raw.detach().cpu().item()),
        "loss_aux": float(aux_weighted.detach().cpu().item()),
        "loss_aux_weight_config": float(aux_weight_config),
        "loss_aux_baseline_ratio": float(applied_aux_ratio),
        "loss_aux_weight": float(applied_aux_weight),
        "num_frames": float(S_low.shape[0]),
        "num_bands": float(S_low.shape[1]),
    }
    for key, value in aux_stats.items():
        stats[f"aux_{key}"] = float(value)
    return stats


def run_epoch(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    model: EndToEndModel,
    *,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    frame_batch_size: int,
    lam_loss: nn.Module,
    aux_enabled: bool,
    aux_weight_config: float,
    effective_aux_weight: float,
    aux_baseline_ratio: float,
    use_model_specific_aux_loss: bool,
    optimiser: OptimiserInput | None = None,
    grad_clip_norm: float = 0.0,
    log_every_files: int = 0,
    freeze_upsampler: bool = False,
) -> dict[str, float]:
    """
    Run one training or validation epoch for the end-to-end model.

    Parameters
    ----------
    model : EndToEndModel
        Wrapper model.
    loader : DataLoader[dict[str, Any]]
        File-level DataLoader.
    device : torch.device
        Training device.
    frame_batch_size : int
        Number of frames per optimiser step.
    lam_loss : nn.Module
        Original-method LAM loss.
    aux_enabled : bool
        Whether the auxiliary loss is enabled.
    aux_weight_config : float
        Auxiliary weight configured in the training file.
    effective_aux_weight : float
        Effective auxiliary multiplier applied during optimisation.
    aux_baseline_ratio : float
        Fixed baseline ratio used to derive ``effective_aux_weight``.
    use_model_specific_aux_loss : bool
        Whether to reuse each upsampler's reconstruction helper.
    optimiser : OptimiserInput | None, optional
        Optimiser input for training mode. ``None`` means validation mode.
    grad_clip_norm : float, optional
        Maximum gradient norm.
    log_every_files : int, optional
        File-level logging cadence.

    Returns
    -------
    dict[str, float]
        Average epoch statistics.
    """
    is_train = optimiser is not None
    model.train(is_train)
    if freeze_upsampler:
        # Keep the upsampler branch in eval mode so running stats remain fixed.
        model.upsampler.eval()

    loss_sums: dict[str, float] = {"loss_total": 0.0}
    chunk_count = 0

    iterator = tqdm(loader, ncols=100, desc="Train" if is_train else "Val")
    for file_index, sample in enumerate(iterator, start=1):
        S_low = sample["S_low"].to(device)
        S_high = sample["S_high"].to(device)
        n_frames = int(S_low.shape[0])

        for start in range(0, n_frames, frame_batch_size):
            end = min(start + frame_batch_size, n_frames)
            x = S_low[start:end]
            y = S_high[start:end]
            stats = run_end_to_end_step(
                model,
                S_low=x,
                S_high=y,
                lam_loss=lam_loss,
                aux_enabled=aux_enabled,
                aux_weight_config=aux_weight_config,
                effective_aux_weight=effective_aux_weight,
                aux_baseline_ratio=aux_baseline_ratio,
                use_model_specific_aux_loss=use_model_specific_aux_loss,
                optimiser=optimiser,
                grad_clip_norm=grad_clip_norm,
                freeze_upsampler=freeze_upsampler,
            )
            for key, value in stats.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
            chunk_count += 1

        if log_every_files > 0 and file_index % log_every_files == 0:
            logging.info(
                "[%s] file %d/%d | avg loss %.6f",
                "train" if is_train else "val",
                file_index,
                len(loader),
                loss_sums["loss_total"] / max(chunk_count, 1),
            )

    denom = max(chunk_count, 1)
    out = {key: value / denom for key, value in loss_sums.items()}
    out["num_chunks"] = float(chunk_count)
    return out
