"""Train end-to-end upsampler + LAM wrapper models."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset

from data.variable_channels import CANONICAL_CHANNELS
from training.end_to_end import (
    build_end_to_end_model,
    build_lam_loss,
    build_optimiser,
    calibrate_aux_weight,
    freeze_upsampler_parameters,
    initialise_model,
    run_epoch,
)
from utils.model_variants import canonical_e2e_checkpoint_prefix, resolve_e2e_variant_kind
from utils.training_utils import (
    build_dataset_list,
    build_train_loader,
    configure_torch_multiprocessing,
    format_compact_loss_series,
    load_conf,
    normalise_stages,
    resolve_training_device,
    sanitise_name,
    save_state_dict_checkpoint,
    save_training_meta,
    setup_logging,
)
from utils.utils import seed_everything

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _clone_torch_generator(generator: torch.Generator) -> torch.Generator:
    """Clone a torch generator so probe work can reuse deterministic RNG state."""
    cloned = torch.Generator()
    cloned.set_state(generator.get_state())
    return cloned


def _load_probe_chunk(  # noqa: PLR0913
    *,
    train_datasets: list[Any],
    device: torch.device,
    sampling: str,
    frame_batch_size: int,
    generator: torch.Generator,
    variable_channel_counts: tuple[int, ...] | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Load one probe chunk for auxiliary-weight calibration without touching the real loader.

    Parameters
    ----------
    train_datasets: list[Any]
        The list of training datasets to load the probe chunk from.
    device: torch.device
        The device to load the probe chunk onto.
    sampling: str
        The sampling strategy to use for loading the probe chunk.
    frame_batch_size: int
        The number of frames to load in the probe chunk.
    generator: torch.Generator
        The torch generator to use for deterministic sampling of the probe chunk.
    variable_channel_counts : tuple[int, ...] | None, optional
        Variable microphone counts for the probe.
    seed : int, optional
        General training seed, also used for variable subsets.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Probe CSMs and optional observed microphone indices.
    """
    probe_loader = build_train_loader(
        datasets=train_datasets,
        num_workers=0,
        device=device,
        sampling=sampling,
        generator=generator,
        variable_channel_counts=variable_channel_counts,
        seed=seed,
    )
    probe_sample = next(
        sample
        for sample in probe_loader
        if variable_channel_counts is None
        or int(sample["input_channel_count"]) < CANONICAL_CHANNELS
    )
    S_low = probe_sample["S_low"]
    S_high = probe_sample["S_high"]
    if int(S_low.shape[0]) <= 0:
        raise ValueError(
            "AuxEn calibration requires at least one training frame in the probe file."
        )

    end = min(frame_batch_size, int(S_low.shape[0]))
    observed_indices = probe_sample.get("observed_channel_indices")
    return (
        S_low[:end].to(device),
        S_high[:end].to(device),
        observed_indices.to(device) if torch.is_tensor(observed_indices) else None,
    )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """
    Run end-to-end wrapper training from configuration.

    The trainer mirrors the staged file-level flow of the existing upsampler
    trainer, but optimises the wrapper model jointly with the original-method
    LAM loss and an optional auxiliary reconstruction term.
    """
    parser = argparse.ArgumentParser("End-to-end upsampler + LAM training")
    parser.add_argument("--config", type=str, default="config/train_end_to_end.yaml")
    parser.add_argument("--device", type=str, default="", help="cpu | mps | cuda")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional combined wrapper checkpoint used to initialise model weights for training.",
    )
    args = parser.parse_args()

    config = load_conf(Path(args.config))
    training_cfg = config["training"]
    model_cfg = config["model"]
    initialisation_cfg = dict(config.get("initialisation", {}))
    loss_cfg = config.get("loss", {})

    output_root = Path(training_cfg.get("output_root", "output/training_end_to_end"))
    setup_logging(
        output_root.joinpath("logs"),
        log_stem="train_end_to_end",
        timestamp=timestamp,
    )

    training_seed = int(training_cfg.get("seed", 42))
    train_generator = seed_everything(training_seed)

    requested_device = args.device if args.device else str(training_cfg.get("device", "cpu"))
    device = resolve_training_device(
        requested_device,
        mps_fallback_reason=(
            "End-to-end training uses LAM, which is configured for CPU/CUDA in this repo."
        ),
    )
    logging.info("Using device: %s", device)

    num_bands = int(config["data"].get("nbands", 9))
    model_name = str(model_cfg["name"])
    model_cfg = dict(model_cfg)
    model_cfg["low_channel_indices"] = config["data"].get("low_channel_indices", [5, 9, 21, 25])
    model = build_end_to_end_model(model_cfg=model_cfg, num_bands=num_bands).to(device)
    variable_counts = (
        tuple(
            int(count)
            for count in model_cfg.get("variable_input_channel_counts", [4, 8, 16, 24, 32])
        )
        if model_name == "VariableSRCNNLAM"
        else None
    )
    if variable_counts is not None:
        eigenscape_cfg = config["data"].get("eigenscape", {})
        if (
            eigenscape_cfg.get("expected_channels", CANONICAL_CHANNELS) != CANONICAL_CHANNELS
            or int(eigenscape_cfg.get("target_high_channels", CANONICAL_CHANNELS))
            != CANONICAL_CHANNELS
            or bool(eigenscape_cfg.get("allow_channel_fallback", False))
        ):
            raise ValueError("VariableSRCNNLAM requires strict 32-channel EigenScape sources")

    resume_checkpoint = (
        args.resume_checkpoint.strip()
        if args.resume_checkpoint.strip()
        else str(training_cfg.get("resume_from_checkpoint", "")).strip()
    )
    if resume_checkpoint:
        initialisation_cfg["resume_checkpoint"] = resume_checkpoint

    initialise_model(
        model,
        model_name=model_name,
        initialisation_cfg=initialisation_cfg,
        resume_strict=bool(initialisation_cfg.get("resume_strict", True)),
        device=device,
    )

    lam_loss = build_lam_loss(loss_cfg=loss_cfg, device=device)

    num_workers = int(training_cfg.get("num_workers", 0))
    frame_batch_size = int(training_cfg.get("frame_batch_size", 64))
    grad_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0))
    log_every_files = int(training_cfg.get("log_every_files", 0))
    sharing_strategy = configure_torch_multiprocessing(num_workers)
    if sharing_strategy is not None:
        logging.info("Using torch multiprocessing sharing strategy: %s", sharing_strategy)

    aux_enabled = bool(loss_cfg.get("aux_enabled", True))
    aux_weight = float(loss_cfg.get("aux_weight", 0.25))
    use_model_specific_aux_loss = bool(loss_cfg.get("use_model_specific_aux_loss", True))
    freeze_upsampler = bool(training_cfg.get("freeze_upsampler", False))
    aux_calibration_enabled = aux_enabled and not freeze_upsampler
    effective_aux_weight = 0.0
    aux_baseline_ratio = 0.0
    aux_calibration_meta: dict[str, Any] = {
        "enabled": aux_calibration_enabled,
        "loss_aux_weight_config": aux_weight,
        "loss_aux_baseline_ratio": None,
        "loss_aux_weight": None,
        "initial_loss_lam_total": None,
        "initial_loss_aux_raw": None,
    }
    if freeze_upsampler and aux_enabled:
        raise ValueError(
            "training.freeze_upsampler=true requires loss.aux_enabled=false because "
            "the frozen-upsampler mode only updates the LAM branch."
        )
    if freeze_upsampler:
        freeze_upsampler_parameters(model)
    if not aux_calibration_enabled:
        aux_calibration_meta["loss_aux_baseline_ratio"] = 0.0
        aux_calibration_meta["loss_aux_weight"] = 0.0

    ckpt_dir = Path(training_cfg.get("checkpoint_dir", "src/lam_min/checkpoints/e2e"))
    ckpt_prefix_raw = str(training_cfg.get("checkpoint_prefix", "")).strip()
    variant_kind = resolve_e2e_variant_kind(
        model_name,
        aux_enabled=aux_enabled,
        freeze_upsampler=freeze_upsampler,
    )
    ckpt_prefix = ckpt_prefix_raw or canonical_e2e_checkpoint_prefix(
        model_name,
        aux_enabled,
        freeze_upsampler=freeze_upsampler,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Using end-to-end retained variant kind: %s", variant_kind)
    logging.info("Using checkpoint prefix: %s", ckpt_prefix)

    history: list[dict[str, Any]] = []
    stages = normalise_stages(training_cfg=training_cfg, data_cfg=config["data"])

    best_val_loss_overall = float("inf")
    best_stage_name = ""
    best_stage_epoch = 0

    for stage_index, stage in enumerate(stages, start=1):
        if not stage["enabled"]:
            logging.info("Skipping disabled stage: %s", stage["name"])
            continue

        stage_name = str(stage["name"])
        stage_tag = sanitise_name(stage_name)
        stage_epochs = int(stage["epochs"])
        if stage_epochs <= 0:
            logging.info("Skipping stage '%s' because epochs <= 0.", stage_name)
            continue

        logging.info("=== Stage %d/%d: %s ===", stage_index, len(stages), stage_name)
        logging.info(
            "Stage datasets: audiblelight=%s eigenscape=%s | train_sampling=%s",
            stage["datasets"]["audiblelight"],
            stage["datasets"]["eigenscape"],
            stage["train_sampling"],
        )

        optimiser = build_optimiser(
            model,
            learning_rate=float(stage["learning_rate"]),
            weight_decay=float(stage["weight_decay"]),
            freeze_upsampler=freeze_upsampler,
        )
        if optimiser is None:
            logging.info(
                "Model '%s' has no trainable parameters. "
                "Running loss-only stage without optimiser updates.",
                model_name,
            )

        train_datasets = build_dataset_list(
            config=config,
            split="train",
            enabled_overrides=stage["datasets"],
            max_files=int(stage["max_train_files"]),
        )
        val_datasets = build_dataset_list(
            config=config,
            split="val",
            enabled_overrides=stage["datasets"],
            max_files=int(stage["max_val_files"]),
        )

        train_dataset = (
            train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        )
        val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
        logging.info(
            "Stage train files: %d | val files: %d",
            len(train_dataset),
            len(val_dataset),
        )

        if aux_calibration_enabled and aux_calibration_meta["loss_aux_weight"] is None:
            probe_generator = _clone_torch_generator(train_generator)
            calibration_generator = _clone_torch_generator(train_generator)
            probe_low, probe_high, probe_indices = _load_probe_chunk(
                train_datasets=train_datasets,
                device=device,
                sampling=str(stage["train_sampling"]),
                frame_batch_size=frame_batch_size,
                generator=probe_generator,
                variable_channel_counts=variable_counts,
                seed=training_seed,
            )
            aux_calibration = calibrate_aux_weight(
                model_cfg=model_cfg,
                num_bands=num_bands,
                device=device,
                S_low=probe_low,
                S_high=probe_high,
                lam_loss=lam_loss,
                aux_weight=aux_weight,
                use_model_specific_aux_loss=use_model_specific_aux_loss,
                random_init_generator=calibration_generator,
                observed_channel_indices=probe_indices,
            )

            effective_aux_weight = aux_calibration.effective_aux_weight
            aux_baseline_ratio = aux_calibration.baseline_ratio
            aux_calibration_meta.update(
                {
                    "loss_aux_baseline_ratio": aux_baseline_ratio,
                    "loss_aux_weight": effective_aux_weight,
                    "initial_loss_lam_total": aux_calibration.initial_lam_total,
                    "initial_loss_aux_raw": aux_calibration.initial_aux_raw,
                }
            )
            logging.info(
                "AuxEn calibration | lam=%.6f | aux=%.6f | ratio=%.6f | "
                "config_weight=%.6f | effective_weight=%.6f",
                aux_calibration.initial_lam_total,
                aux_calibration.initial_aux_raw,
                aux_baseline_ratio,
                aux_weight,
                effective_aux_weight,
            )

        train_loader = build_train_loader(
            datasets=train_datasets,
            num_workers=num_workers,
            device=device,
            sampling=str(stage["train_sampling"]),
            generator=train_generator,
            variable_channel_counts=variable_counts,
            seed=training_seed,
        )
        val_loader = build_train_loader(
            datasets=val_datasets,
            num_workers=num_workers,
            device=device,
            sampling="proportional",
            variable_channel_counts=variable_counts,
            seed=training_seed,
            shuffle=False,
        )

        patience = int(stage["early_stopping_patience"])
        min_delta = float(stage["early_stopping_min_delta"])
        no_improve_epochs = 0
        best_val_loss_stage = float("inf")
        best_epoch_stage = 0

        for epoch in range(1, stage_epochs + 1):
            logging.info("Stage %s | Epoch %d/%d", stage_name, epoch, stage_epochs)

            train_stats = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                frame_batch_size=frame_batch_size,
                lam_loss=lam_loss,
                aux_enabled=aux_enabled,
                aux_weight_config=aux_weight,
                effective_aux_weight=effective_aux_weight,
                aux_baseline_ratio=aux_baseline_ratio,
                use_model_specific_aux_loss=use_model_specific_aux_loss,
                optimiser=optimiser,
                grad_clip_norm=grad_clip_norm,
                log_every_files=log_every_files,
                freeze_upsampler=freeze_upsampler,
                epoch=epoch,
            )

            with torch.no_grad():
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    frame_batch_size=frame_batch_size,
                    lam_loss=lam_loss,
                    aux_enabled=aux_enabled,
                    aux_weight_config=aux_weight,
                    effective_aux_weight=effective_aux_weight,
                    aux_baseline_ratio=aux_baseline_ratio,
                    use_model_specific_aux_loss=use_model_specific_aux_loss,
                    optimiser=None,
                    grad_clip_norm=0.0,
                    log_every_files=0,
                    freeze_upsampler=freeze_upsampler,
                    epoch=epoch,
                )

            epoch_row = {
                "stage": stage_name,
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
            }
            history.append(epoch_row)

            logging.info(
                "Stage %s epoch %d | train loss: %.6f | val loss: %.6f",
                stage_name,
                epoch,
                train_stats["loss_total"],
                val_stats["loss_total"],
            )

            stage_last_ckpt = ckpt_dir / f"{ckpt_prefix}_{stage_tag}_last.pth"
            latest_last_ckpt = ckpt_dir / f"{ckpt_prefix}_last.pth"
            save_state_dict_checkpoint(model, stage_last_ckpt)
            save_state_dict_checkpoint(model, latest_last_ckpt)

            improved = val_stats["loss_total"] < (best_val_loss_stage - min_delta)
            if improved:
                best_val_loss_stage = val_stats["loss_total"]
                best_epoch_stage = epoch
                no_improve_epochs = 0

                stage_best_ckpt = ckpt_dir / f"{ckpt_prefix}_{stage_tag}_best.pth"
                latest_best_ckpt = ckpt_dir / f"{ckpt_prefix}_best.pth"
                save_state_dict_checkpoint(model, stage_best_ckpt)
                save_state_dict_checkpoint(model, latest_best_ckpt)
                logging.info("New stage best checkpoint saved: %s", stage_best_ckpt.resolve())
            else:
                no_improve_epochs += 1

            if val_stats["loss_total"] < best_val_loss_overall:
                best_val_loss_overall = val_stats["loss_total"]
                best_stage_name = stage_name
                best_stage_epoch = epoch

            if patience > 0 and no_improve_epochs >= patience:
                logging.info(
                    "Early stopping in stage '%s' after %d epochs without improvement.",
                    stage_name,
                    no_improve_epochs,
                )
                break

        logging.info(
            "Stage '%s' best val loss: %.6f at epoch %d",
            stage_name,
            best_val_loss_stage,
            best_epoch_stage,
        )

    metrics_out = output_root / f"train_end_to_end_metrics_{timestamp}.json"
    save_training_meta(
        metrics_out,
        {
            "timestamp": timestamp,
            "device": str(device),
            "best_stage": best_stage_name,
            "best_epoch": best_stage_epoch,
            "best_val_loss": best_val_loss_overall,
            "history": history,
            "model": model_cfg,
            "training": training_cfg,
            "initialisation": initialisation_cfg,
            "loss": loss_cfg,
            "aux_calibration": aux_calibration_meta,
            "stages": stages,
        },
    )
    logging.info("Training history saved: %s", metrics_out.resolve())
    logging.info(
        "Best stage: %s | best epoch: %d | best val loss: %.6f",
        best_stage_name,
        best_stage_epoch,
        best_val_loss_overall,
    )

    train_loss_history = [
        {
            "stage": str(entry["stage"]),
            "epoch": int(entry["epoch"]),
            **{
                key: float(value)
                for key, value in dict(entry["train"]).items()
                if key.startswith("loss")
            },
        }
        for entry in history
    ]
    val_loss_history = [
        {
            "stage": str(entry["stage"]),
            "epoch": int(entry["epoch"]),
            **{
                key: float(value)
                for key, value in dict(entry["val"]).items()
                if key.startswith("loss")
            },
        }
        for entry in history
    ]

    logging.info(
        "Per-epoch training losses (compact):\n%s",
        format_compact_loss_series(train_loss_history),
    )
    logging.info(
        "Per-epoch validation losses (compact):\n%s",
        format_compact_loss_series(val_loss_history),
    )


if __name__ == "__main__":
    main()
