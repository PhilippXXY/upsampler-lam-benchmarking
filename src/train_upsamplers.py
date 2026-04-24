"""
Train upsampler models for low-channel to high-channel CSM reconstruction.

The trainer is model-agnostic: each upsampler owns its loss and optimisation
logic via `training_step` and `validation_step`.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from upsampler import TrainableUpsampler
from upsampler.ainn import AINNUpsampler
from upsampler.bicubic import BicubicUpsampler
from upsampler.gan import GANUpsampler
from upsampler.imdn import IMDNUpsampler
from upsampler.safmn import SAFMNUpsampler
from upsampler.srcnn import SRCNNUpsampler
from utils.training_utils import (
    build_dataset_list,
    build_train_loader,
    collate_single_item,
    configure_torch_multiprocessing,
    format_compact_loss_series,
    load_conf,
    load_state_dict_checkpoint,
    normalise_stages,
    resolve_training_device,
    save_state_dict_checkpoint,
    save_training_meta,
    setup_logging,
)
from utils.training_utils import (
    sanitise_name as _sanitise_name,
)
from utils.utils import seed_everything

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def build_model(model_cfg: dict[str, Any]) -> TrainableUpsampler:  # type: ignore[no-any-unimported]
    """
    Instantiate a trainable upsampler.

    Parameters
    ----------
    model_cfg : dict[str, Any]
        Model configuration dictionary with at least a "name" key specifying the
        model type, and optional additional parameters depending on the model.

    Returns
    -------
    TrainableUpsampler
        Instantiated model based on the configuration.

    Raises
    ------
    ValueError
        If the model name is unsupported or required parameters are missing.
    """
    model_name = model_cfg["name"]
    if model_name == "BicubicUpsampler":
        return BicubicUpsampler(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 32)),
            loss_name=str(model_cfg.get("loss_name", "l1")),
        )
    elif model_name == "SRCNNUpsampler":
        return SRCNNUpsampler(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 32)),
            feature_channels=int(model_cfg.get("feature_channels", 64)),
            mapping_channels=int(model_cfg.get("mapping_channels", 32)),
            loss_name=str(model_cfg.get("loss_name", "l1")),
        )
    elif model_name == "IMDNUpsampler":
        return IMDNUpsampler(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 32)),
            feature_channels=int(model_cfg.get("feature_channels", 64)),
            mapping_channels=int(model_cfg.get("mapping_channels", 32)),
            loss_name=str(model_cfg.get("loss_name", "l1")),
        )
    elif model_name == "SAFMNUpsampler":
        return SAFMNUpsampler(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 32)),
            feature_channels=int(model_cfg.get("feature_channels", 36)),
            n_blocks=int(model_cfg.get("n_blocks", 8)),
            ffn_scale=float(model_cfg.get("ffn_scale", 2.0)),
            n_levels=int(model_cfg.get("n_levels", 4)),
            loss_name=str(model_cfg.get("loss_name", "l1")),
            fft_loss_weight=float(model_cfg.get("fft_loss_weight", 0.05)),
        )
    elif model_name == "GANUpsampler":
        return GANUpsampler(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 32)),
            feature_channels=128,
            n_residual_blocks=8,
            loss_name=str(model_cfg.get("loss_name", "l1")),
            adversarial_weight=float(model_cfg.get("adversarial_weight", 0.01)),
            content_weight=float(model_cfg.get("content_weight", 0.1)),
            critic_iters=int(model_cfg.get("critic_iters", 4)),
            discriminator_lr_scale=float(model_cfg.get("discriminator_lr_scale", 5.0)),
            beta1_adam=(model_cfg.get("beta1_adam", 0.9)),
            beta2_adam=(model_cfg.get("beta2_adam", 0.999)),
        )
    elif model_name == "AINNUpsampler":
        return AINNUpsampler(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 32)),
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            low_channel_indices=tuple(
                int(index) for index in model_cfg.get("low_channel_indices", [5, 9, 21, 25])
            ),
            loss_name=str(model_cfg.get("loss_name", "mse")),
            pde_loss_weight=float(model_cfg.get("pde_loss_weight", 0.01)),
            pde_freq_min_hz=float(model_cfg.get("pde_freq_min_hz", 100.0)),
            pde_freq_max_hz=float(model_cfg.get("pde_freq_max_hz", 4000.0)),
            sound_speed=float(model_cfg.get("sound_speed", 343.0)),
        )
    raise ValueError(f"Unsupported model '{model_name}'.")


def run_epoch(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    model: TrainableUpsampler,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    frame_batch_size: int,
    optimiser: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
    grad_clip_norm: float = 0.0,
    log_every_files: int = 0,
) -> dict[str, float]:
    """
    Run one training or validation epoch.

    Parameters
    ----------
    model : TrainableUpsampler
        The model to train or validate.
    loader : DataLoader[dict[str, Any]]
        DataLoader providing batches of samples, where each sample is a dict containing "S_low"
        and "S_high" tensors.
    device : torch.device
        Device to run the computations on.
    frame_batch_size : int
        Number of frames to process in each training step (model forward pass).
    optimiser : torch.optim.Optimizer | dict[str, torch.optim.Optimizer], optional
        Optimiser input to use for training steps. Either a single optimiser,
        or a dictionary of optimisers for multi-optimiser models.
        If None, the model is put in evaluation mode and no optimisation is performed
        (default: None).
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping during training steps.
        If 0.0, no clipping is applied (default: 0.0).
    log_every_files : int, optional
        If greater than 0, log average loss every N files (default: 0, meaning no intermediate
        logging).

    Returns
    -------
    dict[str, float]
        Dictionary of average loss statistics for the epoch, including "loss_total" and any
        additional metrics returned by the model's step functions.
    """
    is_train = optimiser is not None
    model.train(is_train)

    loss_sums: dict[str, float] = {"loss_total": 0.0}
    chunk_count = 0

    iterator = tqdm(loader, ncols=100, desc="Train" if is_train else "Val")
    # We process each file (batch) and then split into frame batches for the model steps.
    for file_idx, sample in enumerate(iterator, start=1):
        S_low = sample["S_low"].to(device)
        S_high = sample["S_high"].to(device)
        n_frames = int(S_low.shape[0])

        for start in range(0, n_frames, frame_batch_size):
            end = min(start + frame_batch_size, n_frames)
            x = S_low[start:end]
            y = S_high[start:end]

            if optimiser is not None:
                stats = model.training_step(x, y, optimiser, grad_clip_norm)
            else:
                stats = model.validation_step(x, y)

            stats = model.normalise_step_stats(stats)
            for key, value in stats.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
            chunk_count += 1

        # Log intermediate average loss every N files if requested.
        if log_every_files > 0 and file_idx % log_every_files == 0:
            logging.info(
                "[%s] file %d/%d | avg loss %.6f",
                "train" if is_train else "val",
                file_idx,
                len(loader),
                loss_sums["loss_total"] / max(chunk_count, 1),
            )

    # Final average loss for the epoch.
    denom = max(chunk_count, 1)
    out = {key: value / denom for key, value in loss_sums.items()}
    out["num_chunks"] = float(chunk_count)
    return out


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """
    Run the training process based on configuration.

    This function handles loading the configuration, setting up logging, resolving the
    device, building the model and datasets, and running the training loop with multiple
    stages and early stopping.

    The training history and final metrics are saved to a JSON file at the end.
    """
    parser = argparse.ArgumentParser("Upsampler training")
    parser.add_argument("--config", type=str, default="config/train_upsamplers.yaml")
    parser.add_argument("--device", type=str, default="", help="cpu | mps | cuda")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to initialise model weights before training.",
    )
    args = parser.parse_args()

    config = load_conf(Path(args.config))
    training_cfg = config["training"]
    model_cfg = config["model"]

    output_root = Path(training_cfg.get("output_root", "output/training"))
    setup_logging(
        output_root.joinpath("logs"),
        log_stem="train_upsamplers",
        timestamp=timestamp,
    )

    train_generator = seed_everything(int(training_cfg.get("seed", 42)))

    requested_device = args.device if args.device else str(training_cfg.get("device", "cpu"))
    device = resolve_training_device(requested_device)
    logging.info("Using device: %s", device)

    # Build model and datasets based on config.
    # The model is expected to be a TrainableUpsampler that implements its own training and
    # validation step logic, returning loss statistics in a consistent format for logging
    # and checkpointing.
    model_cfg = dict(model_cfg)
    model_cfg["low_channel_indices"] = config["data"].get("low_channel_indices", [5, 9, 21, 25])
    model = build_model(model_cfg).to(device)

    resume_checkpoint = (
        args.resume_checkpoint.strip()
        if args.resume_checkpoint.strip()
        else str(training_cfg.get("resume_from_checkpoint", "")).strip()
    )
    resume_strict = bool(training_cfg.get("resume_strict", True))
    if resume_checkpoint:
        load_state_dict_checkpoint(
            model=model,
            path=Path(resume_checkpoint),
            strict=resume_strict,
        )

    num_workers = int(training_cfg.get("num_workers", 0))
    frame_batch_size = int(training_cfg.get("frame_batch_size", 64))
    grad_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0))
    log_every_files = int(training_cfg.get("log_every_files", 0))
    sharing_strategy = configure_torch_multiprocessing(num_workers)
    if sharing_strategy is not None:
        logging.info("Using torch multiprocessing sharing strategy: %s", sharing_strategy)

    ckpt_dir = Path(training_cfg.get("checkpoint_dir", "src/upsampler/bicubic/checkpoints"))
    ckpt_prefix = str(training_cfg.get("checkpoint_prefix", "bicubic_upsampler"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    stages = normalise_stages(training_cfg=training_cfg, data_cfg=config["data"])

    best_val_loss_overall = float("inf")
    best_stage_name = ""
    best_stage_epoch = 0

    # We loop over stages, where each stage can have different dataset enabling,
    # training parameters, and early stopping settings.
    for stage_idx, stage in enumerate(stages, start=1):
        if not stage["enabled"]:
            logging.info("Skipping disabled stage: %s", stage["name"])
            continue

        stage_name = str(stage["name"])
        stage_tag = _sanitise_name(stage_name)
        stage_epochs = int(stage["epochs"])
        if stage_epochs <= 0:
            logging.info("Skipping stage '%s' because epochs <= 0.", stage_name)
            continue

        logging.info("=== Stage %d/%d: %s ===", stage_idx, len(stages), stage_name)
        logging.info(
            "Stage datasets: audiblelight=%s eigenscape=%s | train_sampling=%s",
            stage["datasets"]["audiblelight"],
            stage["datasets"]["eigenscape"],
            stage["train_sampling"],
        )

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimiser: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None
        learning_rate = float(stage["learning_rate"])
        weight_decay = float(stage["weight_decay"])
        if trainable_params:
            if isinstance(model, GANUpsampler):
                beta1 = model.beta1 if model.beta1 is not None else 0.9
                beta2 = model.beta2 if model.beta2 is not None else 0.999
                optimiser = {
                    "generator": torch.optim.Adam(
                        model.generator.parameters(),
                        lr=learning_rate,
                        betas=(beta1, beta2),
                        weight_decay=weight_decay,
                    ),
                    "discriminator": torch.optim.Adam(
                        model.discriminator.parameters(),
                        lr=learning_rate * model.discriminator_lr_scale,
                        betas=(beta1, beta2),
                        weight_decay=weight_decay,
                    ),
                }
            else:
                optimiser = torch.optim.AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
        else:
            logging.info(
                "Model '%s' has no trainable parameters. "
                "Running loss-only stage without optimiser updates.",
                model_cfg["name"],
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
        logging.info("Stage train files: %d | val files: %d", len(train_dataset), len(val_dataset))

        train_loader = build_train_loader(
            datasets=train_datasets,
            num_workers=num_workers,
            device=device,
            sampling=str(stage["train_sampling"]),
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_single_item,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            **(
                {"prefetch_factor": 1, "persistent_workers": True} if num_workers > 0 else {}  # type: ignore[arg-type]
            ),
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
                optimiser=optimiser,
                grad_clip_norm=grad_clip_norm,
                log_every_files=log_every_files,
            )

            with torch.no_grad():
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    frame_batch_size=frame_batch_size,
                    optimiser=None,
                    grad_clip_norm=0.0,
                    log_every_files=0,
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

    # Save final training history and metrics.
    metrics_out = output_root / f"train_metrics_{timestamp}.json"
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
