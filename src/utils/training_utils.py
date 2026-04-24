"""Common helpers shared by the training entrypoints."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data.audiblelight_loader import AudibleLightCSMPairDataset
from data.eigenscape_loader import EigenscapeCSMPairDataset
from utils.utils import resolve_requested_device


def load_conf(config_path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected config file '{config_path}' to contain a mapping.")
    if not all(isinstance(key, str) for key in config):
        raise ValueError(f"Expected config file '{config_path}' to use string keys.")
    return cast(dict[str, Any], config)


def setup_logging(log_dir: Path, *, log_stem: str, timestamp: str) -> None:
    """
    Set up console and file logging.

    Parameters
    ----------
    log_dir : Path
        Directory where log files should be written.
    log_stem : str
        File stem used for the log filename.
    timestamp : str
        Timestamp suffix used in the log filename.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir.joinpath(f"{log_stem}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
    )
    logging.info("Logging to %s", log_file.resolve())


def resolve_training_device(
    requested: str,
    *,
    mps_fallback_reason: str | None = None,
) -> torch.device:
    """
    Resolve the training device with sensible fall-backs.

    Parameters
    ----------
    requested : str
        Requested device string, for example ``cpu``, ``mps``, or ``cuda``.

    Returns
    -------
    torch.device
        Resolved device to use for training.
    """
    resolved = resolve_requested_device(
        requested_device=requested,
        mps_fallback_reason=mps_fallback_reason,
    )
    if not isinstance(resolved, torch.device):
        raise TypeError("resolve_requested_device must return torch.device.")
    return resolved


def _as_tuple(values: Iterable[Any], expected_len: int, name: str) -> tuple[Any, ...]:
    """
    Convert an iterable to a tuple and enforce its length.

    Parameters
    ----------
    values : Iterable[Any]
        Input values to convert.
    expected_len : int
        Expected tuple length.
    name : str
        Parameter name used in the error message.

    Returns
    -------
    tuple[Any, ...]
        Tuple of the expected length.
    """
    out = tuple(values)
    if len(out) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {out}")
    return out


def build_dataset_list(
    config: dict[str, Any],
    split: str,
    enabled_overrides: dict[str, bool] | None = None,
    max_files: int = 0,
) -> list[Dataset[dict[str, Any]]]:
    """
    Build the datasets for a given split.

    Parameters
    ----------
    config : dict[str, Any]
        Full configuration dictionary containing dataset settings.
    split : str
        Dataset split to build, for example ``train`` or ``val``.
    enabled_overrides : dict[str, bool] | None, optional
        Optional per-dataset enable overrides.
    max_files : int, optional
        Optional limit on the number of files per dataset.

    Returns
    -------
    list[Dataset[dict[str, Any]]]
        List of instantiated datasets for the requested split.
    """
    data_cfg = config["data"]
    low_channels = _as_tuple(data_cfg["low_channel_indices"], 4, "low_channel_indices")
    sampling_rate = int(data_cfg.get("sampling_rate", 24000))
    nbands = int(data_cfg.get("nbands", 9))

    datasets: list[Dataset[dict[str, Any]]] = []

    def _enabled(name: str, default: bool) -> bool:
        if enabled_overrides is not None and name in enabled_overrides:
            return bool(enabled_overrides[name])
        return default

    def _optional_path(value: Any) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        return None if not text else Path(text)

    audible_cfg = data_cfg.get("audiblelight", {})
    if _enabled("audiblelight", bool(audible_cfg.get("enabled", False))):
        datasets.append(
            AudibleLightCSMPairDataset(
                root_path=Path(audible_cfg["root_path"]),
                split=split,
                split_ratio=tuple(audible_cfg.get("split_ratio", [0.8, 0.1, 0.1])),
                seed=int(audible_cfg.get("seed", 42)),
                low_channel_indices=low_channels,
                sampling_rate=sampling_rate,
                nbands=nbands,
                cache_csm=bool(audible_cfg.get("cache_csm", False)),
                precomputed_csm_root=_optional_path(audible_cfg.get("precomputed_csm_root")),
                max_files=max_files,
            )
        )

    eigenscape_cfg = data_cfg.get("eigenscape", {})
    if _enabled("eigenscape", bool(eigenscape_cfg.get("enabled", False))):
        expected_channels_cfg = eigenscape_cfg.get("expected_channels", 32)
        expected_channels = None if expected_channels_cfg is None else int(expected_channels_cfg)
        split_ratio_cfg = eigenscape_cfg.get("split_ratio")
        split_ratio = tuple(split_ratio_cfg) if split_ratio_cfg is not None else None
        split_counts = (
            None
            if split_ratio is not None
            else tuple(eigenscape_cfg.get("split_counts", [6, 1, 1]))
        )
        datasets.append(
            EigenscapeCSMPairDataset(
                root_path=Path(eigenscape_cfg["root_path"]),
                split=split,
                split_counts=split_counts,
                split_ratio=split_ratio,
                seed=int(eigenscape_cfg.get("seed", 42)),
                low_channel_indices=low_channels,
                sampling_rate=sampling_rate,
                nbands=nbands,
                cache_csm=bool(eigenscape_cfg.get("cache_csm", False)),
                precomputed_csm_root=_optional_path(eigenscape_cfg.get("precomputed_csm_root")),
                max_files=max_files,
                expected_channels=expected_channels,
                target_high_channels=int(eigenscape_cfg.get("target_high_channels", 32)),
                allow_channel_fallback=bool(eigenscape_cfg.get("allow_channel_fallback", False)),
            )
        )

    if not datasets:
        raise ValueError("No datasets enabled in config.data")
    return datasets


def collate_single_item(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate a single file-level sample.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        Batch produced by the DataLoader.

    Returns
    -------
    dict[str, Any]
        The single sample contained in the batch.
    """
    if len(batch) != 1:
        raise ValueError("This trainer expects file-level batch_size=1")
    return batch[0]


def configure_torch_multiprocessing(num_workers: int) -> str | None:
    """
    Select a safer tensor-sharing strategy when worker processes are enabled.

    This is done as we have had some big issues in the e2e training pipeline.

    Parameters
    ----------
    num_workers : int
        Number of DataLoader workers requested by the training config.

    Returns
    -------
    str | None
        Sharing strategy that was applied, otherwise ``None``.
    """
    if num_workers <= 0:
        return None

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError):
        return None
    return "file_system"


def build_train_loader(
    datasets: list[Dataset[dict[str, Any]]],
    num_workers: int,
    device: torch.device,
    sampling: str,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, Any]]:
    """
    Build the training DataLoader.

    Parameters
    ----------
    datasets : list[Dataset[dict[str, Any]]]
        Datasets to include in the training loader.
    num_workers : int
        Number of worker processes.
    device : torch.device
        Training device, used for pin-memory decisions.
    sampling : str
        Sampling mode, either ``proportional`` or ``balanced``.
    generator : torch.Generator | None, optional
        Optional torch RNG used for deterministic shuffling and weighted sampling.

    Returns
    -------
    DataLoader[dict[str, Any]]
        Training DataLoader.
    """

    def _dataset_len(dataset: Dataset[dict[str, Any]]) -> int:
        if not isinstance(dataset, Sized):
            raise TypeError(f"Dataset '{type(dataset).__name__}' must define __len__().")
        return len(dataset)

    pin_memory = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # Default prefetching is too memory-hungry for these file-level samples.
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True

    if len(datasets) == 1:
        return DataLoader(
            datasets[0],
            batch_size=1,
            shuffle=True,
            collate_fn=collate_single_item,
            generator=generator,
            **loader_kwargs,
        )

    concat: ConcatDataset[dict[str, Any]] = ConcatDataset(datasets)
    sampling_mode = sampling.strip().lower()
    if sampling_mode == "balanced":
        weights: list[float] = []
        for dataset in datasets:
            dataset_len = _dataset_len(dataset)
            n_items = max(dataset_len, 1)
            weights.extend([1.0 / n_items] * dataset_len)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(concat),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            concat,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_single_item,
            generator=generator,
            **loader_kwargs,
        )

    return DataLoader(
        concat,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single_item,
        generator=generator,
        **loader_kwargs,
    )


def normalise_stages(
    training_cfg: dict[str, Any], data_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Normalise stage configuration and fill in defaults.

    Parameters
    ----------
    training_cfg : dict[str, Any]
        Training configuration dictionary.
    data_cfg : dict[str, Any]
        Data configuration dictionary.

    Returns
    -------
    list[dict[str, Any]]
        Normalised stage dictionaries.
    """
    stages = training_cfg.get("stages")
    if not stages:
        return [
            {
                "name": "train",
                "enabled": True,
                "epochs": int(training_cfg.get("epochs", 20)),
                "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
                "weight_decay": float(training_cfg.get("weight_decay", 1e-4)),
                "max_train_files": int(training_cfg.get("max_train_files", 0)),
                "max_val_files": int(training_cfg.get("max_val_files", 0)),
                "train_sampling": "proportional",
                "early_stopping_patience": int(training_cfg.get("early_stopping_patience", 0)),
                "early_stopping_min_delta": float(
                    training_cfg.get("early_stopping_min_delta", 0.0)
                ),
                "datasets": {
                    "audiblelight": bool(data_cfg.get("audiblelight", {}).get("enabled", False)),
                    "eigenscape": bool(data_cfg.get("eigenscape", {}).get("enabled", False)),
                },
            }
        ]

    normalised: list[dict[str, Any]] = []
    for idx, stage in enumerate(stages, start=1):
        stage_name = str(stage.get("name", f"stage_{idx}"))
        stage_datasets = stage.get("datasets", {})
        normalised.append(
            {
                "name": stage_name,
                "enabled": bool(stage.get("enabled", True)),
                "epochs": int(stage.get("epochs", training_cfg.get("epochs", 20))),
                "learning_rate": float(
                    stage.get("learning_rate", training_cfg.get("learning_rate", 1e-4))
                ),
                "weight_decay": float(
                    stage.get("weight_decay", training_cfg.get("weight_decay", 1e-4))
                ),
                "max_train_files": int(
                    stage.get("max_train_files", training_cfg.get("max_train_files", 0))
                ),
                "max_val_files": int(
                    stage.get("max_val_files", training_cfg.get("max_val_files", 0))
                ),
                "train_sampling": str(stage.get("train_sampling", "proportional")),
                "early_stopping_patience": int(
                    stage.get(
                        "early_stopping_patience", training_cfg.get("early_stopping_patience", 0)
                    )
                ),
                "early_stopping_min_delta": float(
                    stage.get(
                        "early_stopping_min_delta",
                        training_cfg.get("early_stopping_min_delta", 0.0),
                    )
                ),
                "datasets": {
                    "audiblelight": bool(
                        stage_datasets.get(
                            "audiblelight", data_cfg.get("audiblelight", {}).get("enabled", False)
                        )
                    ),
                    "eigenscape": bool(
                        stage_datasets.get(
                            "eigenscape", data_cfg.get("eigenscape", {}).get("enabled", False)
                        )
                    ),
                },
            }
        )
    return normalised


def save_state_dict_checkpoint(model: nn.Module, path: Path) -> None:
    """
    Save a model state dict as ``.pth``.

    Parameters
    ----------
    model : nn.Module
        Model whose state dict should be saved.
    path : Path
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state_dict_checkpoint(model: nn.Module, path: Path, strict: bool = True) -> None:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        Model to load.
    path : Path
        Checkpoint path.
    strict : bool, optional
        Whether to enforce exact key matching.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state_dict = (
        payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")

    model.load_state_dict(state_dict, strict=strict)
    logging.info("Loaded checkpoint: %s", path.resolve())


def save_training_meta(path: Path, payload: dict[str, Any]) -> None:
    """
    Save training metadata as JSON.

    Parameters
    ----------
    path : Path
        Output JSON path.
    payload : dict[str, Any]
        Metadata payload to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sanitise_name(name: str) -> str:
    """
    Sanitise a name for checkpoint filenames.

    Parameters
    ----------
    name : str
        Input name.

    Returns
    -------
    str
        Sanitised name.
    """
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def format_compact_loss_series(loss_history: list[dict[str, Any]]) -> str:
    """
    Format per-epoch loss rows as compact stage-grouped text for logging.

    Parameters
    ----------
    loss_history : list[dict[str, Any]]
        Per-epoch loss rows with at least ``stage`` and ``epoch`` keys,
        plus one or more keys starting with ``loss``.

    Returns
    -------
    str
        Multi-line compact text representation of the loss series.
    """
    if not loss_history:
        return "  (no epochs)"

    grouped_epochs: dict[str, list[int]] = {}
    grouped_losses: dict[str, dict[str, list[float]]] = {}
    for row in loss_history:
        stage = str(row["stage"])
        grouped_epochs.setdefault(stage, []).append(int(row["epoch"]))

        stage_losses = grouped_losses.setdefault(stage, {})
        for key, value in row.items():
            if key.startswith("loss"):
                stage_losses.setdefault(key, []).append(float(value))

    lines: list[str] = []
    for stage, metrics in grouped_losses.items():
        epoch_text = ", ".join(str(epoch) for epoch in grouped_epochs[stage])
        metric_text = " | ".join(
            f"{metric}=[{', '.join(f'{value:.6f}' for value in values)}]"
            for metric, values in sorted(metrics.items())
        )
        lines.append(f"  stage={stage} | epochs=[{epoch_text}] | {metric_text}")

    return "\n".join(lines)
