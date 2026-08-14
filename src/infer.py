"""
Inference script for LAM models with upsampling beforehand.

This module loads configuration files, sets up logging, prepares datasets,
runs inference using the specified model, and writes predictions and metrics
to output files in DCASE and CSV formats.
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from data.variable_channels import CANONICAL_CHANNELS, embed_observed_csm
from lam_min.dataset.gen_dataset.gen_dataset import get_visibility_matrix
from lam_min.doa_metrics import compute_seld_metrics_for_files
from lam_min.util.utils import (
    load_ainn_lam_state,
    load_bicubic_lam_state,
    load_checkpoint,
    load_gan_lam_state,
    load_imdn_lam_state,
    load_safmn_lam_state,
    load_srcnn_lam_state,
    resolve_safmn_inference_architecture,
)
from utils.cmd_metrics import (
    add_cmd_metrics,
    add_upsampler_validity_metrics,
    project_to_hermitian_psd,
)
from utils.model_variants import resolve_exact_variant, resolve_variant_path
from utils.runtime_measurement import (
    apply_steady_state_runtime_metrics,
    runtime_measurement_enabled,
    runtime_measurement_summary,
)
from utils.utils import (
    _print_metrics_summary,
    cluster_intensity_maps,
    locata_selection_group,
    prepare_audio_for_inference,
    resolve_requested_device,
    seed_everything,
    select_stratified_file_ids,
    select_stratified_wavs,
    visibility_t_sti_seconds,
    write_output_dcase_csv,
)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LAM_MIN_MODELS = {
    "LAM",
    "UpLAM",
    "BicubicLAM",
    "SRCNNLAM",
    "IMDNLAM",
    "SAFMNLAM",
    "GANLAM",
    "AINNLAM",
    "VariableSRCNNLAM",
}

# The expected length of the model output tuple when metrics are collected.
# Newer models return (S_out, I_pred, metrics), while older models may return just (S_out, I_pred).
MODEL_RESULT_WITH_AUX_LEN = 3

LAM_ARTIFACT_OUTPUT_KEYS = (
    "lam_final",
    "lam_denoise1",
    "lam_denoise2",
    "lam_denoise3",
    "lam_denoise4",
)


def _extract_csm_stage_outputs(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Read detached CSM artefacts cached by the most recent forward pass.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary mapping stage names to their corresponding CSM tensors, if available.
    """
    stage_outputs: dict[str, torch.Tensor] = {}

    upsampler_output = getattr(model, "_last_upsampler_output", None)
    if isinstance(upsampler_output, torch.Tensor):
        stage_outputs["upsampler"] = upsampler_output.detach()

    lam_module = getattr(model, "lam", model)
    lam_artifacts = getattr(lam_module, "_last_forward_artifacts", {})
    if not isinstance(lam_artifacts, dict):
        return stage_outputs

    for artifact_key in LAM_ARTIFACT_OUTPUT_KEYS:
        artifact_value = lam_artifacts.get(artifact_key)
        if isinstance(artifact_value, torch.Tensor):
            stage_outputs[artifact_key] = artifact_value.detach()

    return stage_outputs


def _resolve_inference_variant(
    inference_config: dict[str, Any],
    repo_root: Path,
) -> None:
    """
    Resolve `model_variant` into the runtime loader fields used by inference.

    Parameters
    ----------
    inference_config : dict[str, Any]
        The inference configuration dictionary containing at least the key "model_variant".
    repo_root : Path
        The root path of the repository, used to resolve checkpoint paths.

    Raises
    ------
    ValueError
        If "model_variant" is missing from the configuration or if it cannot be resolved to a known
        variant.
    """
    raw_variant = str(inference_config.get("model_variant", "")).strip().lower()
    if not raw_variant:
        raise ValueError(
            "Config is missing required key inference.model_variant. "
            "Use an exact retained variant identifier such as 'bicubiclam_dist' "
            "or 'ainnlam_e2e_auxen'."
        )

    variant = resolve_exact_variant(raw_variant)
    missing_paths = variant.missing_paths(repo_root)
    if missing_paths:
        raise FileNotFoundError(
            f"Configured variant '{variant.variant_id}' is missing checkpoint(s): "
            + ", ".join(str(path) for path in missing_paths)
        )

    inference_config["model_variant"] = variant.variant_id
    inference_config["family_id"] = variant.family_id
    inference_config["family_label"] = variant.family_label
    inference_config["variant_kind"] = variant.variant_kind
    inference_config["variant_colour"] = variant.colour
    inference_config["model_name"] = variant.infer_model_name
    inference_config["model_checkpoint"] = (
        str(resolve_variant_path(repo_root, variant.checkpoint)) if variant.checkpoint else ""
    )
    if variant.lam_checkpoint is None:
        inference_config.pop("lam_checkpoint", None)
    else:
        inference_config["lam_checkpoint"] = str(
            resolve_variant_path(repo_root, variant.lam_checkpoint)
        )


def load_conf_files(config_path: Path = Path("config/inference_config.yaml")) -> dict[str, Any]:
    """
    Load configuration file.

    Parameters
    ----------
    config_path : Path, optional
        Path to the YAML configuration file. Defaults to 'config/inference_config.yaml'.

    Returns
    -------
    dict[str, Any]
        A dictionary containing configuration data loaded from the config file.
    """
    with open(Path(config_path), "r") as f:
        config = yaml.safe_load(f)

        configs = {}
        configs["inference"] = config["inference"]
        configs["dataset"] = config["dataset"]

        return configs


def setup_logging(inference_config: dict[str, Any]) -> str | None:
    """
    Set up logging configuration from a provided configuration dictionary.

    Parameters
    ----------
    inference_config: dict[str, Any]
        A dictionary containing logging configuration.
            Expected to have a "logging" key with handler configurations.
            Each handler can be of type "file" or "console" and supports:
            - type: "file" or "console"
            - level: logging level (e.g., "INFO", "DEBUG", "WARNING")
            - format: log message format string
            - filename: (file handler only) path to log file, supports {timestamp} placeholder
            - mode: (file handler only) file open mode (default: "a")

    Returns
    -------
    str | None
        The path to the log file if a file handler is configured,
        None otherwise.
    """
    log_filename = None
    if "logging" in inference_config:
        logging_config = inference_config["logging"]

        # Create root logger
        logger = logging.getLogger()
        # Set root logger from config, but keep it low enough for handlers.
        root_level = getattr(
            logging,
            str(logging_config.get("level", "INFO")).upper(),
            logging.INFO,
        )
        if "handlers" in logging_config:
            handler_levels = [
                getattr(logging, h.get("level", "DEBUG"))
                for h in logging_config.get("handlers", [])
            ]
            if handler_levels:
                root_level = min(root_level, *handler_levels)
        logger.setLevel(root_level)
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(logging_config.get("format", "%(levelname)s - %(message)s"))

        # Setup handlers
        for handler_config in logging_config.get("handlers", []):
            if handler_config["type"] == "file":
                # Create log directory if it doesn't exist
                filename = handler_config["filename"]
                # Substitute timestamp placeholder
                if "{timestamp}" in filename:
                    filename = filename.replace("{timestamp}", timestamp)

                log_filename = filename
                log_file = Path(filename)
                log_file.parent.mkdir(parents=True, exist_ok=True)

                handler = logging.FileHandler(filename, mode=handler_config.get("mode", "a"))
                handler.setLevel(getattr(logging, handler_config.get("level", "DEBUG")))
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            elif handler_config["type"] == "console":
                console_handler = logging.StreamHandler()
                console_handler.setLevel(getattr(logging, handler_config.get("level", "INFO")))
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

    return log_filename


def _select_dataset_wavs(
    wavs: list[Path], inference_config: dict[str, Any], generator: torch.Generator
) -> list[Path]:
    """
    Select a subset of dataset files according to the inference configuration.

    Parameters
    ----------
    wavs : list[Path]
        List of available WAV file paths in the dataset.
    inference_config : dict[str, Any]
        The inference configuration dictionary containing selection parameters:
        - max_files: int, maximum number of files to select (0 = all)
        - file_selection_mode: str, one of "sorted", "random", "stratified"
        - selected_files: list[str], optional explicit list of file IDs to select
    generator : torch.Generator
        A seeded torch generator for reproducible random selection.

    Returns
    -------
    list[Path]
        List of selected WAV file paths based on the specified selection criteria.
    """
    max_files = int(inference_config.get("max_files", 0))
    selection_mode = (
        str(inference_config.get("file_selection_mode", "sorted") or "sorted").strip().lower()
    )
    requested_files = [str(file_id) for file_id in (inference_config.get("selected_files") or [])]

    available_by_id = {wav_path.stem: wav_path for wav_path in wavs}

    if requested_files:
        missing = [file_id for file_id in requested_files if file_id not in available_by_id]
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise ValueError(
                "The following selected_files entries were not found in the dataset: "
                f"{missing_preview}"
            )
        selected_wavs = [available_by_id[file_id] for file_id in requested_files]
        if max_files > 0:
            selected_wavs = selected_wavs[:max_files]
        return selected_wavs

    if max_files <= 0 or max_files >= len(wavs):
        return list(wavs)

    if selection_mode == "sorted":
        return list(wavs[:max_files])
    if selection_mode == "random":
        indices = torch.randperm(len(wavs), generator=generator)[:max_files].tolist()
        return sorted([wavs[index] for index in indices])
    if selection_mode == "stratified":
        stratified_wavs: list[Path] = select_stratified_wavs(list(wavs), max_files, generator)
        return stratified_wavs

    raise ValueError(
        f"Unsupported file_selection_mode '{selection_mode}'. "
        "Use one of: sorted, random, stratified."
    )


def _select_locata_file_ids(
    file_ids: list[str], inference_config: dict[str, Any], generator: torch.Generator
) -> list[str]:
    """
    Select LOCATA recordings according to the inference configuration.

    Parameters
    ----------
    file_ids : list[str]
        Available LOCATA file IDs, e.g. ``task1_recording1``.
    inference_config : dict[str, Any]
        Inference configuration dictionary.
    generator : torch.Generator
        Seeded generator for deterministic selection.

    Returns
    -------
    list[str]
        Selected file IDs in deterministic order.
    """
    max_files = int(inference_config.get("max_files", 0))
    selection_mode = (
        str(inference_config.get("file_selection_mode", "sorted") or "sorted").strip().lower()
    )
    requested_files = [str(file_id) for file_id in (inference_config.get("selected_files") or [])]
    available_by_id = {file_id: file_id for file_id in file_ids}

    if requested_files:
        missing = [file_id for file_id in requested_files if file_id not in available_by_id]
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise ValueError(
                "The following selected_files entries were not found in the dataset: "
                f"{missing_preview}"
            )
        selected_file_ids = [available_by_id[file_id] for file_id in requested_files]
        if max_files > 0:
            selected_file_ids = selected_file_ids[:max_files]
        return selected_file_ids

    if max_files <= 0 or max_files >= len(file_ids):
        return list(file_ids)

    if selection_mode == "sorted":
        return list(file_ids[:max_files])
    if selection_mode == "random":
        indices = torch.randperm(len(file_ids), generator=generator)[:max_files].tolist()
        return sorted([file_ids[index] for index in indices])
    if selection_mode == "stratified":
        stratified_file_ids: list[str] = select_stratified_file_ids(
            file_ids=list(file_ids),
            max_files=max_files,
            generator=generator,
            group_fn=locata_selection_group,
        )
        return stratified_file_ids

    raise ValueError(
        f"Unsupported file_selection_mode '{selection_mode}'. "
        "Use one of: sorted, random, stratified."
    )


def _resolve_locata_tasks(inference_config: dict[str, Any]) -> tuple[str, ...]:
    """
    Resolve configured LOCATA tasks, defaulting to tasks 1-4.

    Parameters
    ----------
    inference_config : dict[str, Any]
        Inference configuration dictionary.

    Returns
    -------
    tuple[str, ...]
        Normalised LOCATA task names.
    """
    from data.locata_loader import DEFAULT_LOCATA_TASKS  # noqa: PLC0415

    configured_tasks = inference_config.get("locata_tasks")
    if configured_tasks is None:
        default_tasks = tuple(str(task) for task in DEFAULT_LOCATA_TASKS)
        return default_tasks

    if isinstance(configured_tasks, str):
        tasks = tuple(task.strip() for task in configured_tasks.split(",") if task.strip())
    else:
        tasks = tuple(str(task).strip() for task in configured_tasks if str(task).strip())

    if not tasks:
        raise ValueError("locata_tasks must contain at least one task name")
    return tasks


def _build_ground_truth_loader(
    dataset_name: str,
    dataset_config: dict[str, Any],
    inference_config: dict[str, Any],
    locata_tasks: tuple[str, ...] | None,
) -> Any:
    """
    Instantiate the dataset-specific ground-truth loader.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset (e.g., "starss23", "locata").
    dataset_config : dict[str, Any]
        The dataset configuration dictionary containing paths and settings.
    inference_config : dict[str, Any]
        The inference configuration dictionary containing settings like frame width.
    locata_tasks : tuple[str, ...] | None
        The LOCATA tasks to load, required if dataset_name is "locata" and model is not "LAM".

    Returns
    -------
    object
        An instance of the ground-truth loader appropriate for the specified dataset.
    """
    if dataset_name == "starss23":
        from data.starss_loader import StarssGroundTruthLoader  # noqa: PLC0415

        return StarssGroundTruthLoader(
            dataset_config["data_ground_truth_path"],
            frame_width_ms=inference_config["frame_width_ms"],
        )

    if dataset_name == "locata":
        from data.locata_loader import LocataGroundTruthLoader  # noqa: PLC0415

        return LocataGroundTruthLoader(
            ground_truth_path=Path(dataset_config["data_audio_path"]),
            frame_width_ms=inference_config["frame_width_ms"],
            tasks=locata_tasks or _resolve_locata_tasks(inference_config),
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """
    Execute inference on audio data using a trained localisation model.

    Loads configuration files for inference and dataset settings, sets up logging,
    and initialises the specified model (UpLAM, LAM, BicubicLAM, SRCNNLAM, IMDNLAM,
    SAFMNLAM, GANLAM, AINNLAM, or VariableSRCNNLAM).

    Processes audio files through the data loader, computes visibility matrices,
    performs model inference, and writes predictions in DCASE CSV format unless
    runtime-only benchmark mode is enabled. Optionally collects and logs
    performance metrics including latency, FLOPs, and memory usage.

    Supported datasets: starss23
    Supported models: UpLAM, LAM, BicubicLAM, SRCNNLAM, VariableSRCNNLAM, IMDNLAM,
                      SAFMNLAM, GANLAM, AINNLAM

    Command-line arguments
    ----------------------
    --device (str): Device to run inference on ('cpu', 'mps', 'cuda', 'cuda:0', or '0').
                    Default: 'cpu'
    --input-channel-indices (int ...): Canonical indices required by VariableSRCNNLAM.

    Output
    ------
    - Predictions saved as CSV files in the configured output directory
    - Metrics JSON and CSV files (if collect_metrics is enabled)
    - Log file (if logging is configured)

    Raises
    ------
    ValueError
        If unsupported dataset or model is specified in configuration

    Examples
    --------
    '''
    uv run python src/infer.py
    '''
    """
    ## Parameters
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to run inference on (e.g., 'cpu', 'mps', or 'cuda').",
    )
    parser.add_argument(
        "--config",
        default="config/inference_config.yaml",
        type=str,
        help="Path to the inference configuration YAML file.",
    )
    parser.add_argument(
        "--input-channel-indices",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based canonical Eigenmike indices for VariableSRCNNLAM input channels.",
    )
    args = parser.parse_args()

    ## Configurations
    repo_root = Path(__file__).resolve().parents[1]
    configs = load_conf_files(args.config)
    inference_config = configs["inference"]
    dataset_config = configs["dataset"]
    _resolve_inference_variant(inference_config, repo_root=repo_root)
    is_variable_srcnn = inference_config["model_name"] == "VariableSRCNNLAM"
    if is_variable_srcnn and args.input_channel_indices is None:
        parser.error("VariableSRCNNLAM requires --input-channel-indices.")
    if not is_variable_srcnn and args.input_channel_indices is not None:
        parser.error("--input-channel-indices is only valid for VariableSRCNNLAM.")
    selection_seed = int(inference_config.get("file_selection_seed", 0) or 0)
    selection_generator = seed_everything(selection_seed)

    # Setup logging
    log_filename = None
    if "logging" in inference_config:
        log_filename = setup_logging(inference_config)

    logging.info("Starting inference...")
    logging.info(
        "Resolved retained variant '%s' -> family=%s (%s), kind=%s, model_name=%s",
        inference_config["model_variant"],
        inference_config["family_id"],
        inference_config["family_label"],
        inference_config["variant_kind"],
        inference_config["model_name"],
    )
    logging.info(
        "Resolved checkpoints -> model_checkpoint=%s, lam_checkpoint=%s",
        inference_config["model_checkpoint"] or "<none>",
        inference_config.get("lam_checkpoint", "<none>"),
    )

    ## Preparation
    # Define output paths with model-date subfolder
    base_output_path = Path(inference_config["output_path"])
    date_str = timestamp
    model_name = inference_config["model_name"]
    run_id = str(inference_config["model_variant"])
    output_path = base_output_path.joinpath(f"{run_id}-{date_str}")
    output_path.mkdir(parents=True, exist_ok=True)
    # Set device
    device = resolve_requested_device(
        args.device,
        mps_fallback_reason=(
            "Model " f"'{model_name}' uses lam_min, which is configured for CPU/CUDA in this repo."
            if model_name in LAM_MIN_MODELS
            else None
        ),
    )

    logging.info(f"Using device: {device}")

    # Set data loader
    locata_tasks: tuple[str, ...] | None = None
    data_loader: DataLoader[Any]
    dataset_size: int
    selected_file_preview = ""
    if inference_config["data_set"] == "starss23":
        from data.starss_loader import StarssAudioDataset  # noqa: PLC0415

        starss_dataset = StarssAudioDataset(
            audio_path=Path(dataset_config["data_audio_path"]),
            ground_truth_path=Path(dataset_config["data_ground_truth_path"]),
            load_ground_truth=False,
            frame_width_ms=inference_config["frame_width_ms"],
        )
        original_dataset_size = len(starss_dataset.wavs)
        selected_wavs = _select_dataset_wavs(
            starss_dataset.wavs,
            inference_config,
            selection_generator,
        )
        starss_dataset.wavs = selected_wavs
        dataset_size = len(starss_dataset)
        selected_file_preview = ", ".join(wav_path.stem for wav_path in selected_wavs[:5])
        data_loader = DataLoader(
            starss_dataset,
            batch_size=inference_config["batch_size"],
            num_workers=inference_config["num_workers"],
            shuffle=False,
        )
    elif inference_config["data_set"] == "locata":
        from data.locata_loader import LocataAudioDataset  # noqa: PLC0415

        locata_tasks = _resolve_locata_tasks(inference_config)
        locata_dataset = LocataAudioDataset(
            path=Path(dataset_config["data_audio_path"]),
            load_ground_truth=False,
            frame_width_ms=inference_config["frame_width_ms"],
            tasks=locata_tasks,
        )
        original_dataset_size = len(locata_dataset)
        selected_file_ids = _select_locata_file_ids(
            [entry.file_id for entry in locata_dataset.entries],
            inference_config,
            selection_generator,
        )
        entry_by_id = {entry.file_id: entry for entry in locata_dataset.entries}
        selected_entries = [entry_by_id[file_id] for file_id in selected_file_ids]
        locata_dataset.entries = selected_entries
        locata_dataset.relevant_dir = [entry.eigenmike_dir for entry in selected_entries]
        locata_dataset.wavs = [entry.wav_path for entry in selected_entries]
        dataset_size = len(locata_dataset)
        selected_file_preview = ", ".join(entry.file_id for entry in selected_entries[:5])
        data_loader = DataLoader(
            locata_dataset,
            batch_size=inference_config["batch_size"],
            num_workers=inference_config["num_workers"],
            shuffle=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {inference_config['data_set']}")

    logging.info(
        f"Loaded dataset: {inference_config['data_set']}, "
        f"number of files: {dataset_size} (from {original_dataset_size}), "
        f"batch size: {inference_config['batch_size']}, "
        f"number of workers: {inference_config['num_workers']}, "
        f"frame width (ms): {inference_config['frame_width_ms']}, "
        f"sampling rate: {inference_config['sampling_rate']} Hz, "
        f"max audio length (sec): {inference_config['max_audio_length_sec']}, "
        f"max files: {inference_config['max_files']}, "
        f"file selection mode: {inference_config.get('file_selection_mode', 'sorted')}, "
        f"file selection seed: {selection_seed}"
    )
    if inference_config["data_set"] == "locata":
        logging.info("LOCATA tasks: %s", ", ".join(locata_tasks or ()))
    logging.info("Selected files preview: %s", selected_file_preview or "none")
    logging.info(
        "Visibility regularisation: diagonal_loading=%g, eigenvalue_floor=%g",
        float(inference_config.get("visibility_diagonal_loading", 0.0) or 0.0),
        float(inference_config.get("visibility_eigenvalue_floor", 0.0) or 0.0),
    )
    visibility_t_sti = visibility_t_sti_seconds(float(inference_config["frame_width_ms"]))

    ## Inference
    model: torch.nn.Module
    if inference_config["model_name"] == "UpLAM":
        from lam_min.model.UpLAM import UpLAM  # noqa: PLC0415

        model = UpLAM(num_bands=9)
        state = load_checkpoint(inference_config["model_checkpoint"], device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "LAM":
        from lam_min.model.LAM import LAM  # noqa: PLC0415

        model = LAM(num_bands=9)
        state = load_checkpoint(inference_config["model_checkpoint"], device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "BicubicLAM":
        from lam_min.model.BicubicLAM import BicubicLAM  # noqa: PLC0415

        model = BicubicLAM(num_bands=9, in_channels=4, out_channels=32)
        load_bicubic_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "SRCNNLAM":
        from lam_min.model.SRCNNLAM import SRCNNLAM  # noqa: PLC0415

        model = SRCNNLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            feature_channels=64,
            mapping_channels=32,
        )
        load_srcnn_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "VariableSRCNNLAM":
        from lam_min.model.VariableSRCNNLAM import VariableSRCNNLAM  # noqa: PLC0415

        model = VariableSRCNNLAM(
            num_bands=9,
            out_channels=32,
            feature_channels=64,
            mapping_channels=32,
            variable_input_channel_counts=tuple(
                int(count)
                for count in inference_config.get(
                    "variable_input_channel_counts", [4, 8, 16, 24, 32]
                )
            ),
        )
        load_srcnn_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "IMDNLAM":
        from lam_min.model.IMDNLAM import IMDNLAM  # noqa: PLC0415

        model = IMDNLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            feature_channels=64,
            mapping_channels=32,
        )
        load_imdn_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "SAFMNLAM":
        from lam_min.model.SAFMNLAM import SAFMNLAM  # noqa: PLC0415

        safmn_arch, inferred_safmn = resolve_safmn_inference_architecture(
            inference_config=inference_config,
            checkpoint_path=inference_config["model_checkpoint"],
            device=device,
        )

        if inferred_safmn is not None:
            logging.info(
                "Resolved SAFMN architecture from checkpoint: "
                "feature_channels=%d, n_blocks=%d, ffn_scale=%.6g, n_levels=%d",
                inferred_safmn["feature_channels"],
                inferred_safmn["n_blocks"],
                inferred_safmn["ffn_scale"],
                inferred_safmn["n_levels"],
            )

        logging.info(
            "Using SAFMN architecture for inference: "
            "feature_channels=%d, n_blocks=%d, ffn_scale=%.6g, n_levels=%d",
            safmn_arch["feature_channels"],
            safmn_arch["n_blocks"],
            safmn_arch["ffn_scale"],
            safmn_arch["n_levels"],
        )

        model = SAFMNLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            feature_channels=int(safmn_arch["feature_channels"]),
            n_blocks=int(safmn_arch["n_blocks"]),
            ffn_scale=float(safmn_arch["ffn_scale"]),
            n_levels=int(safmn_arch["n_levels"]),
        )
        load_safmn_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "AINNLAM":
        from lam_min.model.AINNLAM import AINNLAM  # noqa: PLC0415

        model = AINNLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            low_channel_indices=tuple(
                int(index)
                for index in inference_config.get("locata_low_channel_indices", [5, 9, 21, 25])
            ),
        )
        load_ainn_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    elif inference_config["model_name"] == "GANLAM":
        from lam_min.model.GANLAM import GANLAM  # noqa: PLC0415

        model = GANLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            feature_channels=128,
            n_residual_blocks=8,
        )
        load_gan_lam_state(
            model,
            inference_config["model_checkpoint"],
            device,
            lam_checkpoint=inference_config.get("lam_checkpoint"),
        )
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unsupported model: {inference_config['model_name']}")

    # Count inference-relevant parameters once (constant across files).
    # The GAN upsampler contains a discriminator submodule that is not used at inference time,
    # so it is excluded to reflect the actual inference-time parameter footprint.
    _discriminator = getattr(model, "discriminator", None) or getattr(
        getattr(model, "upsampler", None), "discriminator", None
    )
    if _discriminator is not None:
        _disc_params = {id(p) for p in _discriminator.parameters()}
        total_params = sum(p.numel() for p in model.parameters() if id(p) not in _disc_params)
    else:
        total_params = sum(p.numel() for p in model.parameters())

    # Inference loop
    processed_file_ids: list[tuple[str, str]] = []
    all_metrics: list[dict[str, Any]] = []  # Collect metrics for all processed files
    collect_metrics = inference_config.get("collect_metrics", False)

    # Calculate total items to process for progress bar
    total_items = len(data_loader)
    if inference_config["max_files"] > 0:
        total_items = min(total_items, inference_config["max_files"])

    runtime_only_benchmark = bool(inference_config.get("benchmark_runtime_only", False))

    log_msg = (
        "Beginning inference with metrics"
        if collect_metrics
        else "Beginning inference without metrics"
    )
    logging.info(log_msg)
    if runtime_only_benchmark:
        logging.info(
            "Runtime-only benchmark mode enabled: prediction CSV writing and SELD evaluation "
            "will be skipped."
        )
    if collect_metrics and runtime_measurement_enabled(inference_config):
        measurement_cfg = runtime_measurement_summary(inference_config)
        logging.info(
            "Steady-state runtime measurement enabled: "
            "latency warmup=%d runs=%d, memory warmup=%d runs=%d, poll_interval_ms=%.3f",
            int(measurement_cfg["latency_warmup_runs"]),
            int(measurement_cfg["latency_measurement_runs"]),
            int(measurement_cfg["memory_warmup_runs"]),
            int(measurement_cfg["memory_measurement_runs"]),
            float(measurement_cfg["memory_poll_interval_s"]) * 1000.0,
        )

    with torch.no_grad(), logging_redirect_tqdm():
        n_done = 0
        for batch in tqdm(data_loader, ncols=100, desc="Processing", total=total_items):
            if len(batch["file_id"]) != 1:
                raise ValueError("Inference currently supports batch_size=1 only.")
            file_id = batch["file_id"][0]
            audio_np = batch["audio"].cpu().numpy()[0].astype(np.float32)
            sample_rate = int(batch["sample_rate"][0])
            audio_np, full_resolution_audio_np, audio_prep = prepare_audio_for_inference(
                audio_np,
                sample_rate=sample_rate,
                inference_config=inference_config,
                input_channel_indices=args.input_channel_indices,
            )
            fs = int(audio_prep["target_sample_rate"])

            # Compute visibility matrix from audio
            with warnings.catch_warnings(record=True) as visibility_warning_records:
                warnings.simplefilter("always")
                S_in, _ = get_visibility_matrix(
                    audio_in=audio_np,
                    fs=fs,
                    apgd=False,
                    nbands=9,
                    T_sti=visibility_t_sti,
                    diagonal_loading=float(
                        inference_config.get("visibility_diagonal_loading", 0.0) or 0.0
                    ),
                    eigenvalue_floor=float(
                        inference_config.get("visibility_eigenvalue_floor", 0.0) or 0.0
                    ),
                )
                reference_csm = None
                if collect_metrics:
                    if (
                        inference_config["model_name"] == "VariableSRCNNLAM"
                        and audio_prep["full_resolution_num_channels"] != CANONICAL_CHANNELS
                    ):
                        reference_csm = None
                    elif (
                        audio_prep["prepared_num_channels"]
                        == audio_prep["full_resolution_num_channels"]
                    ):
                        reference_csm = np.transpose(S_in, (1, 0, 2, 3))
                    else:
                        S_reference, _ = get_visibility_matrix(
                            audio_in=full_resolution_audio_np,
                            fs=fs,
                            apgd=False,
                            nbands=9,
                            T_sti=visibility_t_sti,
                            diagonal_loading=float(
                                inference_config.get("visibility_diagonal_loading", 0.0) or 0.0
                            ),
                            eigenvalue_floor=float(
                                inference_config.get("visibility_eigenvalue_floor", 0.0) or 0.0
                            ),
                        )
                        reference_csm = np.transpose(S_reference, (1, 0, 2, 3))
            visibility_warnings = [
                f"{warning.category.__name__}: {warning.message}"
                for warning in visibility_warning_records
            ]
            if visibility_warnings:
                logging.warning(
                    "Visibility warnings for %s: %s",
                    file_id,
                    "; ".join(visibility_warnings[:3]),
                )
            channel_desc = (
                str(audio_prep["selected_channel_indices"])
                if audio_prep["selected_channel_indices"] is not None
                else f"all {audio_prep['original_num_channels']} channels"
            )
            logging.info(
                "Prepared %s file %s: sr %d -> %d, channels %d -> %d, "
                "selected channels=%s, CSM shape=%s",
                inference_config["data_set"],
                file_id,
                audio_prep["original_sample_rate"],
                audio_prep["target_sample_rate"],
                audio_prep["original_num_channels"],
                audio_prep["prepared_num_channels"],
                channel_desc,
                tuple(S_in.shape),
            )
            # Keep the model input layout stable across variants (contiguous)
            S_in_t = torch.from_numpy(S_in).to(device).permute(1, 0, 2, 3).contiguous()
            observed_indices_t = None
            if inference_config["model_name"] == "VariableSRCNNLAM":
                observed_indices_t = torch.as_tensor(
                    audio_prep["selected_channel_indices"], dtype=torch.long, device=device
                )
                S_in_t = embed_observed_csm(S_in_t, observed_indices_t)

            # Perform inference (with optional metrics collection)
            if collect_metrics:
                result = (
                    model(S_in_t, observed_indices_t, collect_metrics=True)
                    if observed_indices_t is not None
                    else model(S_in_t, collect_metrics=True)
                )
                _, I_pred, metrics = result

                # For LAM only inference, upsampler metrics are not applicable.
                if inference_config["model_name"] == "LAM":
                    num_frames = int(metrics.get("num_frames", 0) or 0)
                    lam_total_time_ms = float(metrics.get("total_time_ms", 0.0) or 0.0)
                    lam_flops = float(metrics.get("flops", 0.0) or 0.0)
                    lam_memory_mb = float(metrics.get("memory_mb", 0.0) or 0.0)

                    metrics["lam_total_time_ms"] = lam_total_time_ms
                    metrics["lam_flops"] = lam_flops
                    metrics["lam_memory_mb"] = lam_memory_mb
                    metrics["upsampler_time_ms"] = 0.0
                    metrics["upsampler_flops"] = 0.0
                    metrics["upsampler_memory_mb"] = 0.0
                    metrics["total_flops"] = lam_flops
                    metrics["total_memory_mb"] = lam_memory_mb
                    if num_frames > 0 and "latency_per_frame_ms" not in metrics:
                        metrics["latency_per_frame_ms"] = lam_total_time_ms / num_frames

                apply_steady_state_runtime_metrics(
                    metrics,
                    model=model,
                    model_name=str(inference_config["model_name"]),
                    input_tensor=S_in_t,
                    device=device,
                    inference_config=inference_config,
                    observed_channel_indices=observed_indices_t,
                )

                stage_outputs = _extract_csm_stage_outputs(model)
                upsampler_output = stage_outputs.get("upsampler")
                projected_upsampler_output = None
                if upsampler_output is not None:
                    add_upsampler_validity_metrics(metrics, upsampler_output)
                    projected_upsampler_output = project_to_hermitian_psd(upsampler_output)
                    metrics["upsampler_cmd_psd_projection_applied"] = True
                reference_csm_t = None
                if reference_csm is not None:
                    reference_csm_t = torch.from_numpy(reference_csm).to(
                        device=S_in_t.device,
                        dtype=torch.complex128,
                    )
                    add_cmd_metrics(
                        metrics,
                        "cmd_reference_to_upsampler",
                        reference_csm_t,
                        projected_upsampler_output,
                    )
                    add_cmd_metrics(
                        metrics,
                        "cmd_reference_to_lam",
                        reference_csm_t,
                        stage_outputs.get("lam_final"),
                    )
                    for stage_index in range(1, 5):
                        add_cmd_metrics(
                            metrics,
                            f"cmd_reference_to_lam_denoise{stage_index}",
                            reference_csm_t,
                            stage_outputs.get(f"lam_denoise{stage_index}"),
                        )
                add_cmd_metrics(
                    metrics,
                    "cmd_upsampler_to_lam",
                    projected_upsampler_output,
                    stage_outputs.get("lam_final"),
                )

                metrics["file_id"] = str(file_id)
                metrics["total_params"] = total_params
                all_metrics.append(metrics)
            else:
                result = (
                    model(S_in_t, observed_indices_t, collect_metrics=False)
                    if observed_indices_t is not None
                    else model(S_in_t, collect_metrics=False)
                )
                # Old models may return just the prediction without metrics. Handle both cases.
                if isinstance(result, tuple) and len(result) == MODEL_RESULT_WITH_AUX_LEN:
                    _, I_pred, _ = result
                else:
                    _, I_pred = result

            if not runtime_only_benchmark:
                I_pred = I_pred.cpu().detach().numpy()
                I_pred_sq = np.square(I_pred)  # (frames, bands, pixels)
                I_pred = I_pred_sq.sum(axis=1)  # (frames, pixels)
                frame_predictions = cluster_intensity_maps(
                    I_pred,
                    inference_config,
                    band_maps=I_pred_sq,
                )

                output_filename = write_output_dcase_csv(
                    I_pred,
                    inference_config,
                    file_id,
                    timestamp,
                    frame_predictions=frame_predictions,
                )
                processed_file_ids.append((output_filename, str(file_id)))
            n_done += 1

            if inference_config["max_files"] > 0 and n_done >= inference_config["max_files"]:
                break

    logging.info(f"Inference completed. Processed {n_done} files.")

    if log_filename:
        logging.info(f"The logs are saved in: {Path(log_filename).resolve()}")
    logging.info(f"The file's outputs are saved in: {output_path.resolve()}")

    # Save metrics if collected
    if collect_metrics and all_metrics:
        if not runtime_only_benchmark:
            gt_loader = _build_ground_truth_loader(
                inference_config["data_set"],
                dataset_config,
                inference_config,
                locata_tasks,
            )

            num_classes = (
                1
                if inference_config["class_agnostic_evaluation"]
                else inference_config["num_classes"]
            )

            pred_file_ids = [pred_id for pred_id, _ in processed_file_ids]
            file_id_mapping = {pred_id: gt_id for pred_id, gt_id in processed_file_ids}

            ER, F, LE, LR, seld_score, _ = compute_seld_metrics_for_files(
                pred_files_path=output_path,
                gt_loader=gt_loader,
                file_ids=pred_file_ids,
                num_classes=num_classes,
                doa_threshold=inference_config["doa_threshold_deg"],
                average=inference_config["seld_average"],
                use_polar_format=True,
                class_agnostic=inference_config["class_agnostic_evaluation"],
                file_id_mapping=file_id_mapping,
            )

            all_metrics.append(
                {
                    "error_rate": ER,
                    "f_score": F,
                    "localisation_error": LE,
                    "localisation_recall": LR,
                    "seld_score": seld_score,
                }
            )

        # Save metrics to JSON
        metrics_path = Path(output_path).joinpath(f"metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logging.info(f"Metrics saved to: {metrics_path.resolve()}")

        if runtime_only_benchmark:
            logging.info("Skipping terminal metrics summary for the runtime-only benchmark pass.")
        else:
            summary_output = _print_metrics_summary(
                all_metrics,
                inference_config,
                n_done,
                str(device),
            )
            print(summary_output)


if __name__ == "__main__":
    main()
