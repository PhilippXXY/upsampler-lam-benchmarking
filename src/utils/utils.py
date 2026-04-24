"""Utility functions for LAM and upsampler benchmarking."""

import csv
import logging
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.signal import resample_poly

from lam_min.trainer.kmeans import cluster_sequence, get_kmeans_clusters
from lam_min.util.utils import convert_polar_to_cartesian, get_field
from utils.benchmarking import apply_benchmark_audio_length
from utils.cmd_metrics import (
    aggregate_all_cmd_global_medians,
)

DEFAULT_LOCATA_LOW_CHANNEL_INDICES = (5, 9, 21, 25)
LOCATA_FOUR_CHANNEL_MODELS = {
    "UpLAM",
    "BicubicLAM",
    "SRCNNLAM",
    "IMDNLAM",
    "SAFMNLAM",
    "GANLAM",
    "AINNLAM",
}
TWO_D_AUDIO_NDIM = 2
LOCATA_RAW_NUM_CHANNELS = 32
LOCATA_LOW_CHANNEL_COUNT = 4


def resolve_requested_device(
    requested_device: str,
    *,
    mps_fallback_reason: str | None = None,
) -> torch.device:
    """
    Resolve the requested runtime device.

    Parameters
    ----------
    requested_device : str
        Requested device string, for example ``cpu``, ``mps``, or ``cuda``.
    mps_fallback_reason : str | None, optional
        Optional reason for forcing ``mps`` requests back to ``cpu``.

    Returns
    -------
    torch.device
        Resolved device to use.
    """
    requested = requested_device.strip().lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            logging.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        if mps_fallback_reason:
            logging.warning("%s Falling back to CPU on Apple Silicon.", mps_fallback_reason)
            return torch.device("cpu")
        return torch.device("mps")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logging.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{requested_device}'. Use one of: cpu, mps, cuda")


def visibility_t_sti_seconds(frame_width_ms: float) -> float:
    """
    Convert target output frame width to the frontend STI window.

    The visibility frontend aggregates ``10`` short-time integrations into one
    stationarity block, so the short-time window must be one tenth of the
    requested output frame duration.

    Parameters
    ----------
    frame_width_ms : float
        Desired output frame width in milliseconds.

    Returns
    -------
    float
        Short-time integration window in seconds.

    Raises
    ------
    ValueError
        If ``frame_width_ms`` is not positive.
    """
    if frame_width_ms <= 0:
        raise ValueError(f"frame_width_ms must be positive, got {frame_width_ms!r}")
    return frame_width_ms / 1000.0 / 10.0


def locata_model_input_channels(model_name: str) -> int:
    """
    Return the expected LOCATA microphone count for a model.

    Parameters
    ----------
    model_name : str
        Inference model name from the config.

    Returns
    -------
    int
        Expected channel count for the model on LOCATA.

    Raises
    ------
    ValueError
        If the model name is unsupported for LOCATA inference.
    """
    if model_name == "LAM":
        return LOCATA_RAW_NUM_CHANNELS
    if model_name in LOCATA_FOUR_CHANNEL_MODELS:
        return LOCATA_LOW_CHANNEL_COUNT
    raise ValueError(f"Unsupported model '{model_name}' for LOCATA inference.")


def resample_audio_tc(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    """
    Resample time-major audio from ``orig_sr`` to ``target_sr``.

    The goal of this function is to resample LOCATA audio while preserving the number of frames
    when using a fixed frame width (e.g., 100 ms). This is achieved by using
    `scipy.signal.resample_poly` with integer up/down factors derived from the greatest common
    divisor of the original and target sample rates.

    Parameters
    ----------
    audio : np.ndarray
        Audio array with shape ``(T, C)``.
    orig_sr : int
        Original sample rate in Hz.
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    np.ndarray
        Resampled audio with shape ``(T_resampled, C)``.

    Raises
    ------
    ValueError
        If the audio shape or sampling rates are invalid.
    """
    if audio.ndim != TWO_D_AUDIO_NDIM:
        raise ValueError(f"Expected audio with shape (T, C), got {audio.shape}")
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Invalid sample rates orig={orig_sr}, target={target_sr}")
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    # Get the greatest common divisor to determine up/down factors for resample_poly
    divisor = math.gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    resampled = resample_poly(audio, up=up, down=down, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def prepare_audio_for_inference(
    audio: NDArray[np.float32],
    sample_rate: int,
    inference_config: dict[str, Any],
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    """
    Resample and select channels for inference.

    Works for both LOCATA (always 32 channels) and STARSS (4 or 32 channels).
    When the input has 32 channels and the model expects 4, the 4-channel subset
    is selected using ``locata_low_channel_indices``.  When the input already
    matches the expected channel count it is passed through unchanged.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio with shape ``(T, C)`` where ``C`` is 4 or 32.
    sample_rate : int
        Original sample rate in Hz.
    inference_config : dict[str, Any]
        Inference configuration containing the model name, target sampling rate,
        benchmark duration settings, and optional low-channel indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        Prepared model-input audio, prepared full-resolution audio before any
        channel reduction, and metadata describing the preprocessing.

    Raises
    ------
    ValueError
        If the audio shape, channel count, or configured channel indices are
        invalid for the requested model.
    """
    audio_tc = np.asarray(audio, dtype=np.float32)
    if audio_tc.ndim != TWO_D_AUDIO_NDIM:
        raise ValueError(f"Expected audio with shape (T, C), got {audio_tc.shape}")

    num_input_channels = audio_tc.shape[1]
    model_name = str(inference_config["model_name"])
    target_sr = int(inference_config["sampling_rate"])

    prepared_audio = resample_audio_tc(audio_tc, orig_sr=int(sample_rate), target_sr=target_sr)
    prepared_audio = apply_benchmark_audio_length(
        prepared_audio,
        sample_rate=target_sr,
        inference_config=inference_config,
    )
    full_resolution_audio = np.asarray(prepared_audio, dtype=np.float32, copy=False)

    expected_channels = locata_model_input_channels(model_name)
    selected_channel_indices: tuple[int, ...] | None = None

    if (
        num_input_channels == LOCATA_RAW_NUM_CHANNELS
        and expected_channels == LOCATA_LOW_CHANNEL_COUNT
    ):
        configured = tuple(
            int(index)
            for index in inference_config.get(
                "locata_low_channel_indices",
                DEFAULT_LOCATA_LOW_CHANNEL_INDICES,
            )
        )
        if len(configured) != LOCATA_LOW_CHANNEL_COUNT:
            raise ValueError(
                "locata_low_channel_indices must contain exactly "
                f"{LOCATA_LOW_CHANNEL_COUNT} channel indices for "
                f"{model_name}, got {configured}"
            )
        if min(configured) < 0 or max(configured) >= num_input_channels:
            raise ValueError(
                f"locata_low_channel_indices {configured} are out of range for "
                f"{num_input_channels}-channel audio"
            )
        prepared_audio = prepared_audio[:, list(configured)]
        selected_channel_indices = configured

    if prepared_audio.shape[1] != expected_channels:
        raise ValueError(
            f"Prepared audio has {prepared_audio.shape[1]} channels, "
            f"expected {expected_channels} for {model_name}"
        )

    return (
        prepared_audio,
        full_resolution_audio,
        {
            "original_sample_rate": int(sample_rate),
            "target_sample_rate": target_sr,
            "original_num_channels": int(audio_tc.shape[1]),
            "full_resolution_num_channels": int(full_resolution_audio.shape[1]),
            "prepared_num_channels": int(prepared_audio.shape[1]),
            "selected_channel_indices": selected_channel_indices,
        },
    )


def cluster_intensity_maps(
    intensity_maps: NDArray[np.float32] | NDArray[np.float64],
    inference_config: dict[str, Any],
    band_maps: NDArray[np.float32] | NDArray[np.float64] | None = None,
) -> list[list[dict[str, float | int]]]:
    """
    Cluster frame-wise intensity maps into DoA predictions.

    Parameters
    ----------
    intensity_maps : np.ndarray
        Array of shape (n_frames, ...) containing intensity predictions for each frame.
    inference_config : dict[str, Any]
        Inference configuration containing clustering parameters such as:
            - "n_max": Maximum number of clusters to consider.
            - "max_sources": Maximum number of sources to predict per frame.
            - "intensity_threshold": Minimum intensity threshold for cluster consideration.
            - "adaptive_k": Whether to use adaptive clustering based on intensity distribution.
            - "peak_ratio_threshold": Minimum ratio of cluster peak intensity to global max for
                consideration.
    band_maps : np.ndarray or None, optional
        Per-band squared intensity maps of shape ``(n_frames, n_bands, n_pixels)``.
        When provided and ``adaptive_k`` is true, used for band-support scoring.

    Returns
    -------
    list[list[dict[str, float | int]]]
        A list of frame-wise predictions, where each frame's predictions are a list of dictionaries
        containing:
            - "source_index": Integer index of the predicted source.
            - "azimuth": Float azimuth angle in degrees.
            - "elevation": Float elevation angle in degrees.
    """
    field = get_field()

    if inference_config.get("adaptive_k", True):
        return cluster_sequence(  # type: ignore[no-any-return]
            intensity_maps,
            field,
            max_sources=inference_config["max_sources"],
            merge_radius_deg=inference_config.get("merge_radius_deg", 25.0),
            candidate_peak_ratio=inference_config.get("candidate_peak_ratio", 0.3),
            candidate_mass_ratio=inference_config.get("candidate_mass_ratio", 0.05),
            track_max_jump_deg=inference_config.get("track_max_jump_deg", 20.0),
            track_max_gap=inference_config.get("track_max_gap", 3),
            track_min_frames=inference_config.get("track_min_frames", 3),
            track_min_active_ratio=inference_config.get("track_min_active_ratio", 0.03),
            track_min_peak_rel=inference_config.get("track_min_peak_rel", 0.0),
            band_maps=band_maps,
            band_peak_ratio=inference_config.get("band_peak_ratio", 0.2),
            activity_threshold=inference_config.get("activity_threshold", 0.01),
        )

    # adaptive_k=False: fixed K=1 single-source mode
    frame_predictions: list[list[dict[str, float | int]]] = []
    for frame_idx in range(intensity_maps.shape[0]):
        lon, lat = get_kmeans_clusters(
            intensity_maps[frame_idx],
            field,
            N_max=inference_config["n_max"],
            max_sources=1,
            intensity_threshold=inference_config["intensity_threshold"],
            adaptive_k=False,
            peak_ratio_threshold=inference_config["peak_ratio_threshold"],
        )
        frame_predictions.append(
            [
                {
                    "source_index": int(source_index),
                    "azimuth": float(lon[source_index]),
                    "elevation": float(lat[source_index]),
                }
                for source_index in range(len(lon))
            ]
        )

    return frame_predictions


def select_memory_display_unit(memory_values_mb: list[float]) -> tuple[float, str]:
    """
    Choose a readable binary memory unit for display from MiB-valued inputs.

    Parameters
    ----------
    memory_values_mb : list[float]
        A list of memory values in MB to analyze for selecting an appropriate display unit.

    Returns
    -------
    tuple[float, str]
        A tuple containing:
        - scale_factor: The factor to multiply the original MB values by to convert to the
            selected unit.
        - unit_name: The name of the selected memory unit (e.g. "MiB", "GiB").
    """
    kib_per_mib = 1024.0
    finite_values = [abs(value) for value in memory_values_mb if not math.isnan(value)]
    max_value_mb = max(finite_values) if finite_values else 0.0
    if max_value_mb >= kib_per_mib:
        return 1.0 / kib_per_mib, "GiB"
    if max_value_mb >= 1.0:
        return 1.0, "MiB"
    if max_value_mb >= 1.0 / kib_per_mib:
        return kib_per_mib, "KiB"
    return kib_per_mib * kib_per_mib, "B"


def _print_metrics_summary(  # noqa: C901, PLR0915
    metrics_list: list[dict[str, Any]],
    inference_config: dict[str, Any],
    n_files: int,
    device: str,
) -> str:
    """
    Generate a formatted summary of computational metrics as a string.

    Parameters
    ----------
    metrics_list : list[dict[str, Any]]
        List of metrics dictionaries collected during inference.
    inference_config : dict[str, Any]
        Configuration dictionary with dataset information.
    n_files : int
        Number of files processed.
    device : str
        Device used for inference (e.g., 'cpu' or 'cuda').

    Returns
    -------
    str
        Formatted string containing all metrics tables and summary information.
    """
    from prettytable import PrettyTable  # noqa: PLC0415

    def _mean_or_nan(values: list[float]) -> float:
        """Return the arithmetic mean or ``nan`` when the input list is empty."""
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _median_and_max(values: list[float]) -> tuple[float, float]:
        """Return the median and worst-case value for a list of finite samples."""
        finite_values = [float(value) for value in values if not math.isnan(float(value))]
        if not finite_values:
            return float("nan"), float("nan")
        return float(statistics.median(finite_values)), float(max(finite_values))

    def _format_optional(
        value: float,
        template: str,
        *,
        nan_text: str = "N/A",
    ) -> str:
        """Format a floating-point value and fall back to ``nan_text`` when unavailable."""
        if math.isnan(value):
            return nan_text
        return template.format(value=value)

    output_lines: list[str] = []
    runtime_metrics = [m for m in metrics_list if "file_id" in m]
    if not runtime_metrics:
        runtime_metrics = [m for m in metrics_list if "num_frames" in m]
    cmd_medians = aggregate_all_cmd_global_medians(runtime_metrics)

    # Evaluation metrics
    seld_score = [m.get("seld_score", 0) for m in metrics_list if "seld_score" in m]
    f_score = [m.get("f_score", 0) for m in metrics_list if "f_score" in m]
    error_rate = [m.get("error_rate", 0) for m in metrics_list if "error_rate" in m]
    localisation_error = [
        m.get("localisation_error", 0) for m in metrics_list if "localisation_error" in m
    ]
    localisation_recall = [
        m.get("localisation_recall", 0) for m in metrics_list if "localisation_recall" in m
    ]

    # Calculate statistics
    total_frames = sum(m.get("num_frames", 0) for m in runtime_metrics)
    frame_duration_ms = inference_config.get("frame_width_ms", 100.0)
    total_audio_sec = total_frames * frame_duration_ms / 1000.0

    # LAM metrics
    lam_latencies = [m.get("lam_total_time_ms", 0) for m in runtime_metrics]
    lam_total_latency = sum(lam_latencies)
    lam_per_frame_latency_ms = lam_total_latency / total_frames if total_frames > 0 else 0.0
    lam_total_flops = sum(m.get("lam_flops", 0) for m in runtime_metrics)
    lam_total_gflops = lam_total_flops / 1e9
    lam_per_frame_gflops = lam_total_flops / total_frames / 1e9 if total_frames > 0 else 0.0
    lam_memory_samples_mb = [float(m.get("lam_memory_mb", float("nan"))) for m in runtime_metrics]
    lam_memory_median_mb, lam_memory_max_mb = _median_and_max(lam_memory_samples_mb)

    # Upsampler metrics
    upsampler_latencies = [m.get("upsampler_time_ms", 0) for m in runtime_metrics]
    upsampler_total_latency = sum(upsampler_latencies)
    upsampler_per_frame_latency_ms = (
        upsampler_total_latency / total_frames if total_frames > 0 else 0.0
    )
    upsampler_total_flops = sum(m.get("upsampler_flops", 0) for m in runtime_metrics)
    upsampler_total_gflops = upsampler_total_flops / 1e9
    upsampler_per_frame_gflops = (
        upsampler_total_flops / total_frames / 1e9 if total_frames > 0 else 0.0
    )
    upsampler_memory_samples_mb = [
        float(m.get("upsampler_memory_mb", float("nan"))) for m in runtime_metrics
    ]
    upsampler_memory_median_mb, upsampler_memory_max_mb = _median_and_max(
        upsampler_memory_samples_mb
    )
    total_latency_ms = sum(m.get("total_time_ms", 0) for m in runtime_metrics)
    total_per_frame_latency_ms = total_latency_ms / total_frames if total_frames > 0 else 0.0
    total_flops = sum(m.get("total_flops", 0) for m in runtime_metrics)
    total_per_frame_gflops = total_flops / total_frames / 1e9 if total_frames > 0 else 0.0
    total_memory_samples_mb = [
        float(m.get("total_memory_mb", float("nan"))) for m in runtime_metrics
    ]
    total_memory_median_mb, total_memory_max_mb = _median_and_max(total_memory_samples_mb)

    total_params = next(
        (int(m["total_params"]) for m in runtime_metrics if "total_params" in m),
        None,
    )

    memory_scale, memory_unit = select_memory_display_unit(
        [
            total_memory_median_mb,
            total_memory_max_mb,
            lam_memory_median_mb,
            lam_memory_max_mb,
            upsampler_memory_median_mb,
            upsampler_memory_max_mb,
        ]
    )

    # Build header
    separator = "=" * 150
    output_lines.append("")
    output_lines.append(separator)
    output_lines.append("EVALUATION SUMMARY")
    output_lines.append(separator)
    output_lines.append("")

    # Prepare data for each section
    dataset_name = inference_config.get("data_set", "Unknown")

    # Build one comprehensive table with all metrics
    main_table = PrettyTable()
    main_table.field_names = [
        "Summary",
        "Values",
        "Evaluation Metrics",
        " Values",
        "LAM Metrics",
        "  Values",
        "Upsampler Metrics",
        "   Values",
    ]

    # Set alignment for all columns
    main_table.align["Summary"] = "l"
    main_table.align["Values"] = "r"
    main_table.align["Evaluation Metrics"] = "l"
    main_table.align[" Values"] = "r"
    main_table.align["LAM Metrics"] = "l"
    main_table.align["  Values"] = "r"
    main_table.align["Upsampler Metrics"] = "l"
    main_table.align["   Values"] = "r"

    # Prepare all rows of data
    summary_data = [
        ["Dataset", dataset_name.upper()],
        ["Files processed", str(n_files)],
        ["Frame duration", f"{frame_duration_ms} ms"],
        ["Total frames", str(total_frames)],
        ["Total audio", f"{total_audio_sec:.1f} s"],
        ["Device", device.upper()],
        ["Total model latency", f"{total_per_frame_latency_ms:.5f} ms/frame"],
        ["Total model GFLOPs", f"{total_per_frame_gflops:.6f} /frame"],
        [
            "Parameters",
            f"{total_params / 1e6:.3f} M" if total_params is not None else "N/A",
        ],
        [
            "Peak memory (median)",
            _format_optional(
                total_memory_median_mb * memory_scale,
                "{value:.4f} " + memory_unit,
            ),
        ],
        [
            "Peak memory (worst case)",
            _format_optional(
                total_memory_max_mb * memory_scale,
                "{value:.4f} " + memory_unit,
            ),
        ],
    ]

    eval_data = [
        ["SELD score", _format_optional(_mean_or_nan(seld_score), "{value:.4f}")],
        ["F-score", _format_optional(_mean_or_nan(f_score) * 100.0, "{value:.2f} %")],
        ["Error rate", _format_optional(_mean_or_nan(error_rate), "{value:.2f}")],
        [
            "Localisation error",
            _format_optional(_mean_or_nan(localisation_error), "{value:.2f} °"),
        ],
        [
            "Localisation recall",
            _format_optional(_mean_or_nan(localisation_recall) * 100.0, "{value:.2f} %"),
        ],
    ]

    lam_data = [
        ["Total Latency", f"{lam_total_latency:.3f} ms"],
        ["Per-Frame Latency", f"{lam_per_frame_latency_ms:.5f} ms/frame"],
        ["Total GFLOPs", f"{lam_total_gflops:.3f}"],
        ["Per-Frame GFLOPs", f"{lam_per_frame_gflops:.6f} /frame"],
        [
            "Peak memory (median)",
            _format_optional(
                lam_memory_median_mb * memory_scale,
                "{value:.4f} " + memory_unit,
            ),
        ],
        [
            "Peak memory (worst case)",
            _format_optional(
                lam_memory_max_mb * memory_scale,
                "{value:.4f} " + memory_unit,
            ),
        ],
    ]

    upsampler_data = [
        ["Total Latency", f"{upsampler_total_latency:.3f} ms"],
        ["Per-Frame Latency", f"{upsampler_per_frame_latency_ms:.5f} ms/frame"],
        ["Total GFLOPs", f"{upsampler_total_gflops:.3f}"],
        ["Per-Frame GFLOPs", f"{upsampler_per_frame_gflops:.6f} /frame"],
        [
            "Peak memory (median)",
            _format_optional(
                upsampler_memory_median_mb * memory_scale,
                "{value:.4f} " + memory_unit,
            ),
        ],
        [
            "Peak memory (worst case)",
            _format_optional(
                upsampler_memory_max_mb * memory_scale,
                "{value:.4f} " + memory_unit,
            ),
        ],
    ]

    # Find the maximum number of rows needed
    max_rows = max(len(summary_data), len(eval_data), len(lam_data), len(upsampler_data))

    # Pad shorter sections with empty rows
    for data_list in [summary_data, eval_data, lam_data, upsampler_data]:
        while len(data_list) < max_rows:
            data_list.append(["", ""])

    # Add all rows to the table
    for i in range(max_rows):
        main_table.add_row(
            [
                summary_data[i][0],
                summary_data[i][1],
                eval_data[i][0],
                eval_data[i][1],
                lam_data[i][0],
                lam_data[i][1],
                upsampler_data[i][0],
                upsampler_data[i][1],
            ]
        )

    output_lines.append(str(main_table))
    output_lines.append("")

    cmd_rows = [
        (
            "Reference -> Upsampler",
            cmd_medians["cmd_reference_to_upsampler_median"],
        ),
        (
            "Upsampler -> LAM",
            cmd_medians["cmd_upsampler_to_lam_median"],
        ),
        ("Reference -> LAM", cmd_medians["cmd_reference_to_lam_median"]),
        ("Reference -> LAM Denoise1", cmd_medians["cmd_reference_to_lam_denoise1_median"]),
        ("Reference -> LAM Denoise2", cmd_medians["cmd_reference_to_lam_denoise2_median"]),
        ("Reference -> LAM Denoise3", cmd_medians["cmd_reference_to_lam_denoise3_median"]),
        ("Reference -> LAM Denoise4", cmd_medians["cmd_reference_to_lam_denoise4_median"]),
    ]
    finite_cmd_rows = [(label, value) for label, value in cmd_rows if not math.isnan(float(value))]
    if finite_cmd_rows:
        cmd_table = PrettyTable()
        cmd_table.field_names = ["CMD Metrics", " Values"]
        cmd_table.align["CMD Metrics"] = "l"
        cmd_table.align[" Values"] = "r"
        for label, value in finite_cmd_rows:
            cmd_table.add_row([label, f"{value:.6f}"])
        output_lines.append("CMD SUMMARY")
        output_lines.append(str(cmd_table))
        output_lines.append("")

    return "\n".join(output_lines)


def write_output_dcase_csv(
    I_pred: np.ndarray[Any, np.dtype[np.float64]],
    inference_config: dict[str, Any],
    file_id: Path,
    timestamp: str,
    frame_predictions: list[list[dict[str, float | int]]] | None = None,
) -> str:
    """
    Write predicted direction of arrival (DoA) estimates to a CSV file in DCASE format.

    This function processes intensity predictions, clusters them to identify sound source
    locations, and outputs the results as a CSV file with frame-wise DoA estimates.

    Parameters
    ----------
    I_pred: np.ndarray[Any, np.dtype[np.float64]]
        Numpy array of predicted intensity values with shape (n_frames, ...).
    inference_config: dict[str, Any]
        Dictionary containing configuration parameters including:
            - "output_path": Output directory path (optional).
            - "n_max_sources": Maximum number of sources to detect per frame.
            - "intensity_threshold": Threshold for intensity-based filtering.
            - "adaptive_k": Whether to use adaptive clustering.
            - "peak_ratio_threshold": Threshold for peak ratio filtering.
            - "frame_width_ms": Duration of each frame in milliseconds.
    file_id: Path
        Identifier for the output CSV file.
    timestamp: str
        Timestamp string to append to the output filename.
    frame_predictions : list[list[dict[str, float | int]]] | None, optional
        Precomputed per-frame DoA predictions. If omitted, they are computed from ``I_pred``.

    Returns
    -------
    str
        The output filename (without extension) that was written.
    """
    output_dict: dict[int, list[list[float]]] = {}
    frame_offset = 0
    predictions = frame_predictions or cluster_intensity_maps(I_pred, inference_config)

    output_filename = f"{file_id}_{timestamp}"
    if inference_config["output_path"] is not None:
        base_output_path = Path(inference_config["output_path"])
        date_str = timestamp
        run_id = str(inference_config.get("model_variant", inference_config["model_name"]))
        subdir = f"{run_id}-{date_str}"
        output_path = base_output_path.joinpath(subdir, f"{output_filename}.csv")
    else:
        output_path = Path(f"output/{output_filename}.csv")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(I_pred.shape[0]):
        frame_index = i + frame_offset
        output_dict[frame_index] = []
        for prediction in predictions[i]:
            output_dict[frame_index].append(
                [float(prediction["azimuth"]), float(prediction["elevation"])]
            )

    # Write to CSV in DCASE format: frame, class_idx, source_id, azimuth, elevation
    # Using class_idx=0 for class-agnostic predictions
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(I_pred.shape[0]):
            frame_index = i + frame_offset
            # Iterate through number of predicted DoAs
            for j in range(len(output_dict[frame_index])):
                x, y, z = convert_polar_to_cartesian(
                    output_dict[frame_index][j][0],
                    output_dict[frame_index][j][1],
                )
                # DCASE format: frame_idx, class_idx, track_id, x, y, z
                row = [frame_index, 0, j, x, y, z]
                csv_writer.writerow(row)

    return output_filename


def starss_selection_group(file_id: str) -> str:
    """
    Derive a STARSS grouping key for representative file sampling.

    Files are grouped by recording prefix, e.g. `fold4_room23`, so a
    stratified sample does not collapse onto the first room in sorted order.

    Parameters
    ----------
    file_id : str
        The file ID (stem of the filename) to derive the group from.

    Returns
    -------
    str
        The derived group key for the given file ID.
    """
    if "_mix" in file_id:
        return file_id.rsplit("_mix", 1)[0]
    return file_id


def locata_selection_group(file_id: str) -> str:
    """
    Derive a LOCATA grouping key for stratified file selection.

    Parameters
    ----------
    file_id : str
        LOCATA file ID, e.g. ``task3_recording2``.

    Returns
    -------
    str
        Task-level grouping key for the file.
    """
    if "_" in file_id:
        return file_id.split("_", 1)[0]
    return file_id


def seed_everything(seed: int) -> torch.Generator:
    """
    Set deterministic NumPy and torch seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value to set for NumPy and torch.

    Returns
    -------
    torch.Generator
        A seeded torch generator for deterministic torch sampling calls.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def permute_items(items: list[Any], generator: torch.Generator) -> list[Any]:
    """
    Return a deterministically permuted copy of a list using torch RNG.

    Parameters
    ----------
    items : list[Any]
        The list of items to permute.
    generator : torch.Generator
        A seeded torch generator to ensure reproducibility of the permutation.

    Returns
    -------
    list[Any]
        A new list containing the items in a permuted order.
    """
    if len(items) <= 1:
        return list(items)
    indices = torch.randperm(len(items), generator=generator).tolist()
    return [items[index] for index in indices]


def select_stratified_file_ids(
    file_ids: list[str],
    max_files: int,
    generator: torch.Generator,
    group_fn: Callable[[str], str],
) -> list[str]:
    """
    Select file IDs round-robin across dataset-specific groups.

    Parameters
    ----------
    file_ids : list[str]
        Available file IDs.
    max_files : int
        Maximum number of file IDs to select.
    generator : torch.Generator
        Seeded generator for deterministic ordering.
    group_fn : Callable[[str], str]
        Function mapping a file ID to its stratification group.

    Returns
    -------
    list[str]
        Selected file IDs, sorted lexicographically.
    """
    grouped: dict[str, list[str]] = {}
    for file_id in file_ids:
        grouped.setdefault(group_fn(file_id), []).append(file_id)

    group_keys = permute_items(sorted(grouped), generator)
    for key in group_keys:
        grouped[key] = permute_items(sorted(grouped[key]), generator)

    selected: list[str] = []
    while len(selected) < max_files:
        added_in_round = False
        for key in group_keys:
            if not grouped[key]:
                continue
            selected.append(grouped[key].pop(0))
            added_in_round = True
            if len(selected) >= max_files:
                break
        if not added_in_round:
            break

    return sorted(selected)


def select_stratified_wavs(
    wavs: list[Path], max_files: int, generator: torch.Generator
) -> list[Path]:
    """
    Select files round-robin across STARSS recording groups.

    Parameters
    ----------
    wavs : list[Path]
        List of available WAV file paths.
    max_files : int
        Maximum number of files to select.
    generator : torch.Generator
        Seeded torch generator used to randomise group order and within-group order.

    Returns
    -------
    list[Path]
        List of selected WAV file paths, sorted by filename.

    """
    wav_by_id = {wav_path.stem: wav_path for wav_path in wavs}
    selected_ids = select_stratified_file_ids(
        file_ids=list(wav_by_id),
        max_files=max_files,
        generator=generator,
        group_fn=starss_selection_group,
    )
    return [wav_by_id[file_id] for file_id in selected_ids]
