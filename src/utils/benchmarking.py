"""Scientific benchmarking helpers for runtime aggregation and workload shaping."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from utils.cmd_metrics import aggregate_all_cmd_global_medians

TWO_D_AUDIO_NDIM = 2


@dataclass(frozen=True)
class AggregatedBenchmarkMetrics:
    """
    Aggregated benchmark metrics derived from one inference metrics JSON file.

    Parameters
    ----------
    files_processed : int
        Number of per-file metric rows found in the metrics JSON payload.
    total_frames : int
        Total number of frames processed across all per-file metric rows.
    latency_per_frame_ms : float
        End-to-end latency per frame in milliseconds, aggregated from total per-file latency.
    gflops_per_frame : float
        End-to-end GFLOPs per frame, aggregated from total per-file FLOPs.
    lam_latency_per_frame_ms : float
        LAM-only latency per frame in milliseconds.
    lam_gflops_per_frame : float
        LAM-only GFLOPs per frame.
    memory_peak_max_mb : float
        Worst-case end-to-end peak memory across files, in MiB.
    memory_peak_median_mb : float
        Median end-to-end peak memory across files, in MiB.
    cmd_reference_to_upsampler_median : float
        Global median CMD over all frame-level `reference -> upsampler` values after Hermitian-PSD
        projection of the upsampler output.
    cmd_upsampler_to_lam_median : float
        Global median CMD over all frame-level `upsampler -> lam_final` values after
        Hermitian-PSD projection of the upsampler output.
    cmd_reference_to_lam_median : float
        Global median CMD over all frame-level `reference -> lam_final` values.
    cmd_reference_to_lam_denoise1_median : float
        Global median CMD over all frame-level `reference -> lam_denoise1` values.
    cmd_reference_to_lam_denoise2_median : float
        Global median CMD over all frame-level `reference -> lam_denoise2` values.
    cmd_reference_to_lam_denoise3_median : float
        Global median CMD over all frame-level `reference -> lam_denoise3` values.
    cmd_reference_to_lam_denoise4_median : float
        Global median CMD over all frame-level `reference -> lam_denoise4` values.
    localisation_error_deg : float
        Aggregated localisation error in degrees, or ``nan`` if unavailable.
    localisation_recall : float
        Aggregated localisation recall on the native 0..1 scale, or ``nan`` if unavailable.
    total_params : int
        Total number of model parameters, or ``0`` if unavailable.
    file_ids : tuple[str, ...]
        Ordered tuple of file identifiers observed in the per-file metric rows.
    """

    files_processed: int
    total_frames: int
    latency_per_frame_ms: float
    gflops_per_frame: float
    lam_latency_per_frame_ms: float
    lam_gflops_per_frame: float
    memory_peak_max_mb: float
    memory_peak_median_mb: float
    cmd_reference_to_upsampler_median: float
    cmd_upsampler_to_lam_median: float
    cmd_reference_to_lam_median: float
    cmd_reference_to_lam_denoise1_median: float
    cmd_reference_to_lam_denoise2_median: float
    cmd_reference_to_lam_denoise3_median: float
    cmd_reference_to_lam_denoise4_median: float
    localisation_error_deg: float
    localisation_recall: float
    total_params: int
    file_ids: tuple[str, ...]


def _float_from_candidates(
    row: dict[str, Any],
    *keys: str,
    default: float = 0.0,
) -> float:
    """
    Read the first numeric value present under the provided keys.

    Parameters
    ----------
    row : dict[str, Any]
        Mapping containing metric fields.
    *keys : str
        Candidate field names in priority order.
    default : float, optional
        Fallback value when none of the candidate fields can be parsed.

    Returns
    -------
    float
        Parsed floating-point value.
    """
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def apply_benchmark_audio_length(
    audio_tc: NDArray[np.float32],
    *,
    sample_rate: int,
    inference_config: dict[str, Any],
) -> NDArray[np.float32]:
    """
    Crop and optionally zero-pad time-major audio for benchmark comparability.

    Parameters
    ----------
    audio_tc : NDArray[np.float32]
        Audio array with shape ``(T, C)``.
    sample_rate : int
        Sampling rate of ``audio_tc`` in Hz.
    inference_config : dict[str, Any]
        Inference configuration. The helper reads:
        - ``max_audio_length_sec``
        - ``benchmark_runtime_only``
        - ``normalised_memory_enabled``
        - ``normalised_memory_duration_sec``
        - ``normalised_memory_pad_short_files``

    Returns
    -------
    NDArray[np.float32]
        Cropped and, when configured, zero-padded audio.

    Raises
    ------
    ValueError
        If the sample rate is not positive or the audio is not two-dimensional.
    """
    if audio_tc.ndim != TWO_D_AUDIO_NDIM:
        raise ValueError(f"Expected audio with shape (T, C), got {audio_tc.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

    runtime_only = bool(inference_config.get("benchmark_runtime_only", False))
    normalised_enabled = bool(inference_config.get("normalised_memory_enabled", False))
    target_duration_sec = _float_from_candidates(
        inference_config,
        "max_audio_length_sec",
        default=0.0,
    )
    if runtime_only and normalised_enabled:
        configured_duration = _float_from_candidates(
            inference_config,
            "normalised_memory_duration_sec",
            default=target_duration_sec,
        )
        if configured_duration > 0.0:
            target_duration_sec = configured_duration

    if target_duration_sec <= 0.0:
        return np.asarray(audio_tc, dtype=np.float32, copy=False)

    target_samples = int(round(sample_rate * target_duration_sec))
    prepared_audio = np.asarray(audio_tc[:target_samples], dtype=np.float32, copy=False)
    should_pad = (
        runtime_only
        and normalised_enabled
        and bool(inference_config.get("normalised_memory_pad_short_files", False))
    )
    if not should_pad or prepared_audio.shape[0] >= target_samples:
        return prepared_audio

    padded_audio = np.zeros((target_samples, prepared_audio.shape[1]), dtype=np.float32)
    padded_audio[: prepared_audio.shape[0], :] = prepared_audio
    return padded_audio


def aggregate_metrics_json(metrics_path: Path) -> AggregatedBenchmarkMetrics:
    """
    Aggregate scientific benchmark metrics from one metrics JSON file.

    Parameters
    ----------
    metrics_path : Path
        Path to the ``metrics_*.json`` file produced by ``src/infer.py``.

    Returns
    -------
    AggregatedBenchmarkMetrics
        Aggregated benchmark metrics for the run.

    Raises
    ------
    ValueError
        If the metrics payload is malformed or contains no per-file rows.
    """
    with open(metrics_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected list in metrics JSON: {metrics_path}")

    per_file_rows = [row for row in payload if isinstance(row, dict) and "file_id" in row]
    if not per_file_rows:
        raise ValueError(f"No per-file entries found in metrics JSON: {metrics_path}")

    files_processed = len(per_file_rows)
    total_frames = int(sum(_float_from_candidates(row, "num_frames") for row in per_file_rows))
    if total_frames <= 0:
        raise ValueError(f"Non-positive total frame count in metrics JSON: {metrics_path}")

    total_time_ms = sum(
        _float_from_candidates(
            row,
            "total_time_ms",
            default=_float_from_candidates(row, "latency_per_frame_ms")
            * _float_from_candidates(row, "num_frames"),
        )
        for row in per_file_rows
    )
    total_flops = sum(
        _float_from_candidates(
            row,
            "total_flops",
            default=_float_from_candidates(row, "flops_per_frame")
            * _float_from_candidates(row, "num_frames"),
        )
        for row in per_file_rows
    )
    lam_total_time_ms = sum(
        _float_from_candidates(row, "lam_total_time_ms") for row in per_file_rows
    )
    lam_total_flops = sum(_float_from_candidates(row, "lam_flops") for row in per_file_rows)

    memory_peaks_mb = [
        _float_from_candidates(row, "total_memory_mb", "memory_mb") for row in per_file_rows
    ]
    finite_memory_peaks_mb = [value for value in memory_peaks_mb if math.isfinite(value)]
    memory_peak_max_mb = max(finite_memory_peaks_mb) if finite_memory_peaks_mb else float("nan")
    memory_peak_median_mb = (
        float(statistics.median(finite_memory_peaks_mb)) if finite_memory_peaks_mb else float("nan")
    )

    total_params_values = [
        int(_float_from_candidates(row, "total_params", default=0)) for row in per_file_rows
    ]
    total_params = max(total_params_values) if total_params_values else 0

    evaluation_rows = [
        row
        for row in payload
        if isinstance(row, dict)
        and "file_id" not in row
        and (
            "localisation_error" in row
            or "localization_error" in row
            or "localisation_recall" in row
            or "localization_recall" in row
        )
    ]
    evaluation_row = evaluation_rows[-1] if evaluation_rows else {}
    cmd_medians = aggregate_all_cmd_global_medians(per_file_rows)

    return AggregatedBenchmarkMetrics(
        files_processed=files_processed,
        total_frames=total_frames,
        latency_per_frame_ms=total_time_ms / float(total_frames),
        gflops_per_frame=(total_flops / float(total_frames)) / 1e9,
        lam_latency_per_frame_ms=lam_total_time_ms / float(total_frames),
        lam_gflops_per_frame=(lam_total_flops / float(total_frames)) / 1e9,
        memory_peak_max_mb=float(memory_peak_max_mb),
        memory_peak_median_mb=float(memory_peak_median_mb),
        cmd_reference_to_upsampler_median=cmd_medians["cmd_reference_to_upsampler_median"],
        cmd_upsampler_to_lam_median=cmd_medians["cmd_upsampler_to_lam_median"],
        cmd_reference_to_lam_median=cmd_medians["cmd_reference_to_lam_median"],
        cmd_reference_to_lam_denoise1_median=cmd_medians["cmd_reference_to_lam_denoise1_median"],
        cmd_reference_to_lam_denoise2_median=cmd_medians["cmd_reference_to_lam_denoise2_median"],
        cmd_reference_to_lam_denoise3_median=cmd_medians["cmd_reference_to_lam_denoise3_median"],
        cmd_reference_to_lam_denoise4_median=cmd_medians["cmd_reference_to_lam_denoise4_median"],
        localisation_error_deg=_float_from_candidates(
            evaluation_row,
            "localisation_error",
            "localization_error",
            default=float("nan"),
        ),
        localisation_recall=_float_from_candidates(
            evaluation_row,
            "localisation_recall",
            "localization_recall",
            default=float("nan"),
        ),
        total_params=total_params,
        file_ids=tuple(str(row["file_id"]) for row in per_file_rows),
    )
