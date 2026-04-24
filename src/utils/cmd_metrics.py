"""Helpers for correlation matrix distance (CMD) evaluation on frame-wise CSM tensors."""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import torch

CMD_COMPARISON_PREFIXES = (
    "cmd_reference_to_upsampler",
    "cmd_upsampler_to_lam",
    "cmd_reference_to_lam",
    "cmd_reference_to_lam_denoise1",
    "cmd_reference_to_lam_denoise2",
    "cmd_reference_to_lam_denoise3",
    "cmd_reference_to_lam_denoise4",
)
UPSAMPLER_DIAGNOSTIC_PREFIXES = (
    "upsampler_hermitian_residual",
    "upsampler_psd_projection_residual",
)
MATRIX_NDIM = 2
FRAME_BAND_NDIM = 4


def _finite_median(values: list[float]) -> float:
    """
    Return the median of the finite values or ``nan`` when none are available.

    Parameters
    ----------
    values : list[float]
        A list of numeric values, potentially containing non-finite entries.

    Returns
    -------
    float
        The median of the finite values, or ``nan`` if no finite values are present.
    """
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(statistics.median(finite_values))


def _to_complex_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    """
    Convert a tensor-like input into a complex-valued torch tensor.

    Parameters
    ----------
    value : Any
        A tensor-like object that may have methods to detach from computation graphs,
        move to CPU, or convert to NumPy arrays.

    Returns
    -------
    torch.Tensor
        A complex-valued torch tensor representation of the input value.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        target_device = device or tensor.device
        return tensor.to(device=target_device, dtype=torch.complex128)
    return torch.as_tensor(value, dtype=torch.complex128, device=device)


def _to_float_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    """Convert a tensor-like input into a float64 torch tensor."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        target_device = device or tensor.device
        return tensor.to(device=target_device, dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _infer_device(*values: Any) -> torch.device | None:
    """Return the first torch device found in the provided values, if any."""
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _finite_median_tensor(values: torch.Tensor) -> torch.Tensor:
    """Return the finite-value median using torch or ``nan`` when unavailable."""
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return torch.full((), float("nan"), dtype=torch.float64, device=values.device)
    return torch.quantile(finite_values, 0.5)


def _hermitian_part(value: torch.Tensor) -> torch.Tensor:
    """Return the Hermitian part of a complex square-matrix tensor."""
    return 0.5 * (value + value.mH)


def _matrix_is_finite(value: torch.Tensor) -> torch.Tensor:
    """Return a per-matrix finiteness mask for a complex square-matrix tensor."""
    return torch.isfinite(value.real).all(dim=(-2, -1)) & torch.isfinite(value.imag).all(
        dim=(-2, -1)
    )


def _safe_relative_matrix_norm(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """
    Return `||numerator||_F / ||denominator||_F` with `nan` for invalid matrices.

    Parameters
    ----------
    numerator : torch.Tensor
        A complex square-matrix tensor representing the numerator of the relative norm.
    denominator : torch.Tensor
        A complex square-matrix tensor representing the denominator of the relative norm.

    Returns
    -------
    torch.Tensor
        A tensor containing the relative Frobenius norms for each pair of matrices, with `nan`
        for any pairs where the numerator or denominator is non-finite or where the denominator has
        zero Frobenius norm.

    """
    numerator_norm = torch.linalg.matrix_norm(numerator, ord="fro", dim=(-2, -1))
    denominator_norm = torch.linalg.matrix_norm(denominator, ord="fro", dim=(-2, -1))
    finite_mask = (
        _matrix_is_finite(numerator)
        & _matrix_is_finite(denominator)
        & torch.isfinite(numerator_norm)
        & torch.isfinite(denominator_norm)
        & (denominator_norm > 0.0)
    )
    ratios = torch.full(
        numerator_norm.shape,
        float("nan"),
        dtype=torch.float64,
        device=numerator.device,
    )
    if torch.any(finite_mask):
        ratios[finite_mask] = numerator_norm[finite_mask] / denominator_norm[finite_mask]
    return ratios


def project_to_hermitian_psd(value: Any) -> torch.Tensor:
    """
    Project a complex square-matrix tensor onto the Hermitian PSD cone.

    Parameters
    ----------
    value : Any
        A tensor-like input with shape ``(..., C, C)``.

    Returns
    -------
    torch.Tensor
        Hermitian PSD tensor with the same shape as the input. Invalid matrices
        are returned as ``nan`` matrices.
    """
    target_device = _infer_device(value)
    matrix = _to_complex_tensor(value, device=target_device)
    if matrix.ndim < MATRIX_NDIM or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"PSD projection expects square matrices, got {matrix.shape}.")

    hermitian = _hermitian_part(matrix)
    finite_mask = _matrix_is_finite(hermitian)
    safe_hermitian = torch.where(
        finite_mask[..., None, None],
        hermitian,
        torch.zeros_like(hermitian),
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(safe_hermitian)
    clipped_eigenvalues = torch.clamp(eigenvalues.real, min=0.0).to(dtype=torch.complex128)
    projected = eigenvectors @ torch.diag_embed(clipped_eigenvalues) @ eigenvectors.mH
    projected = _hermitian_part(projected)

    nan_matrix = torch.full_like(projected, complex(float("nan"), float("nan")))
    return torch.where(finite_mask[..., None, None], projected, nan_matrix)


def _compute_cmd_tensor(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    Compute CMD for tensor inputs with shape ``(..., C, C)``.

    Parameters
    ----------
    left : torch.Tensor
        A complex-valued tensor representing the first set of correlation matrices, expected to have
        shape ``(..., C, C)``.
    right : torch.Tensor
        A complex-valued tensor representing the second set of correlation matrices, expected to
        have shape ``(..., C, C)``.

    Returns
    -------
    torch.Tensor
        A tensor containing the CMD values for each pair of correlation matrices.
    """
    left_matrix = _hermitian_part(left)
    right_matrix = _hermitian_part(right)

    left_is_finite = _matrix_is_finite(left_matrix)
    right_is_finite = _matrix_is_finite(right_matrix)

    left_norm = torch.linalg.matrix_norm(left_matrix, ord="fro", dim=(-2, -1))
    right_norm = torch.linalg.matrix_norm(right_matrix, ord="fro", dim=(-2, -1))
    norm_product = left_norm * right_norm
    norms_are_valid = torch.isfinite(left_norm) & torch.isfinite(right_norm) & (norm_product > 0.0)

    similarity = torch.diagonal(left_matrix @ right_matrix, dim1=-2, dim2=-1).sum(dim=-1)
    similarity_is_valid = torch.isfinite(similarity.real) & torch.isfinite(similarity.imag)

    valid_mask = left_is_finite & right_is_finite & norms_are_valid & similarity_is_valid
    cmd = torch.full(left_matrix.shape[:-2], float("nan"), dtype=torch.float64, device=left.device)
    if torch.any(valid_mask):
        normalised_similarity = similarity.real[valid_mask] / norm_product[valid_mask]
        normalised_similarity = torch.clamp(normalised_similarity, min=0.0, max=1.0)
        cmd[valid_mask] = torch.clamp(1.0 - normalised_similarity, min=0.0, max=1.0)
    return cmd


def compute_correlation_matrix_distance(left: Any, right: Any) -> float:
    """
    Compute CMD for one same-size complex correlation matrix pair.

    CMD follows the normalised-trace definition used for Hermitian correlation
    matrices, clipped into ``[0, 1]`` to absorb numerical spill.

    Parameters
    ----------
    left : Any
        A tensor-like object representing the first correlation matrix.
    right : Any
        A tensor-like object representing the second correlation matrix.

    Returns
    -------
    float
        The computed CMD value between the two input matrices, or ``nan`` if the inputs are invalid
        or contain non-finite values.

    """
    target_device = _infer_device(left, right)
    left_matrix = _to_complex_tensor(left, device=target_device)
    right_matrix = _to_complex_tensor(right, device=target_device)

    if left_matrix.shape != right_matrix.shape:
        raise ValueError(
            f"CMD expects matching shapes, got {left_matrix.shape} and {right_matrix.shape}."
        )
    if left_matrix.ndim != MATRIX_NDIM or left_matrix.shape[0] != left_matrix.shape[1]:
        raise ValueError(f"CMD expects square matrices, got {left_matrix.shape}.")
    return float(_compute_cmd_tensor(left_matrix, right_matrix).item())


def compute_cmd_per_frame_per_band(left: Any, right: Any) -> torch.Tensor:
    """
    Compute correlation matrix distance (CMD) for all frame-band CSM pairs.

    Parameters
    ----------
    left : Any
        A tensor-like object representing the first set of CSMs, expected to have shape
         ``(frames, bands, C, C)``.
    right : Any
        A tensor-like object representing the second set of CSMs, expected to have shape
         ``(frames, bands, C, C)``.

    Returns
    -------
    torch.Tensor
        A tensor of CMD values with shape ``(frames, bands)``.
    """
    target_device = _infer_device(left, right)
    left_csm = _to_complex_tensor(left, device=target_device)
    right_csm = _to_complex_tensor(right, device=target_device)

    if left_csm.shape != right_csm.shape:
        raise ValueError(
            f"Frame-wise CMD expects matching shapes, got {left_csm.shape} and {right_csm.shape}."
        )
    if left_csm.ndim != FRAME_BAND_NDIM:
        raise ValueError(f"Frame-wise CMD expects rank-4 tensors, got {left_csm.shape}.")
    if left_csm.shape[-1] != left_csm.shape[-2]:
        raise ValueError(f"Frame-wise CMD expects square CSMs, got {left_csm.shape}.")

    return _compute_cmd_tensor(left_csm, right_csm)


def compute_hermitian_residual_per_frame_per_band(value: Any) -> torch.Tensor:
    """
    Compute the relative Hermitian residual for frame-band matrices.

    The residual is ``||M - M^H||_F / ||M||_F`` and is ``nan`` when the input
    matrix is non-finite or has zero Frobenius norm.
    """
    target_device = _infer_device(value)
    matrix = _to_complex_tensor(value, device=target_device)
    if matrix.ndim != FRAME_BAND_NDIM or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"Hermitian residual expects rank-4 square matrices, got {matrix.shape}.")
    return _safe_relative_matrix_norm(matrix - matrix.mH, matrix)


def compute_psd_projection_residual_per_frame_per_band(value: Any) -> torch.Tensor:
    """
    Compute the relative PSD-projection residual for frame-band matrices.

    The residual is ``||H - proj_psd(H)||_F / ||H||_F`` where ``H`` is the
    Hermitian part of the raw matrix.

    Parameters
    ----------
    value : Any
        A tensor-like object representing the frame-band matrices, expected to have shape
        ``(frames, bands, C, C)``.

    Returns
    -------
    torch.Tensor
        A tensor of relative PSD-projection residuals with shape ``(frames, bands)``.
    """
    target_device = _infer_device(value)
    matrix = _to_complex_tensor(value, device=target_device)
    if matrix.ndim != FRAME_BAND_NDIM or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"PSD projection residual expects rank-4 square matrices, got {matrix.shape}."
        )
    hermitian = _hermitian_part(matrix)
    projected = project_to_hermitian_psd(hermitian)
    return _safe_relative_matrix_norm(hermitian - projected, hermitian)


def reduce_cmd_per_frame(per_frame_per_band: Any) -> torch.Tensor:
    """
    Reduce a ``(frames, bands)`` CMD array to one median CMD value per frame.

    Parameters
    ----------
    per_frame_per_band : Any
        A tensor-like object representing the per-frame, per-band CMD values, expected to have shape
         ``(frames, bands)``.

    Returns
    -------
    torch.Tensor
        A tensor of median CMD values with shape ``(frames,)``.
        Each value is the median of the finite CMD values across bands for the corresponding frame,
        or ``nan`` if no finite values are present.
    """
    values = _to_float_tensor(per_frame_per_band, device=_infer_device(per_frame_per_band))
    if values.ndim != MATRIX_NDIM:
        raise ValueError(f"Per-frame CMD reduction expects rank-2 input, got {values.shape}.")
    per_frame = [_finite_median_tensor(frame_values) for frame_values in values]
    return torch.stack(per_frame)


def to_json_compatible_nested_list(values: Any) -> list[Any]:
    """
    Convert a NumPy-like numeric container into nested Python lists for JSON output.

    Parameters
    ----------
    values : Any
        A tensor-like object that can be converted to a NumPy array, potentially containing numeric
        values of any shape.

    Returns
    -------
    list[Any]
        A nested list representation of the input values, with all numeric entries converted to
        Python floats.
    """
    return _to_float_tensor(values, device=torch.device("cpu")).tolist()


def summarise_metric_per_frame_per_band(per_frame_per_band: Any) -> dict[str, Any]:
    """
    Build the per-frame and file-level reductions for one frame-band metric.

    Parameters
    ----------
    per_frame_per_band : Any
        A tensor-like object representing the per-frame, per-band CMD values, expected to have shape
        ``(frames, bands)``.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the following keys:
        - "per_frame_per_band": A NumPy array of shape ``(frames, bands)`` containing the original
        CMD values.
        - "per_frame": A NumPy array of shape ``(frames,)`` containing the median CMD value for
        each frame, computed across bands.
        - "median": A float representing the median of the finite per-frame values across all
        frames, or ``nan`` if no finite values are present.
    """
    values = _to_float_tensor(per_frame_per_band, device=_infer_device(per_frame_per_band))
    per_frame = reduce_cmd_per_frame(values)
    return {
        "per_frame_per_band": values,
        "per_frame": per_frame,
        "median": float(_finite_median_tensor(per_frame).item()),
    }


def add_per_frame_per_band_metrics(
    metrics_row: dict[str, Any],
    prefix: str,
    per_frame_per_band: Any,
) -> bool:
    """
    Attach one already-computed frame-band metric series to a metrics row.

    Parameters
    ----------
    metrics_row : dict[str, Any]
        A dictionary to which the computed metrics will be added under keys prefixed by the provided
        prefix.
    prefix : str
        A string prefix to use for the keys under which the metrics will be stored in the
        metrics_row dictionary.
    per_frame_per_band : Any
        A tensor-like object representing the per-frame, per-band metric values, expected to have
        shape ``(frames, bands)``.

    Returns
    -------
    bool
        ``True`` if the metrics were successfully computed and added to the metrics_row, or
        ``False`` if the input was unavailable or shape-incompatible.
    """
    values = _to_float_tensor(per_frame_per_band, device=_infer_device(per_frame_per_band))
    if values.ndim != MATRIX_NDIM:
        return False

    summary = summarise_metric_per_frame_per_band(values)
    metrics_row[f"{prefix}_per_frame_per_band"] = to_json_compatible_nested_list(
        summary["per_frame_per_band"]
    )
    metrics_row[f"{prefix}_per_frame"] = to_json_compatible_nested_list(summary["per_frame"])
    metrics_row[f"{prefix}_median"] = float(summary["median"])
    return True


def add_cmd_metrics(metrics_row: dict[str, Any], prefix: str, left: Any, right: Any) -> bool:
    """
    Compute and attach CMD series for one named comparison.

    Returns ``False`` when the inputs are unavailable or shape-incompatible.

    Parameters
    ----------
    metrics_row : dict[str, Any]
        A dictionary to which the computed CMD metrics will be added under keys prefixed by the
        provided prefix.
    prefix : str
        A string prefix to use for the keys under which the CMD metrics will be stored in the
        metrics_row dictionary.
    left : Any
        A tensor-like object representing the first set of CSMs, expected to have shape
        ``(frames, bands, C, C)``.
    right : Any
        A tensor-like object representing the second set of CSMs, expected to have shape
        ``(frames, bands, C, C)``.

    Returns
    -------
    bool
        ``True`` if the CMD metrics were successfully computed and added to the metrics_row,
        or ``False`` if the inputs were unavailable or shape-incompatible.
    """
    if left is None or right is None:
        return False

    target_device = _infer_device(left, right)
    left_csm = _to_complex_tensor(left, device=target_device)
    right_csm = _to_complex_tensor(right, device=target_device)
    if left_csm.shape != right_csm.shape:
        return False
    if left_csm.ndim != FRAME_BAND_NDIM or left_csm.shape[-1] != left_csm.shape[-2]:
        return False

    return add_per_frame_per_band_metrics(
        metrics_row,
        prefix,
        compute_cmd_per_frame_per_band(left_csm, right_csm),
    )


def add_upsampler_validity_metrics(metrics_row: dict[str, Any], upsampler_output: Any) -> bool:
    """
    Attach raw-upsampler CSM validity diagnostics to a metrics row.

    These diagnostics include the relative Hermitian residual and the relative PSD-projection
    residual, both computed on a per-frame, per-band basis.

    Parameters
    ----------
    metrics_row : dict[str, Any]
        A dictionary to which the computed diagnostics will be added under keys prefixed by
        "upsampler".
    upsampler_output : Any
        A tensor-like object representing the raw output CSMs from the upsampler, expected to have
        shape ``(frames, bands, C, C)``.

    Returns
    -------
    bool
        ``True`` if the diagnostics were successfully computed and added to the metrics_row, or
        ``False`` if the input was unavailable or shape-incompatible.
    """
    target_device = _infer_device(upsampler_output)
    upsampler_tensor = _to_complex_tensor(upsampler_output, device=target_device)
    if (
        upsampler_tensor.ndim != FRAME_BAND_NDIM
        or upsampler_tensor.shape[-1] != upsampler_tensor.shape[-2]
    ):
        return False

    added = False
    added |= add_per_frame_per_band_metrics(
        metrics_row,
        "upsampler_hermitian_residual",
        compute_hermitian_residual_per_frame_per_band(upsampler_tensor),
    )
    added |= add_per_frame_per_band_metrics(
        metrics_row,
        "upsampler_psd_projection_residual",
        compute_psd_projection_residual_per_frame_per_band(upsampler_tensor),
    )
    return added


def aggregate_global_frame_median(per_file_rows: list[dict[str, Any]], prefix: str) -> float:
    """
    Flatten one comparison across files and return the global frame-level median.

    Parameters
    ----------
    per_file_rows : list[dict[str, Any]]
        A list of dictionaries, each representing the CMD metrics for a single file.
    prefix : str
        The prefix used to identify the CMD metrics in each dictionary.

    Returns
    -------
    float
        The global frame-level median across all files, or ``nan`` if no valid values are
        present.
    """
    flattened: list[float] = []
    key = f"{prefix}_per_frame"
    for row in per_file_rows:
        if key not in row:
            continue
        values = np.asarray(row[key], dtype=np.float64).reshape(-1)
        flattened.extend(float(value) for value in values if math.isfinite(float(value)))
    if not flattened:
        return float("nan")
    return float(statistics.median(flattened))


def aggregate_all_global_medians(
    per_file_rows: list[dict[str, Any]],
    *,
    prefixes: tuple[str, ...],
) -> dict[str, float]:
    """Aggregate any ``*_per_frame`` series into ``*_median`` fields."""
    return {
        f"{prefix}_median": aggregate_global_frame_median(per_file_rows, prefix)
        for prefix in prefixes
    }


def aggregate_all_cmd_global_medians(
    per_file_rows: list[dict[str, Any]],
    *,
    prefixes: tuple[str, ...] = CMD_COMPARISON_PREFIXES,
) -> dict[str, float]:
    """
    Aggregate all configured CMD comparisons into ``*_median`` output fields.

    Parameters
    ----------
    per_file_rows : list[dict[str, Any]]
        A list of dictionaries, each representing the CMD metrics for a single file.
    prefixes : tuple[str, ...], optional
        A tuple of string prefixes identifying the CMD comparisons to aggregate, by default
        set to CMD_COMPARISON_PREFIXES.
    """
    return aggregate_all_global_medians(per_file_rows, prefixes=prefixes)
