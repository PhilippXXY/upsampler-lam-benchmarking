"""
Visibility matrix and covariance matrix construction for acoustic inference.

This module provides functions to compute spatial covariance matrices (also called
visibility matrices) from multi-channel audio signals for Direction of Arrival
estimation. The implementation is tailored for inference on STARSS23 data.

Key Functions
-------------
- extract_visibilities : Compute per-frame covariance matrices via STFT
- form_visibility : Aggregate short-time matrices into stationarity blocks
- get_visibility_matrix : Main entry point for multi-band CSM computation

The visibility matrix represents the spatial coherence between microphone pairs
across frequency bands and time frames. These matrices serve as input to the
LAM/UpLAM models for Direction of Arrival estimation.

Processing Pipeline
-------------------
1. Segment audio into overlapping time windows (100ms frames)
2. Apply Tukey window and compute STFT
3. Extract frequency band of interest via summation
4. Form covariance matrices from STFT coefficients
5. Normalise eigenvalues for numerical stability
6. Stack across frequency bands and time frames

Notes
-----
This is a trimmed inference-only version. Training features such as APGD (Adaptive
Projected Gradient Descent) preprocessing and non-linear frequency scaling have
been removed to simplify the codebase.

References
----------
.. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
.. [2] STARSS23 Challenge: https://arxiv.org/pdf/2306.09126v1
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import skimage.util as skutil
from scipy import linalg
from scipy.signal import windows


def _ensure_tc(audio_in: np.ndarray) -> np.ndarray:
    """
    Ensure audio array has shape (T, C) for time-major processing.

    Accepts both (T, C) and (C, T) arrays, transposing if necessary based
    on heuristic channel/sample dimension detection.

    Parameters
    ----------
    audio_in : np.ndarray
        Input audio array, either (T, C) or (C, T)

    Returns
    -------
    np.ndarray
        Audio array in (T, C) format

    Raises
    ------
    ValueError
        If audio_in is not 2-dimensional
    """
    if audio_in.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"audio_in must be 2D (T,C) or (C,T), got shape={audio_in.shape}")

    # Heuristic: if first dim looks like channels and second like samples -> transpose
    if audio_in.shape[0] <= 64 and audio_in.shape[1] > audio_in.shape[0]:  # noqa: PLR2004
        return audio_in.T
    return audio_in


def extract_visibilities(  # noqa: PLR0913
    data: np.ndarray,
    rate: int,
    T: float,
    fc: float,
    bw: float,
    alpha: float,
) -> np.ndarray:
    """
    Transform multi-channel time-series to per-frame visibility matrices.

    Computes spatial covariance matrices for consecutive time frames by:
    1. Segmenting audio into frames of duration T
    2. Applying Tukey window with parameter alpha
    3. Computing FFT for each frame
    4. Extracting frequency band [fc - bw/2, fc + bw/2]
    5. Forming covariance matrix from STFT coefficients

    Parameters
    ----------
    data : np.ndarray
        Multi-channel audio of shape (T_samples, C) where C is number of channels
    rate : int
        Sample rate in Hz
    T : float
        Frame duration in seconds (e.g., 0.01 for 10ms frames)
    fc : float
        Centre frequency of band in Hz
    bw : float
        Bandwidth in Hz
    alpha : float
        Tukey window parameter in [0, 1]
        0 = rectangular, 1 = Hann window

    Returns
    -------
    np.ndarray
        Visibility matrices of shape (N_frames, C, C) where N_frames is the
        number of complete time windows fitting in the input data
        Complex-valued covariance matrices

    Raises
    ------
    ValueError
        If frame duration results in zero samples

    Notes
    -----
    - Incomplete final frame is discarded
    - Covariance computed as: S = X @ X^H where X is STFT vector
    - Frequency band extraction uses summation to collapse STFT bins
    """
    N_stft_sample = int(rate * T)
    if N_stft_sample <= 0:
        raise ValueError("Not enough samples per STFT frame.")

    N_sample = (data.shape[0] // N_stft_sample) * N_stft_sample
    N_channel = data.shape[1]

    stf_data = skutil.view_as_blocks(data[:N_sample], (N_stft_sample, N_channel)).squeeze(
        axis=1
    )  # (N_stf, N_stft_sample, N_channel)

    window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window

    stft_data = np.fft.fft(stf_win_data, axis=1)

    idx_start = int((fc - 0.5 * bw) * N_stft_sample / rate)
    idx_end = int((fc + 0.5 * bw) * N_stft_sample / rate)

    collapsed = np.sum(stft_data[:, idx_start : idx_end + 1, :], axis=1)
    N_stf = collapsed.shape[0]

    S = collapsed.reshape(N_stf, -1, 1).conj() * collapsed.reshape(N_stf, 1, -1)
    return S


def form_visibility(  # noqa: PLR0913
    data: np.ndarray,
    rate: int,
    fc: float,
    bw: float,
    T_sti: float,
    T_stationarity: float,
) -> np.ndarray:
    """
    Aggregate short-time visibilities into stationarity blocks.

    Computes visibility matrices at short time intervals (T_sti) and
    accumulates them over longer stationarity windows (T_stationarity)
    to reduce variance whilst assuming local stationarity.

    Parameters
    ----------
    data : np.ndarray
        Multi-channel audio of shape (T_samples, C)
    rate : int
        Sample rate in Hz
    fc : float
        Centre frequency in Hz
    bw : float
        Bandwidth in Hz
    T_sti : float
        Short-time integration window in seconds (e.g., 0.01 for 10ms)
    T_stationarity : float
        Stationarity assumption window in seconds (e.g., 0.1 for 100ms)

    Returns
    -------
    np.ndarray
        Accumulated visibility matrices of shape (N_frames, C, C)
        where N_frames = floor(N_sti / N_sti_per_block)
        Complex-valued covariance matrices
    """
    S_sti = extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0)

    _, N_channel = data.shape
    N_sti_per_block = int(T_stationarity / T_sti)

    S = (
        skutil.view_as_windows(
            S_sti,
            (N_sti_per_block, N_channel, N_channel),
            (N_sti_per_block, N_channel, N_channel),  # type: ignore
        )
        .squeeze(axis=(1, 2))
        .sum(axis=1)
    )
    return S


def _normalise_visibility_frame(
    visibility_frame: np.ndarray,
    diagonal_loading: float = 0.0,
    eigenvalue_floor: float = 0.0,
) -> np.ndarray:
    """
    Sanitise and normalise one visibility matrix frame.

    Parameters
    ----------
    visibility_frame : np.ndarray
        Complex-valued covariance matrix for one band and one frame, shape ``(C, C)``.
    diagonal_loading : float, optional
        Non-negative scalar added to the diagonal before eigendecomposition.
        Helps stabilise nearly singular covariance matrices.
    eigenvalue_floor : float, optional
        Relative floor applied after eigenvalue normalisation. Eigenvalues whose
        normalised magnitude is below this threshold are set to zero.

    Returns
    -------
    np.ndarray
        Sanitised and normalised covariance matrix of shape ``(C, C)``.
    """
    frame = np.asarray(visibility_frame, dtype=np.complex128)
    frame = np.nan_to_num(frame.real, nan=0.0, posinf=0.0, neginf=0.0) + 1j * np.nan_to_num(
        frame.imag, nan=0.0, posinf=0.0, neginf=0.0
    )
    frame = 0.5 * (frame + frame.conj().T)

    max_abs = float(np.max(np.abs(frame))) if frame.size > 0 else 0.0
    if np.isfinite(max_abs) and max_abs > 0.0:
        frame = frame / max_abs

    if diagonal_loading > 0.0:
        frame = frame + diagonal_loading * np.eye(frame.shape[0], dtype=np.complex128)

    eigenvalues, eigenvectors = linalg.eigh(frame, check_finite=False)
    eigenvalues = np.nan_to_num(
        np.asarray(eigenvalues, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size > 0 else 0.0
    if not np.isfinite(max_eigenvalue) or max_eigenvalue <= 0.0:
        normalised_eigenvalues = np.zeros_like(eigenvalues, dtype=np.float64)
    else:
        normalised_eigenvalues = np.clip(eigenvalues / max_eigenvalue, 0.0, None)
        if eigenvalue_floor > 0.0:
            normalised_eigenvalues[normalised_eigenvalues < eigenvalue_floor] = 0.0

    normalised = np.einsum(
        "ij,j,kj->ik",
        eigenvectors,
        normalised_eigenvalues,
        eigenvectors.conj(),
        optimize=True,
    )
    normalised = np.nan_to_num(
        normalised.real,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ) + 1j * np.nan_to_num(
        normalised.imag,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return 0.5 * (normalised + normalised.conj().T)


def get_visibility_matrix(  # noqa: PLR0913
    audio_in: np.ndarray,
    fs: int,
    apgd: bool = False,
    bands: Iterable[float] | None = None,
    T_sti: float = 10e-3,
    scale: str = "linear",
    nbands: int = 9,
    diagonal_loading: float = 0.0,
    eigenvalue_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multi-band visibility/covariance matrices from multi-channel audio.

    Main entry point for converting raw audio into the spatial covariance matrix
    representation used by LAM/UpLAM models. Processes multiple frequency bands
    in parallel and returns stacked tensors.

    Parameters
    ----------
    audio_in : np.ndarray
        Multi-channel audio, shape (T_samples, C) or (C, T_samples)
        Automatically transposed to (T, C) if needed
    fs : int
        Sample rate in Hz
    apgd : bool, optional
        Whether to apply APGD preprocessing (default: False)
        Not supported in this inference-only version
    bands : Iterable[float] or None, optional
        Custom frequency band centres in Hz (default: None)
        If None, uses linear spacing from 1500-4500 Hz
    T_sti : float, optional
        Short-time integration window in seconds (default: 10e-3 = 10ms)
    scale : str, optional
        Frequency scaling method (default: "linear")
        Only "linear" is supported in this version
    nbands : int, optional
        Number of frequency bands (default: 9)
    diagonal_loading : float, optional
        Non-negative scalar added to each visibility-frame diagonal before
        eigendecomposition (default: 0.0).
    eigenvalue_floor : float, optional
        Relative floor in normalised eigenvalue units. Values below this are
        suppressed after eigenvalue normalisation (default: 0.0).

    Returns
    -------
    visibilities : np.ndarray
        Covariance matrices of shape (nbands, N_frames, C, C)
        Complex-valued spatial covariance for each band and frame
    apgd_map : np.ndarray
        Placeholder array of zeros with shape (nbands, N_frames, 1)
        Included for interface compatibility but unused

    Raises
    ------
    ValueError
        If apgd=True (not supported in inference version)
        If scale != "linear" (only linear scaling supported)
    TypeError
        If fs is not an integer type

    Examples
    --------
    >>> audio = np.random.randn(24000, 4)  # 1 second, 4 channels
    >>> S, _ = get_visibility_matrix(audio, fs=24000, nbands=9)
    >>> print(S.shape)
    (9, 10, 4, 4)  # 9 bands, 10 frames, 4x4 covariance
    """
    if apgd:
        raise ValueError("apgd=True is not supported in the trimmed inference version.")
    if scale != "linear":
        raise ValueError("Only scale='linear' is supported in the trimmed inference version.")
    if diagonal_loading < 0.0:
        raise ValueError("diagonal_loading must be non-negative.")
    if eigenvalue_floor < 0.0:
        raise ValueError("eigenvalue_floor must be non-negative.")

    audio_tc = _ensure_tc(audio_in)
    if not isinstance(fs, (int, np.integer)):
        raise TypeError(f"fs must be int, got {type(fs)}")

    freq = skutil.view_as_windows(np.linspace(1500, 4500, nbands + 1), (2,), 1).mean(axis=-1)
    bw = 50.0

    visibilities: list[list[np.ndarray]] = []
    apgd_map: list[np.ndarray] = []

    for i in range(nbands):
        T_stationarity = 10 * T_sti
        S = form_visibility(audio_tc, int(fs), float(freq[i]), bw, T_sti, T_stationarity)

        N_frame = S.shape[0]
        per_frame: list[np.ndarray] = []
        per_band_dummy = np.zeros((N_frame, 1), dtype=np.float32)

        for s_idx in range(N_frame):
            per_frame.append(
                _normalise_visibility_frame(
                    S[s_idx],
                    diagonal_loading=diagonal_loading,
                    eigenvalue_floor=eigenvalue_floor,
                )
            )

        visibilities.append(per_frame)
        apgd_map.append(per_band_dummy)

    return np.array(visibilities), np.array(apgd_map)
