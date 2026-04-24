"""
Inference utilities for acoustic mapping.

Provides essential functions for LAM/UpLAM inference including:
    - Steering operator computation for spherical microphone arrays
    - Fibonacci sphere sampling for field-of-view generation
    - Coordinate transformations (Cartesian, polar, equatorial)
    - EigenMike array geometry definitions

This is a trimmed version containing only inference-time requirements.
Training utilities and advanced mathematical tables have been removed.

References
----------
.. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
"""

from __future__ import annotations

import math
import warnings
from collections import abc

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from scipy import constants

# EigenMike 32-channel spherical array capsule positions:
# [colatitude_deg, azimuth_deg, radius_m]
EIGENMIKE_RAW: dict[str, list[float]] = {
    "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042], "4": [90, 328, 0.042],
    "5": [32, 0, 0.042], "6": [55, 45, 0.042], "7": [90, 69, 0.042], "8": [125, 45, 0.042],
    "9": [148, 0, 0.042], "10": [125, 315, 0.042], "11": [90, 291, 0.042], "12": [55, 315, 0.042],
    "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042], "16": [159, 89, 0.042],
    "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
    "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],
    "25": [148, 180, 0.042], "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042],
    "29": [21, 269, 0.042], "30": [58, 270, 0.042], "31": [122, 270, 0.042], "32": [159, 271, 0.042],
}


def _deg2rad(coords_dict: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    Convert spherical coordinates from degrees to radians.

    Parameters
    ----------
    coords_dict : dict[str, list[float]]
        Dictionary mapping channel identifiers to [colatitude_deg, azimuth_deg, radius_m]

    Returns
    -------
    dict[str, list[float]]
        Dictionary with angles converted to radians [colatitude_rad, azimuth_rad, radius_m]
    """
    return {m: [math.radians(c[0]), math.radians(c[1]), c[2]] for m, c in coords_dict.items()}


def _polar2cart(coords_dict: dict[str, list[float]], *, units: str) -> dict[str, list[float]]:
    """
    Convert polar/spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    coords_dict : dict[str, list[float]]
        Dictionary mapping identifiers to [colatitude, azimuth, radius]
    units : str
        Angular unit specification: 'degrees' or 'radians'

    Returns
    -------
    dict[str, list[float]]
        Dictionary mapping identifiers to [x, y, z] Cartesian coordinates

    Raises
    ------
    ValueError
        If units is not 'degrees' or 'radians'
    """
    if units not in {"degrees", "radians"}:
        raise ValueError("units must be 'degrees' or 'radians'")
    if units == "degrees":
        coords_dict = _deg2rad(coords_dict)

    out: dict[str, list[float]] = {}
    for m, (colat, az, r) in coords_dict.items():
        x = r * math.sin(colat) * math.cos(az)
        y = r * math.sin(colat) * math.sin(az)
        z = r * math.cos(colat)
        out[m] = [x, y, z]
    return out


def get_xyz() -> list[list[float]]:
    """
    Get Cartesian coordinates of EigenMike 32-channel array capsules.

    Returns
    -------
    list[list[float]]
        List of [x, y, z] positions for each microphone capsule
    """
    mic_coords = _polar2cart(EIGENMIKE_RAW, units="degrees")
    return [[coord for coord in mic_coords[ch]] for ch in mic_coords]


def fibonacci(N: int, *, shift_lon: float = 0.0, shift_colat: float = 0.0) -> np.ndarray:
    """
    Generate Fibonacci sphere sampling points for field-of-view discretisation.

    Creates a quasi-uniform distribution of points on a unit sphere using
    the Fibonacci spiral algorithm. Provides excellent spatial uniformity
    for acoustic field sampling.

    Parameters
    ----------
    N : int
        Fibonacci sphere order, determines number of points (4*(N+1)^2 total points)
    shift_lon : float, optional
        Longitude/azimuth shift in radians (default: 0.0)
    shift_colat : float, optional
        Colatitude shift in radians (default: 0.0)

    Returns
    -------
    np.ndarray
        Array of shape (3, N_px) containing [x, y, z] unit sphere coordinates
        where N_px = 4*(N+1)^2

    Raises
    ------
    ValueError
        If N is negative
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)

    colat = np.arccos(1 - (2 * n + 1) / N_px) + shift_colat
    lon = (4 * np.pi * n) / (1 + np.sqrt(5)) + shift_lon

    x = np.sin(colat) * np.cos(lon)
    y = np.sin(colat) * np.sin(lon)
    z = np.cos(colat)
    return np.stack([x, y, z], axis=0)  # (3, N_px)


def get_field(
    min_freq: float = 1500.0,
    max_freq: float = 4500.0,
    nbands: int = 10,
    *,
    shift_lon: float = 0.0,
    shift_colat: float = 0.0,
) -> np.ndarray:
    """
    Generate spherical field-of-view sampling points for acoustic mapping.

    Creates a Fibonacci sphere grid representing the spatial field-of-view
    for Direction of Arrival estimation. Points within ±90° elevation are
    retained for practical acoustic sensing scenarios.

    Parameters
    ----------
    min_freq : float, optional
        Minimum frequency in Hz (default: 1500.0)
        Currently unused but kept for interface compatibility
    max_freq : float, optional
        Maximum frequency in Hz (default: 4500.0)
        Currently unused but kept for interface compatibility
    nbands : int, optional
        Number of frequency bands (default: 10)
        Currently unused but kept for interface compatibility
    shift_lon : float, optional
        Longitude shift in radians (default: 0.0)
    shift_colat : float, optional
        Colatitude shift in radians (default: 0.0)

    Returns
    -------
    np.ndarray
        Array of shape (3, N_px) containing [x, y, z] coordinates of spatial pixels
        Filtered to ±90° elevation range
    """
    sh_order = 10

    R = fibonacci(sh_order, shift_lon=shift_lon, shift_colat=shift_colat)

    mask = np.abs(R[2, :]) < np.sin(np.deg2rad(90))
    return R[:, mask]


def steering_operator(
    min_freq: float = 1500.0,
    max_freq: float = 4500.0,
    nbands: int = 10
    ) -> np.ndarray:
    """
    Compute steering operator matrix for spherical microphone array.

    Generates the complex-valued steering matrix that relates spatial pixels
    to microphone array measurements. Based on plane wave propagation model
    with far-field assumptions.

    Parameters
    ----------
    min_freq : float, optional
        Minimum frequency in Hz (default: 1500.0)
        Currently unused but kept for interface compatibility
    max_freq : float, optional
        Maximum frequency in Hz (default: 4500.0)
        Used to compute representative wavelength
    nbands : int, optional
        Number of frequency bands (default: 10)
        Currently unused but kept for interface compatibility

    Returns
    -------
    np.ndarray
        Complex steering matrix of shape (N_mics, N_px)
        Maps spatial pixels to expected microphone array responses

    Raises
    ------
    ValueError
        If computed wavelength is non-positive
    """
    xyz = get_xyz()
    XYZ = np.array(xyz).T
    R = get_field(min_freq=min_freq, max_freq=max_freq, nbands=nbands)

    wl = constants.speed_of_sound / (max_freq + 500.0)
    if wl <= 0:
        raise ValueError("Computed wavelength must be positive")

    scale = 2 * np.pi / wl
    # Suppress spurious warnings from Apple Accelerate BLAS on M-series chips
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        A = np.exp((-1j * scale * XYZ.T) @ R)  # (N_mics, N_px)
    return A


# ---------------------------------------------------------------------------
# Coordinate conversion helpers (for k-means / output formatting)
# ---------------------------------------------------------------------------

def is_scalar(x) -> bool:
    """
    Check if object is a scalar value.

    Parameters
    ----------
    x : any
        Object to test

    Returns
    -------
    bool
        True if x is scalar (not a container), False otherwise
    """
    if not isinstance(x, abc.Container):
        return True
    return False


def cart2pol(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar/spherical coordinates.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : np.ndarray
        Z coordinates

    Returns
    -------
    r : np.ndarray
        Radial distance from origin
    colat : np.ndarray
        Colatitude angle in radians [0, π]
    lon : np.ndarray
        Longitude angle in radians [-π, π]
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon # type: ignore


def cart2eq(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to equatorial coordinates.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : np.ndarray
        Z coordinates

    Returns
    -------
    r : np.ndarray
        Radial distance from origin
    lat : np.ndarray
        Latitude (elevation) angle in radians [-π/2, π/2]
    lon : np.ndarray
        Longitude (azimuth) angle in radians [-π, π]
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


def wrapped_rad2deg(lat_r: np.ndarray, lon_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert equatorial coordinates from radians to degrees with wrapping.

    Parameters
    ----------
    lat_r : np.ndarray
        Latitude angles in radians
    lon_r : np.ndarray
        Longitude angles in radians

    Returns
    -------
    lat_d : np.ndarray
        Latitude angles in degrees [-90, 90]
    lon_d : np.ndarray
        Longitude angles in degrees [-180, 180)
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg) # type: ignore
    return lat_d, lon_d # type: ignore
