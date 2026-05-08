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
from typing import Any

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from scipy import constants

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.basemap import Basemap

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


def to_RGB(I: np.ndarray) -> np.ndarray:
    """
    Collapse frequency-band intensity maps into RGB channels.

    Parameters
    ----------
    I : np.ndarray
        Real-valued intensity array with shape ``(9, N_px)``. If more than nine
        bands are provided, bands 1..9 are used as in the upstream visualisation.

    Returns
    -------
    np.ndarray
        RGB intensity array with shape ``(3, N_px)``.
    """
    N_px = I.shape[1]
    I_copy = I.copy()
    if I.shape[0] != 9:
        I_copy = I_copy[list(range(1, 10)), :]
    I_rgb = I_copy.reshape((3, 3, N_px)).sum(axis=1)
    return I_rgb


def cmap_from_list(name: str, colors: Any, N: int = 256, gamma: float = 1.0) -> Any:
    """
    Build a Matplotlib colour map from colour samples.

    Parameters
    ----------
    name : str
        Colour map name.
    colors : Any
        Iterable of colour values or ``(value, colour)`` pairs.
    N : int, optional
        Number of RGB quantisation levels.
    gamma : float, optional
        Gamma correction value forwarded to Matplotlib.

    Returns
    -------
    Any
        Matplotlib linear segmented colour map.

    Raises
    ------
    ValueError
        If ``colors`` is not iterable.
    """
    from collections.abc import Sized  # noqa: PLC0415

    import matplotlib.colors  # noqa: PLC0415

    if not isinstance(colors, abc.Iterable):
        raise ValueError("colors must be iterable")

    if (
        isinstance(colors[0], Sized)
        and len(colors[0]) == 2
        and not isinstance(colors[0], str)
    ):
        vals, colors = zip(*colors, strict=False)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict: dict[str, list[tuple[Any, float, float]]] = {
        "red": [],
        "green": [],
        "blue": [],
        "alpha": [],
    }
    for val, color in zip(vals, colors, strict=False):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        cdict["red"].append((val, r, r))
        cdict["green"].append((val, g, g))
        cdict["blue"].append((val, b, b))
        cdict["alpha"].append((val, a, a))

    return matplotlib.colors.LinearSegmentedColormap(name, cdict, N, gamma)

def draw_ellipse(position, covariance, ax=None, **kwargs) -> None:
    """
    Draw an ellipse with a given position and covariance.

    Parameters
    ----------
    position : array-like, shape (2,)
        The (x, y) coordinates of the ellipse center.
    covariance : array-like, shape (2, 2) or (2,)
        The covariance matrix or variances for the ellipse.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, uses current axis.
    **kwargs : dict
        Additional keyword arguments passed to the Ellipse patch.
    """
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        print(width, height)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        print("Width", Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs).width)
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None) -> None:
    """
    Plot the Gaussian Mixture Model components as ellipses.

    Parameters
    ----------
    gmm : GaussianMixture
        Fitted Gaussian Mixture Model object.
    X : array-like, shape (n_samples, n_features)
        Data points used for fitting the GMM.
    label : bool, optional
        Whether to color data points by their assigned GMM component (default: True).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, uses current axis.
    """
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def draw_map(  # noqa: PLR0913
    I: np.ndarray,
    R: np.ndarray,
    lon_ticks: np.ndarray,
    catalog: Any | None = None,
    show_labels: bool = False,
    show_axis: bool = False,
    fig: Any | None = None,
    ax: Any | None = None,
    kmeans: bool = False,
    gaussian_mixture: bool = False,
) -> tuple[Any, Any, Any | None]:
    """
    Draw a spherical acoustic map using the upstream LAM projection style.

    Parameters
    ----------
    I : np.ndarray
        RGB intensity map with shape ``(3, N_px)``.
    R : np.ndarray
        Field sampling coordinates with shape ``(3, N_px)``.
    lon_ticks : np.ndarray
        Longitude tick positions in degrees.
    catalog : Any | None, optional
        Kept for upstream interface compatibility.
    show_labels : bool, optional
        Whether to show axis labels.
    show_axis : bool, optional
        Whether to show graticule labels.
    fig : Any | None, optional
        Existing Matplotlib figure.
    ax : Any | None, optional
        Existing Matplotlib axis.
    kmeans : bool, optional
        Kept for upstream interface compatibility.
    gaussian_mixture : bool, optional
        Kept for upstream interface compatibility.

    Returns
    -------
    tuple[Any, Any, Any | None]
        Figure, axis, and optional cluster centre. The cluster centre is always
        ``None`` for the compact inference-only implementation.
    """
    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])

    bm = Basemap(
        projection="mill",
        llcrnrlat=R_el_min,
        urcrnrlat=R_el_max,
        llcrnrlon=R_az_min,
        urcrnrlon=R_az_max,
        resolution="c",
        ax=ax,
    )
    bm_labels = [1, 0, 0, 1] if show_axis else [0, 0, 0, 0]
    bm.drawparallels(
        np.linspace(R_el_min, R_el_max, 5),
        color="w",
        dashes=[1, 0],
        labels=bm_labels,
        labelstyle="+/-",
        textcolor="#565656",
        zorder=0,
        linewidth=2,
    )
    bm.drawmeridians(
        lon_ticks,
        color="w",
        dashes=[1, 0],
        labels=bm_labels,
        labelstyle="+/-",
        textcolor="#565656",
        zorder=0,
        linewidth=2,
    )

    if show_labels:
        ax.set_xlabel("Azimuth", labelpad=20)
        ax.set_ylabel("Elevation", labelpad=40)

    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)

    N_px = I.shape[1]
    mycmap = cmap_from_list("mycmap", I.T, N=N_px)
    colours_cmap = np.arange(N_px)

    ax.tripcolor(
        triangulation,
        colours_cmap,
        cmap=mycmap,
        shading="gouraud",
        alpha=0.9,
        edgecolors="w",
        linewidth=0.1,
    )

    cluster_centre = None
    if kmeans:
        Npts = 18
        I_s = np.square(I.sum(axis=0))
        max_idx = I_s.argsort()[-Npts:][::-1]
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        km_res = KMeans(n_clusters=3).fit(x_y)
        clusters = km_res.cluster_centers_
        ax.scatter(R_x[max_idx], R_y[max_idx], c='b', s=5)
        ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)
        cluster_centre = bm(clusters[:, 0], clusters[:, 1], inverse=True)
    elif gaussian_mixture:
        Npts = 18
        I_s = np.square(I.sum(axis=0))
        max_idx = I_s.argsort()[-Npts:][::-1]
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        gmm = GaussianMixture(n_components=3, random_state=42).fit(x_y)
        plot_gmm(gmm, x_y)

    return fig, ax, cluster_centre
