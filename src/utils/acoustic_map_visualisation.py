"""Combined acoustic map visualisation and media helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence, cast

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from lam_min.trainer.utils import draw_map, to_RGB

VISUALISATION_MODE_LATEX_FONT_ONLY = "latex-font-only"
VISUALISATION_MODE_NO_TITLE = "no-title"
VISUALISATION_MODE_SAVE_EPS = "save-eps"
VISUALISATION_MODE_SAVE_SVG = "save-svg"
VALID_VISUALISATION_MODES = [
    VISUALISATION_MODE_LATEX_FONT_ONLY,
    VISUALISATION_MODE_NO_TITLE,
    VISUALISATION_MODE_SAVE_EPS,
    VISUALISATION_MODE_SAVE_SVG,
]
DEFAULT_PNG_DPI = 150
VISUALISATION_EMA_ALPHA = 0.65
VISUALISATION_FONT_SIZE = 15
COMBINED_GIF_FILENAME = "combined.gif"
COMBINED_MP4_FILENAME = "combined.mp4"
FRAME_MAP_NDIM = 3
GRAYSCALE_NDIM = 2
RGB_CHANNELS = 3
RGBA_CHANNELS = 4


def positive_int(raw_value: str) -> int:
    """
    Parse a strictly positive integer CLI value.

    Parameters
    ----------
    raw_value : str
        Raw command-line value.

    Returns
    -------
    int
        Parsed positive integer.

    Raises
    ------
    ValueError
        If the value is not a positive integer.
    """
    value = int(raw_value)
    if value <= 0:
        raise ValueError("value must be positive")
    return value


def resolve_visualisation_modes(cli_modes: list[str] | None) -> set[str]:
    """
    Resolve visualisation mode values from the CLI.

    Parameters
    ----------
    cli_modes : list[str] | None
        Mode values passed by argparse.

    Returns
    -------
    set[str]
        Active mode identifiers.
    """
    return set(cli_modes or [])


def build_visualisation_rc_params(modes: set[str]) -> dict[str, Any]:
    """
    Build temporary Matplotlib rc parameters for visualisation plots.

    Parameters
    ----------
    modes : set[str]
        Active visualisation mode identifiers.

    Returns
    -------
    dict[str, Any]
        Rc parameter overrides.
    """
    rc_params: dict[str, Any] = {
        "font.size": VISUALISATION_FONT_SIZE,
        "axes.labelsize": VISUALISATION_FONT_SIZE,
        "xtick.labelsize": VISUALISATION_FONT_SIZE,
        "ytick.labelsize": VISUALISATION_FONT_SIZE,
        "legend.fontsize": VISUALISATION_FONT_SIZE,
        "axes.titlesize": VISUALISATION_FONT_SIZE,
        "figure.titlesize": VISUALISATION_FONT_SIZE,
        "figure.labelsize": VISUALISATION_FONT_SIZE,
    }
    if VISUALISATION_MODE_LATEX_FONT_ONLY in modes:
        rc_params.update(
            {
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "text.usetex": False,
            }
        )
    return rc_params


def frame_timestamp_ms(frame_index: int, frame_width_ms: float) -> int:
    """
    Return the rounded frame timestamp in milliseconds.

    Parameters
    ----------
    frame_index : int
        Zero-based frame index.
    frame_width_ms : float
        Frame width in milliseconds.

    Returns
    -------
    int
        Rounded timestamp in milliseconds.
    """
    return int(round(frame_index * frame_width_ms))


def frame_png_path(frame_dir: Path, frame_index: int, frame_width_ms: float) -> Path:
    """
    Build the upstream-style PNG frame path.

    Parameters
    ----------
    frame_dir : Path
        Directory containing visualisation frames for one file.
    frame_index : int
        Zero-based frame index.
    frame_width_ms : float
        Frame width in milliseconds.

    Returns
    -------
    Path
        PNG path for the frame.
    """
    timestamp_ms = frame_timestamp_ms(frame_index, frame_width_ms)
    return frame_dir / f"frame_{frame_index:04d}_{timestamp_ms:06d}ms.png"


def frames_per_second(frame_width_ms: float) -> float:
    """
    Convert frame width in milliseconds to frames per second.

    Parameters
    ----------
    frame_width_ms : float
        Frame width in milliseconds.

    Returns
    -------
    float
        Frames per second.

    Raises
    ------
    ValueError
        If ``frame_width_ms`` is not positive.
    """
    if frame_width_ms <= 0.0:
        raise ValueError("frame_width_ms must be positive")
    return 1000.0 / frame_width_ms


def normalise_combined_rgb(frame_bands: NDArray[Any]) -> NDArray[Any]:
    """
    Collapse and normalise one frame's band maps into RGB intensities.

    Parameters
    ----------
    frame_bands : np.ndarray
        Band intensity maps with shape ``(bands, N_px)``.

    Returns
    -------
    np.ndarray
        Normalised RGB intensity map with shape ``(3, N_px)``.
    """
    frame_rgb = to_RGB(frame_bands)
    max_val = float(np.nanmax(frame_rgb)) if frame_rgb.size else 0.0
    if np.isfinite(max_val) and max_val > 0.0:
        frame_rgb = frame_rgb / max_val
    return frame_rgb


def _save_vector_formats(fig: plt.Figure, png_path: Path, modes: set[str]) -> None:
    """
    Save optional vector companions for a PNG frame.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    png_path : Path
        Primary PNG output path.
    modes : set[str]
        Active visualisation mode identifiers.
    """
    if VISUALISATION_MODE_SAVE_EPS in modes:
        ps_logger = logging.getLogger("matplotlib.backends.backend_ps")
        previous_level = ps_logger.level
        ps_logger.setLevel(logging.ERROR)
        try:
            fig.savefig(png_path.with_suffix(".eps"))
        finally:
            ps_logger.setLevel(previous_level)
    if VISUALISATION_MODE_SAVE_SVG in modes:
        fig.savefig(png_path.with_suffix(".svg"))


def render_combined_acoustic_maps(  # noqa: PLR0913
    band_maps: NDArray[Any],
    *,
    output_path: Path,
    file_id: str,
    frame_width_ms: float,
    r_field: NDArray[Any],
    lon_ticks: NDArray[Any],
    modes: set[str],
    png_dpi: int,
) -> tuple[list[Path], Path, Path]:
    """
    Render combined RGB acoustic maps and media for one input file.

    Parameters
    ----------
    band_maps : np.ndarray
        Model output intensity maps with shape ``(frames, bands, N_px)``.
    output_path : Path
        Inference run output directory.
    file_id : str
        Input file identifier.
    frame_width_ms : float
        Frame width in milliseconds.
    r_field : np.ndarray
        Field sampling coordinates with shape ``(3, N_px)``.
    lon_ticks : np.ndarray
        Longitude tick positions in degrees.
    modes : set[str]
        Active visualisation mode identifiers.
    png_dpi : int
        DPI used for PNG frame output.

    Returns
    -------
    tuple[list[Path], Path, Path]
        Generated PNG frame paths, GIF path, and MP4 path.

    Raises
    ------
    ValueError
        If ``band_maps`` does not have shape ``(frames, bands, N_px)``.
    """
    if band_maps.ndim != FRAME_MAP_NDIM:
        raise ValueError("band_maps must have shape (frames, bands, N_px)")

    frame_dir = output_path / file_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    hide_title = VISUALISATION_MODE_NO_TITLE in modes

    n_bands = band_maps.shape[1]
    running_max = np.zeros(n_bands)

    with plt.rc_context(rc=build_visualisation_rc_params(modes)):
        for i, frame_bands in enumerate(band_maps):
            t_ms = frame_timestamp_ms(i, frame_width_ms)

            # normalised per-band arrays reused for both per-band and combined render
            normalised_bands = np.empty_like(frame_bands, dtype=float)
            for b, band in enumerate(frame_bands):
                current_max = float(band.max())
                if i == 0:
                    running_max[b] = current_max
                else:
                    running_max[b] = (
                        VISUALISATION_EMA_ALPHA * current_max
                        + (1 - VISUALISATION_EMA_ALPHA) * running_max[b]
                    )
                norm_val = running_max[b] if running_max[b] > 1e-10 else 1.0  # noqa: PLR2004
                band_normalised = np.clip(band / norm_val, 0, 1)
                normalised_bands[b] = band_normalised

                # --- per-band grayscale frame ---
                band_rgb = np.tile(band_normalised[np.newaxis], (3, 1))
                band_dir = frame_dir / "bands" / f"band{b:02d}"
                band_dir.mkdir(parents=True, exist_ok=True)
                band_png_path = band_dir / f"frame_{i:04d}_{t_ms:06d}ms_band{b:02d}.png"
                fig, axis = plt.subplots(1, 1, figsize=(10, 5))
                draw_map(
                    band_rgb,
                    r_field,
                    lon_ticks=lon_ticks,
                    catalog=None,
                    show_labels=True,
                    show_axis=True,
                    fig=fig,
                    ax=axis,
                    kmeans=False,
                    gaussian_mixture=False,
                )
                if not hide_title:
                    axis.set_title(f"{file_id}  —  t = {t_ms} ms  |  band {b}")
                fig.savefig(band_png_path, bbox_inches="tight", dpi=png_dpi)
                _save_vector_formats(fig, band_png_path, modes)
                plt.close(fig)

            # --- combined RGB frame ---
            frame_rgb = normalise_combined_rgb(normalised_bands)
            png_path = frame_png_path(frame_dir, i, frame_width_ms)
            fig, axis = plt.subplots(1, 1, figsize=(10, 5))
            draw_map(
                frame_rgb,
                r_field,
                lon_ticks=lon_ticks,
                catalog=None,
                show_labels=True,
                show_axis=True,
                fig=fig,
                ax=axis,
                kmeans=False,
                gaussian_mixture=False,
            )
            if not hide_title:
                axis.set_title(f"{file_id}  —  t = {t_ms} ms")
            fig.savefig(png_path, bbox_inches="tight", dpi=png_dpi)
            _save_vector_formats(fig, png_path, modes)
            plt.close(fig)
            frame_paths.append(png_path)

    gif_path, mp4_path = write_media_from_frames(
        frame_paths,
        output_dir=frame_dir,
        fps=frames_per_second(frame_width_ms),
    )
    return frame_paths, gif_path, mp4_path


def _as_rgb_uint8(frame: NDArray[Any]) -> NDArray[np.uint8]:
    """
    Convert an image frame to RGB uint8.

    Parameters
    ----------
    frame : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        RGB uint8 image.
    """
    if frame.dtype != np.uint8:
        max_value = 1.0 if np.issubdtype(frame.dtype, np.floating) else 255.0
        frame = np.clip(frame, 0, max_value) * (255.0 / max_value)
        frame = frame.astype(np.uint8)
    if frame.ndim == GRAYSCALE_NDIM:
        return np.asarray(np.repeat(frame[:, :, np.newaxis], RGB_CHANNELS, axis=2), dtype=np.uint8)
    if frame.shape[2] == 1:
        return np.asarray(np.repeat(frame, RGB_CHANNELS, axis=2), dtype=np.uint8)
    if frame.shape[2] >= RGBA_CHANNELS:
        alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
        rgb = frame[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return np.asarray(rgb, dtype=np.uint8)
    return np.asarray(frame[:, :, :3], dtype=np.uint8)


def _even(value: int) -> int:
    """
    Round an image dimension up to an even value.

    Parameters
    ----------
    value : int
        Input dimension.

    Returns
    -------
    int
        Even output dimension.
    """
    return value if value % 2 == 0 else value + 1


def _media_shape(frame_paths: Sequence[Path]) -> tuple[int, int]:
    """
    Resolve the padded media frame shape.

    Parameters
    ----------
    frame_paths : Sequence[Path]
        PNG frame paths.

    Returns
    -------
    tuple[int, int]
        Target height and width.

    Raises
    ------
    ValueError
        If no frames are provided.
    """
    if not frame_paths:
        raise ValueError("Cannot build media without PNG frames")
    heights: list[int] = []
    widths: list[int] = []
    for frame_path in frame_paths:
        frame = _as_rgb_uint8(np.asarray(imageio.imread(frame_path)))
        heights.append(int(frame.shape[0]))
        widths.append(int(frame.shape[1]))
    return _even(max(heights)), _even(max(widths))


def _read_media_frame(frame_path: Path, target_shape: tuple[int, int]) -> NDArray[np.uint8]:
    """
    Read and pad one PNG frame for animated media.

    Parameters
    ----------
    frame_path : Path
        PNG frame path.
    target_shape : tuple[int, int]
        Target height and width.

    Returns
    -------
    np.ndarray
        Padded RGB uint8 frame.
    """
    frame = _as_rgb_uint8(np.asarray(imageio.imread(frame_path)))
    target_h, target_w = target_shape
    padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    padded[: frame.shape[0], : frame.shape[1], :] = frame
    return padded


def write_media_from_frames(
    frame_paths: Sequence[Path],
    *,
    output_dir: Path,
    fps: float,
) -> tuple[Path, Path]:
    """
    Assemble GIF and MP4 media from PNG frames.

    Parameters
    ----------
    frame_paths : Sequence[Path]
        Ordered PNG frame paths.
    output_dir : Path
        Directory receiving the generated media.
    fps : float
        Output media frames per second.

    Returns
    -------
    tuple[Path, Path]
        Generated GIF and MP4 paths.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_shape = _media_shape(frame_paths)
    gif_path = output_dir / COMBINED_GIF_FILENAME
    mp4_path = output_dir / COMBINED_MP4_FILENAME
    duration = 1.0 / fps

    gif_writer_context = cast(Any, imageio.get_writer(gif_path, mode="I", duration=duration))
    with gif_writer_context as gif_writer:
        for frame_path in frame_paths:
            gif_writer.append_data(_read_media_frame(frame_path, target_shape))

    mp4_writer_context = cast(
        Any,
        imageio.get_writer(mp4_path, fps=fps, codec="libx264", macro_block_size=1),
    )
    with mp4_writer_context as mp4_writer:
        for frame_path in frame_paths:
            mp4_writer.append_data(_read_media_frame(frame_path, target_shape))

    return gif_path, mp4_path
