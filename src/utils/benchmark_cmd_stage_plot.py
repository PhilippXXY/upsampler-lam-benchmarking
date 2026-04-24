"""Helpers for plotting CMD stage trajectories from consolidated benchmark CSV rows."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.benchmark_results import PLOT_LEGEND_MARKER_SIZE, PlotRenderContext, marker_for_row

CMD_STAGE_LINESTYLE_BY_VARIANT_KIND = {
    "dist": "-",
    "e2e_auxdis": "--",
    "e2e_upfroz": ":",
}
CMD_STAGE_KIND_LABELS = {
    "dist": "Distinct",
    "e2e_auxdis": "End-to-end",
    "e2e_upfroz": "Frozen upsampler",
}
CMD_STAGE_SEQUENCE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Upsampler", ("cmd_reference_to_upsampler_median",)),
    ("LAM Denoise1", ("cmd_reference_to_lam_denoise1_median",)),
    ("LAM Denoise2", ("cmd_reference_to_lam_denoise2_median",)),
    ("LAM Denoise3", ("cmd_reference_to_lam_denoise3_median",)),
    ("LAM D4 / Final", ("cmd_reference_to_lam_median", "cmd_reference_to_lam_denoise4_median")),
)
CMD_STAGE_PLOT_WIDTH_IN = 12.0
CMD_STAGE_PLOT_HEIGHT_IN = 6.8
CMD_STAGE_THREE_PANEL_WIDTH_IN = 15.2
CMD_STAGE_THREE_PANEL_HEIGHT_IN = 5.9
CMD_STAGE_PLOT_FILENAME = "combined.png"


@dataclass(frozen=True)
class CMDStagePlotStyle:
    """
    Rendering configuration for the consolidated CMD stage plot.

    Parameters
    ----------
    rc_params : dict[str, Any]
        Matplotlib rc overrides for this render.
    hide_title : bool
        Whether to suppress the figure title.
    save_vector_formats : Any
        Callback used to emit optional vector companions next to the PNG.
    modes : set[str]
        Active plot presentation modes forwarded to ``save_vector_formats``.
    png_dpi : int
        PNG export DPI.
    """

    rc_params: dict[str, Any]
    hide_title: bool
    save_vector_formats: Any
    modes: set[str]
    png_dpi: int


def _parse_optional_float(raw_value: str | float | int | None) -> float:
    """
    Parse an optional numeric value and fall back to ``nan``.

    Parameters
    ----------
    raw_value : str | float | int | None
        Raw CSV value or in-memory scalar.

    Returns
    -------
    float
        Parsed value, or ``nan`` when unavailable.
    """
    if raw_value is None or raw_value == "":
        return float("nan")
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_first_finite_row_value(
    row: dict[str, str],
    column_names: tuple[str, ...],
) -> float:
    """
    Return the first finite numeric value available under the requested CSV columns.

    Parameters
    ----------
    row : dict[str, str]
        Consolidated benchmark CSV row.
    column_names : tuple[str, ...]
        Candidate column names in priority order.

    Returns
    -------
    float
        First finite parsed value, or ``nan`` when none are available.
    """
    for column_name in column_names:
        value = _parse_optional_float(row.get(column_name))
        if math.isfinite(value):
            return value
    return float("nan")


def cmd_stage_series_for_row(
    row: dict[str, str],
) -> tuple[list[float], list[float]]:
    """
    Build one row's CMD-to-reference stage trajectory for the line plot.

    Parameters
    ----------
    row : dict[str, str]
        Consolidated benchmark CSV row.

    Returns
    -------
    tuple[list[float], list[float]]
        Stage x positions and finite CMD y-values for the row.
    """
    x_values: list[float] = []
    y_values: list[float] = []
    for stage_index, (_, column_names) in enumerate(CMD_STAGE_SEQUENCE):
        value = _resolve_first_finite_row_value(row, column_names)
        if not math.isfinite(value):
            continue
        x_values.append(float(stage_index))
        y_values.append(value)
    return x_values, y_values


def _cmd_stage_variant_kind_handles(  # type: ignore[no-any-unimported]
    render_context: PlotRenderContext,
) -> list[Line2D]:
    """
    Build line-plot legend handles for variant-kind styles.

    Parameters
    ----------
    render_context : PlotRenderContext
        Plot render context used to resolve marker shapes.

    Returns
    -------
    list[Line2D]
        Legend handles describing variant-kind line styles and markers.
    """
    handles: list[Line2D] = []
    for variant_kind in ("dist", "e2e_auxdis", "e2e_upfroz"):
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle=CMD_STAGE_LINESTYLE_BY_VARIANT_KIND[variant_kind],
                marker=marker_for_row(
                    {"variant_kind": variant_kind},
                    render_context,
                    three_panel=False,
                ),
                markersize=PLOT_LEGEND_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.4,
                linewidth=2.0,
                label=CMD_STAGE_KIND_LABELS[variant_kind],
            )
        )
    return handles


def _cmd_stage_family_handles(plot_rows: list[dict[str, str]]) -> list[Line2D]:
    """
    Build colour-only legend handles for all plotted model families.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Benchmark rows included in the stage plot.

    Returns
    -------
    list[Line2D]
        Legend handles keyed by family colour.
    """
    handles: list[Line2D] = []
    seen_family_ids: set[str] = set()
    for row in plot_rows:
        family_id = str(row.get("family_id", "")).strip()
        if family_id in seen_family_ids:
            continue
        seen_family_ids.add(family_id)
        family_label = (
            str(row.get("family_label", "")).strip()
            or family_id
            or str(row.get("variant_id", "")).strip()
        )
        family_label = family_label.removesuffix(" (No upsampler)")
        family_colour = str(row.get("family_colour", "")).strip() or "black"
        handles.append(
            Line2D(
                [],
                [],
                color=family_colour,
                linewidth=2.5,
                label=family_label,
            )
        )
    return handles


def _cmd_stage_finite_rows(
    plot_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], list[float], list[float]]]:
    """
    Collect finite CMD stage trajectories for the provided rows.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Consolidated benchmark CSV rows.

    Returns
    -------
    list[tuple[dict[str, str], list[float], list[float]]]
        Row plus finite stage series for every row with at least one finite CMD value.
    """
    finite_rows = [(row, *cmd_stage_series_for_row(row)) for row in plot_rows]
    return [(row, x_values, y_values) for row, x_values, y_values in finite_rows if y_values]


def _cmd_stage_bounds(
    finite_rows: list[tuple[dict[str, str], list[float], list[float]]],
) -> tuple[float, float]:
    """
    Compute a tight shared y-range for CMD stage plots.

    Parameters
    ----------
    finite_rows : list[tuple[dict[str, str], list[float], list[float]]]
        Finite CMD stage trajectories.

    Returns
    -------
    tuple[float, float]
        Lower and upper y-axis bounds.
    """
    all_y_values = [value for _, _, y_values in finite_rows for value in y_values]
    y_min = min(all_y_values)
    y_max = max(all_y_values)
    y_span = max(y_max - y_min, 0.05)
    y_padding = max(0.03, y_span * 0.12)
    return max(0.0, y_min - y_padding), min(1.0, y_max + y_padding)


def _plot_cmd_stage_lines(  # type: ignore[no-any-unimported]
    axis: Any,
    finite_rows: list[tuple[dict[str, str], list[float], list[float]]],
    render_context: PlotRenderContext,
    *,
    panel_variant_kind: str | None = None,
) -> None:
    """
    Render CMD stage trajectories on one axis.

    Parameters
    ----------
    axis : Any
        Matplotlib axis receiving the lines.
    finite_rows : list[tuple[dict[str, str], list[float], list[float]]]
        Finite CMD stage trajectories.
    render_context : PlotRenderContext
        Plot render context used to resolve markers.
    panel_variant_kind : str | None, optional
        When set, render a panel-specific simplified style for one variant kind.
    """
    sorted_rows = sorted(
        finite_rows,
        key=lambda item: str(item[0].get("variant_id", "")).strip() == "lam",
    )
    for row, x_values, y_values in sorted_rows:
        variant_kind = str(row.get("variant_kind", "")).strip()
        family_colour = str(row.get("family_colour", "")).strip() or "black"
        variant_id = str(row.get("variant_id", "")).strip()
        if panel_variant_kind is None:
            axis.plot(
                x_values,
                y_values,
                color=family_colour,
                linestyle=CMD_STAGE_LINESTYLE_BY_VARIANT_KIND.get(variant_kind, "-"),
                marker=marker_for_row(row, render_context, three_panel=False),
                markersize=7 if variant_id == "lam" else 6,
                markerfacecolor="white",
                markeredgecolor=family_colour,
                markeredgewidth=1.6,
                linewidth=2.6 if variant_id == "lam" else 1.5,
                alpha=1.0 if variant_id == "lam" else 0.58,
                zorder=3 if variant_id == "lam" else 2,
            )
            continue

        axis.plot(
            x_values,
            y_values,
            color=family_colour,
            linestyle="-",
            marker="o",
            markersize=7 if variant_id == "lam" else 5.5,
            markerfacecolor="white",
            markeredgecolor=family_colour,
            markeredgewidth=1.5,
            linewidth=2.6 if variant_id == "lam" else 1.7,
            alpha=1.0 if variant_id == "lam" else 0.78,
            zorder=3 if variant_id == "lam" else 2,
        )


def plot_cmd_stage_trajectories(  # type: ignore[no-any-unimported]
    plot_rows: list[dict[str, str]],
    output_path: Path,
    render_context: PlotRenderContext,
    style: CMDStagePlotStyle,
) -> Path | None:
    """
    Plot CMD-to-reference trajectories across model stages for all variants.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Consolidated benchmark rows included in plotted variant kinds.
    output_path : Path
        Output path for the generated stage plot.
    render_context : PlotRenderContext
        Plot render context used to resolve markers.
    style : CMDStagePlotStyle
        Rendering configuration for this plot.

    Returns
    -------
    Path | None
        Generated plot path, or ``None`` when the CSV contains no finite CMD stage values.
    """
    finite_rows = _cmd_stage_finite_rows(plot_rows)
    if not finite_rows:
        logging.info(
            "Skipping CMD stage trajectory plot because the consolidated CSV contains no "
            "finite stagewise CMD values."
        )
        return None

    stage_positions = list(range(len(CMD_STAGE_SEQUENCE)))
    stage_labels = [
        "Upsampler",
        "LAM\nDenoise1",
        "LAM\nDenoise2",
        "LAM\nDenoise3",
        "LAM\nD4 / Final",
    ]
    y_lower, y_upper = _cmd_stage_bounds(finite_rows)

    with plt.rc_context(rc=style.rc_params):
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(CMD_STAGE_PLOT_WIDTH_IN, CMD_STAGE_PLOT_HEIGHT_IN),
        )
        fig.subplots_adjust(right=0.98, bottom=0.33, top=0.91)

        _plot_cmd_stage_lines(axis, finite_rows, render_context)

        if not style.hide_title:
            axis.set_title("CMD to Reference Across Pipeline Stages")
        axis.set_xlabel("Pipeline Stage")
        axis.set_ylabel("CMD to Reference (median)")
        axis.set_xticks(stage_positions, stage_labels)
        axis.set_xlim(-0.2, len(stage_positions) - 0.8)
        axis.set_ylim(y_lower, y_upper)
        axis.set_axisbelow(True)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)

        fig.legend(
            handles=_cmd_stage_variant_kind_handles(render_context),
            loc="lower left",
            bbox_to_anchor=(0.10, 0.01),
            borderaxespad=0.0,
            title="Variant Style",
            ncol=3,
        )
        fig.legend(
            handles=_cmd_stage_family_handles(plot_rows),
            loc="lower right",
            bbox_to_anchor=(0.92, 0.01),
            borderaxespad=0.0,
            ncol=4,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=style.png_dpi)
        style.save_vector_formats(fig, output_path, style.modes)
        plt.close(fig)

    return output_path


def plot_cmd_stage_trajectories_three_panel(  # type: ignore[no-any-unimported]
    panels: list[tuple[str, str, list[dict[str, str]]]],
    output_path: Path,
    render_context: PlotRenderContext,
    style: CMDStagePlotStyle,
) -> Path | None:
    """
    Plot CMD-to-reference trajectories as a shared three-panel comparison.

    Parameters
    ----------
    panels : list[tuple[str, str, list[dict[str, str]]]]
        Ordered tuples containing variant kind, panel title, and panel rows.
    output_path : Path
        Output path for the generated three-panel plot.
    render_context : PlotRenderContext
        Plot render context used to resolve markers.
    style : CMDStagePlotStyle
        Rendering configuration for this plot.

    Returns
    -------
    Path | None
        Generated plot path, or ``None`` when no panel contains finite CMD stage values.
    """
    panel_finite_rows = [
        (variant_kind, panel_title, _cmd_stage_finite_rows(panel_rows))
        for variant_kind, panel_title, panel_rows in panels
    ]
    if not any(finite_rows for _, _, finite_rows in panel_finite_rows):
        logging.info(
            "Skipping CMD stage three-panel plot because the consolidated CSV contains no "
            "finite stagewise CMD values across panel subsets."
        )
        return None

    stage_positions = list(range(len(CMD_STAGE_SEQUENCE)))
    stage_labels = ["Up", "D1", "D2", "D3", "Final"]
    combined_finite_rows = [
        finite_row for _, _, finite_rows in panel_finite_rows for finite_row in finite_rows
    ]
    y_lower, y_upper = _cmd_stage_bounds(combined_finite_rows)
    legend_rows = [row for _, _, panel_rows in panels for row in panel_rows]

    with plt.rc_context(rc=style.rc_params):
        fig, axes = plt.subplots(
            1,
            len(panel_finite_rows),
            sharey=True,
            figsize=(CMD_STAGE_THREE_PANEL_WIDTH_IN, CMD_STAGE_THREE_PANEL_HEIGHT_IN),
        )
        axes_list = list(axes if isinstance(axes, (list, tuple)) else axes.flat)
        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.28 if not style.hide_title else 0.22,
            top=0.88 if not style.hide_title else 0.94,
            wspace=0.08,
        )

        for axis, panel_data in zip(axes_list, panel_finite_rows, strict=False):
            variant_kind, panel_title, finite_rows = panel_data
            if finite_rows:
                _plot_cmd_stage_lines(
                    axis,
                    finite_rows,
                    render_context,
                    panel_variant_kind=variant_kind,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="0.4",
                )

            axis.set_title(panel_title)
            axis.set_xticks(stage_positions, stage_labels)
            axis.set_xlim(-0.2, len(stage_positions) - 0.8)
            axis.set_ylim(y_lower, y_upper)
            axis.set_axisbelow(True)
            axis.grid(True, axis="y", linestyle="--", alpha=0.35)

        axes_list[0].set_ylabel("CMD to Reference (median)")
        if not style.hide_title:
            fig.suptitle("CMD to Reference Across Pipeline Stages")
            fig.supxlabel(
                "Stages: Up = Upsampler, D1-D3 = LAM denoise steps",
                y=0.01,
            )

        fig.legend(
            handles=_cmd_stage_family_handles(legend_rows),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07 if not style.hide_title else 0.01),
            borderaxespad=0.0,
            ncol=4,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=style.png_dpi)
        style.save_vector_formats(fig, output_path, style.modes)
        plt.close(fig)

    return output_path
