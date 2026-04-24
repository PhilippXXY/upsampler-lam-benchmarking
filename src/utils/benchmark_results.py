"""Helpers for reading and replaying consolidated benchmark result artefacts."""

from __future__ import annotations

import logging
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection

from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from utils.model_variants import (
    FAMILY_COLOURS,
    FAMILY_LABELS,
    MARKER_BY_VARIANT_KIND,
    VARIANT_BY_ID,
)

BENCHMARK_RESULTS_CSV_NAME = "benchmark_metrics.csv"
BENCHMARK_PLOT_BASENAME = "benchmark_metrics.png"
RIGHT_EDGE_THRESHOLD = 0.82
TOP_EDGE_THRESHOLD = 0.88
SCATTER_MARKER_SIZE = 150
ANNOTATION_FONT_SIZE = 12
ANNOTATION_TEXT_WRAP_WIDTH = 14
ANNOTATION_BBOX_ALPHA = 0.9
ANNOTATION_AXIS_MARGIN_PX = 4.0
ANNOTATION_BBOX_EXPAND_X = 1.04
ANNOTATION_BBOX_EXPAND_Y = 1.12
ANNOTATION_MARKER_PADDING_PX = 4.0
ANNOTATION_OFFSET_LEVELS_PT = (8, 14, 22, 30, 40)
MIN_LINE_PATH_POINTS = 2
PLOT_LEGEND_MARKER_SIZE = 13
THREE_PANEL_VARIANT_KIND_ORDER = (
    "dist",
    "e2e_upfroz",
    "e2e_auxdis",
)
THREE_PANEL_TITLE_BY_VARIANT_KIND = {
    "dist": "Distinct",
    "e2e_upfroz": "Aligned",
    "e2e_auxdis": "End-to-End",
}


@dataclass(frozen=True)
class PlotRenderContext:
    """
    Plot-style decisions resolved once per CSV before rendering.

    Parameters
    ----------
    force_three_panel_circle_markers : bool
        Whether three-panel plots should force circle markers in single-dataset mode.
    """

    force_three_panel_circle_markers: bool


def filter_rows_by_variant_kinds(
    rows: list[dict[str, str]],
    variant_kinds: Collection[str] | None,
    *,
    include_lam_variant: bool,
) -> list[dict[str, str]]:
    """
    Filter benchmark rows to the requested variant-kind set.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Candidate benchmark rows.
    variant_kinds : Collection[str] | None
        Variant kinds to include. ``None`` keeps all variant kinds.
    include_lam_variant : bool
        Whether to always include the row whose ``variant_id`` equals ``lam``.

    Returns
    -------
    list[dict[str, str]]
        Rows matching the requested variant kinds.
    """
    retained_variant_kinds = set(variant_kinds) if variant_kinds is not None else None
    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        if include_lam_variant and str(row.get("variant_id", "")).strip() == "lam":
            filtered_rows.append(row)
            continue
        variant_kind = str(row.get("variant_kind", "")).strip()
        if retained_variant_kinds is None or variant_kind in retained_variant_kinds:
            filtered_rows.append(row)
    return filtered_rows


def build_three_panel_rows(
    rows: list[dict[str, str]],
) -> list[tuple[str, str, list[dict[str, str]]]]:
    """
    Build ordered row subsets for the fixed three-panel benchmark composite.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Candidate benchmark rows.

    Returns
    -------
    list[tuple[str, str, list[dict[str, str]]]]
        Ordered tuples containing ``variant_kind``, panel title, and panel rows.
    """
    panel_rows: list[tuple[str, str, list[dict[str, str]]]] = []
    for variant_kind in THREE_PANEL_VARIANT_KIND_ORDER:
        title = THREE_PANEL_TITLE_BY_VARIANT_KIND[variant_kind]
        panel_rows.append(
            (
                variant_kind,
                title,
                filter_rows_by_variant_kinds(
                    rows,
                    (variant_kind,),
                    include_lam_variant=True,
                ),
            )
        )
    return panel_rows


def resolve_results_csv(repo_root: Path, raw_results_path: str) -> tuple[Path, Path]:
    """
    Resolve an existing benchmark results directory or consolidated CSV path.

    Parameters
    ----------
    repo_root : Path
        Repository root used to resolve relative paths.
    raw_results_path : str
        CLI path passed via ``--results`` or ``--result``. The path may point to either
        the benchmark-comparison directory or directly to the consolidated CSV file.

    Returns
    -------
    tuple[Path, Path]
        Resolved results directory and the consolidated CSV path.
    """
    results_dir = Path(raw_results_path)
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results path does not exist: {results_dir}")

    if results_dir.is_file():
        return results_dir.parent, results_dir

    comparison_csv = results_dir / BENCHMARK_RESULTS_CSV_NAME
    if not comparison_csv.is_file():
        raise FileNotFoundError(
            f"Expected {BENCHMARK_RESULTS_CSV_NAME} inside results directory: {results_dir}"
        )
    return results_dir, comparison_csv


def build_plot_render_context(
    *,
    force_three_panel_circle_markers: bool,
) -> PlotRenderContext:
    """
    Resolve plot-style behaviour from the active single-dataset plot modes.

    Parameters
    ----------
    force_three_panel_circle_markers : bool
        Whether three-panel plots should force circle markers in single-dataset mode.

    Returns
    -------
    PlotRenderContext
        Resolved plot render context for the current plot type.
    """
    return PlotRenderContext(
        force_three_panel_circle_markers=force_three_panel_circle_markers,
    )


def marker_for_row(
    row: dict[str, str],
    render_context: PlotRenderContext,
    *,
    three_panel: bool,
) -> str:
    """
    Resolve the marker shape for one plotted row in the current render context.

    Parameters
    ----------
    row : dict[str, str]
        Benchmark row data.
    render_context : PlotRenderContext
        Current plot render context.
    three_panel : bool
        Whether the plot is a three-panel plot.

    Returns
    -------
    str
        Marker shape for the plotted row.
    """
    if three_panel and render_context.force_three_panel_circle_markers:
        return "o"
    return str(MARKER_BY_VARIANT_KIND[row["variant_kind"]])


def _build_variant_kind_legend_handles() -> list[Line2D]:
    """
    Build the legacy single-dataset legend keyed by variant kind.

    Returns
    -------
    list[Line2D]
        Legend handles for the legacy single-dataset plot.
    """
    return [
        Line2D(
            [],
            [],
            marker=MARKER_BY_VARIANT_KIND["dist"],
            linestyle="None",
            color="black",
            markersize=PLOT_LEGEND_MARKER_SIZE,
            label="Distinct",
        ),
        Line2D(
            [],
            [],
            marker=MARKER_BY_VARIANT_KIND["e2e_auxdis"],
            linestyle="None",
            color="black",
            markersize=PLOT_LEGEND_MARKER_SIZE,
            label="End-to-end",
        ),
        Line2D(
            [],
            [],
            marker=MARKER_BY_VARIANT_KIND["e2e_upfroz"],
            linestyle="None",
            color="black",
            markersize=PLOT_LEGEND_MARKER_SIZE,
            label="Frozen upsampler",
        ),
    ]


def build_legend_handles(
    plot_rows: list[dict[str, str]],
    render_context: PlotRenderContext,
    *,
    three_panel: bool,
) -> list[Line2D]:
    """
    Resolve the legend handles required for the current plot type.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Benchmark row data for the current plot.
    render_context : PlotRenderContext
        Current plot render context.
    three_panel : bool
        Whether the plot is a three-panel plot.

    Returns
    -------
    list[Line2D]
        Legend handles for the current plot type and data.
    """
    return _build_variant_kind_legend_handles()


def should_annotate_points(
    render_context: PlotRenderContext,
    *,
    three_panel: bool,
) -> bool:
    """
    Return whether per-point text annotations should be rendered for this plot.

    Parameters
    ----------
    render_context : PlotRenderContext
        Current plot render context.
    three_panel : bool
        Whether the plot is a three-panel plot.

    Returns
    -------
    bool
        True if per-point text annotations should be rendered, False otherwise.
    """
    return True


def resolve_frame_width_ms(
    rows: list[dict[str, str]],
    *,
    fallback_frame_width_ms: float,
) -> float:
    """
    Resolve the frame width used for plot labels from CSV metadata or a fallback.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Consolidated benchmark CSV rows.
    fallback_frame_width_ms : float
        Base-config fallback used when the CSV predates ``frame_width_ms`` persistence.

    Returns
    -------
    float
        Frame width in milliseconds for plot labels.
    """
    csv_values = []
    for row in rows:
        raw_value = row.get("frame_width_ms")
        if raw_value is None or raw_value == "":
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            csv_values.append(value)
    if not csv_values:
        return fallback_frame_width_ms

    distinct_values = sorted({f"{value:.12g}" for value in csv_values})
    if len(distinct_values) > 1:
        logging.warning(
            "Comparison CSV contains multiple frame_width_ms values (%s); using the first row.",
            ", ".join(distinct_values),
        )
    return csv_values[0]


def apply_current_style_guide(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Override CSV styling fields with the current style guide from ``model_variants``.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Consolidated benchmark CSV rows.

    Returns
    -------
    list[dict[str, str]]
        Copies of the rows with plot-facing metadata refreshed from the current registry. When the
        row's ``variant_id`` matches a known retained variant, the override updates:
        ``family_id``, ``family_label``, ``variant_kind``, ``infer_model_name``, and
        ``family_colour``. If the exact variant is unknown but ``family_id`` is known, the helper
        still refreshes ``family_label`` and ``family_colour`` from the current family registry.
    """
    overridden_rows: list[dict[str, str]] = []
    for row in rows:
        updated_row = dict(row)
        variant_id = str(row.get("variant_id", "")).strip()
        variant = VARIANT_BY_ID.get(variant_id)
        if variant is not None:
            updated_row["family_id"] = variant.family_id
            updated_row["family_label"] = variant.family_label
            updated_row["variant_kind"] = variant.variant_kind
            updated_row["infer_model_name"] = variant.infer_model_name
            updated_row["family_colour"] = variant.colour
            overridden_rows.append(updated_row)
            continue

        family_id = str(row.get("family_id", "")).strip()
        if family_id in FAMILY_LABELS:
            updated_row["family_label"] = FAMILY_LABELS[family_id]
        if family_id in FAMILY_COLOURS:
            updated_row["family_colour"] = FAMILY_COLOURS[family_id]
        overridden_rows.append(updated_row)
    return overridden_rows


def annotate_points(  # noqa: PLR0913
    axis: Any,
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    colours: list[str],
    *,
    obstacle_x_values: list[float],
    obstacle_y_values: list[float],
) -> list[Any]:
    """
    Add wrapped labels near scatter points while avoiding collisions where possible.

    Parameters
    ----------
    axis : Any
        Matplotlib axis that receives the annotations.
    x_values : list[float]
        X coordinates for the annotation anchors.
    y_values : list[float]
        Y coordinates for the annotation anchors.
    labels : list[str]
        Annotation text labels.
    colours : list[str]
        Text colours for the annotations.
    obstacle_x_values : list[float]
        X coordinates of all plotted marker positions that labels must avoid.
    obstacle_y_values : list[float]
        Y coordinates of all plotted marker positions that labels must avoid.

    Returns
    -------
    list[Any]
        Annotation artists added to the axis.
    """
    if not x_values:
        return []

    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    axis_bbox = axis.get_window_extent(renderer)
    marker_bboxes = marker_bboxes_in_display(axis, obstacle_x_values, obstacle_y_values)
    line_paths = line_paths_in_display(axis)
    placed_bboxes: list[Bbox] = []
    annotations: list[Any] = []

    for x_value, y_value, label, colour in zip(x_values, y_values, labels, colours, strict=False):
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue

        wrapped_label = textwrap.fill(
            label,
            width=ANNOTATION_TEXT_WRAP_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )

        best_annotation = None
        best_bbox = None
        best_score = None
        for candidate_index, (x_offset, y_offset, horizontal_align, vertical_align) in enumerate(
            ordered_annotation_candidates(axis, x_value, y_value)
        ):
            annotation = axis.annotate(
                wrapped_label,
                xy=(x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_align,
                va=vertical_align,
                fontsize=ANNOTATION_FONT_SIZE,
                color=colour,
                annotation_clip=True,
                clip_on=True,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": ANNOTATION_BBOX_ALPHA,
                },
            )
            annotation_bbox = annotation.get_window_extent(renderer).expanded(
                ANNOTATION_BBOX_EXPAND_X,
                ANNOTATION_BBOX_EXPAND_Y,
            )
            overlap_area = sum(
                bbox_overlap_area(annotation_bbox, placed_bbox) for placed_bbox in placed_bboxes
            ) + sum(
                bbox_overlap_area(annotation_bbox, marker_bbox) for marker_bbox in marker_bboxes
            )
            line_overlap_count = sum(
                1
                for line_path in line_paths
                if line_path.intersects_bbox(annotation_bbox, filled=False)
            )
            score = (
                bbox_outside_area(annotation_bbox, axis_bbox),
                overlap_area,
                line_overlap_count,
                abs(x_offset) + abs(y_offset),
                candidate_index,
            )
            if best_score is None or score < best_score:  # type: ignore[unreachable]
                if best_annotation is not None:
                    best_annotation.remove()  # type: ignore[unreachable]
                best_annotation = annotation
                best_bbox = annotation_bbox
                best_score = score
            else:
                annotation.remove()  # type: ignore[unreachable]
            if score[0] == 0.0 and score[1] == 0.0 and score[2] == 0:
                break

        if best_annotation is None or best_bbox is None:
            continue
        annotations.append(best_annotation)
        placed_bboxes.append(best_bbox)

    return annotations


def ordered_annotation_candidates(
    axis: Any,
    x_value: float,
    y_value: float,
) -> list[tuple[float, float, str, str]]:
    """
    Return candidate annotation offsets ordered by likely fit inside the current axis.

    Parameters
    ----------
    axis : Any
        Matplotlib axis receiving the annotation.
    x_value : float
        X coordinate of the annotation anchor.
    y_value : float
        Y coordinate of the annotation anchor.

    Returns
    -------
    list[tuple[float, float, str, str]]
        Candidate ``(x_offset_pt, y_offset_pt, ha, va)`` tuples in priority order.
    """
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_span = (x_max - x_min) if x_max != x_min else 1.0
    y_span = (y_max - y_min) if y_max != y_min else 1.0
    x_ratio = (x_value - x_min) / x_span
    y_ratio = (y_value - y_min) / y_span

    preferred_x_sign = -1 if x_ratio > RIGHT_EDGE_THRESHOLD else 1
    preferred_y_sign = -1 if y_ratio > TOP_EDGE_THRESHOLD else 1

    candidates: list[tuple[float, float, str, str]] = []
    for distance in ANNOTATION_OFFSET_LEVELS_PT:
        for x_sign, y_sign in (
            (preferred_x_sign, preferred_y_sign),
            (preferred_x_sign, -preferred_y_sign),
            (-preferred_x_sign, preferred_y_sign),
            (-preferred_x_sign, -preferred_y_sign),
        ):
            candidates.append(
                (
                    float(x_sign * distance),
                    float(y_sign * distance),
                    "left" if x_sign > 0 else "right",
                    "bottom" if y_sign > 0 else "top",
                )
            )
        axis_distance = float(distance + 4)
        candidates.extend(
            [
                (
                    0.0,
                    float(preferred_y_sign * axis_distance),
                    "center",
                    "bottom" if preferred_y_sign > 0 else "top",
                ),
                (
                    float(preferred_x_sign * axis_distance),
                    0.0,
                    "left" if preferred_x_sign > 0 else "right",
                    "center",
                ),
                (
                    0.0,
                    float(-preferred_y_sign * axis_distance),
                    "center",
                    "bottom" if -preferred_y_sign > 0 else "top",
                ),
                (
                    float(-preferred_x_sign * axis_distance),
                    0.0,
                    "left" if -preferred_x_sign > 0 else "right",
                    "center",
                ),
            ]
        )
    return candidates


def marker_bboxes_in_display(
    axis: Any,
    x_values: list[float],
    y_values: list[float],
) -> list[Bbox]:
    """
    Build display-space marker exclusion boxes for plotted points.

    Parameters
    ----------
    axis : Any
        Matplotlib axis containing the markers.
    x_values : list[float]
        X coordinates of plotted markers.
    y_values : list[float]
        Y coordinates of plotted markers.

    Returns
    -------
    list[Bbox]
        Marker exclusion boxes in display coordinates.
    """
    half_size_px = (
        math.sqrt(SCATTER_MARKER_SIZE / math.pi) * axis.figure.dpi / 72.0
        + ANNOTATION_MARKER_PADDING_PX
    )
    marker_bboxes: list[Bbox] = []
    for x_value, y_value in zip(x_values, y_values, strict=False):
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        x_display, y_display = axis.transData.transform((x_value, y_value))
        marker_bboxes.append(
            Bbox.from_extents(
                x_display - half_size_px,
                y_display - half_size_px,
                x_display + half_size_px,
                y_display + half_size_px,
            )
        )
    return marker_bboxes


def line_paths_in_display(axis: Any) -> list[Any]:
    """
    Return all current line paths on the axis in display coordinates.

    Parameters
    ----------
    axis : Any
        Matplotlib axis whose lines should be considered as secondary label obstacles.

    Returns
    -------
    list[Any]
        Display-space paths for all visible line artists on the axis.
    """
    return [
        line.get_path().transformed(line.get_transform())
        for line in axis.lines
        if len(line.get_xdata()) >= MIN_LINE_PATH_POINTS
    ]


def bbox_overlap_area(first_bbox: Bbox, second_bbox: Bbox) -> float:
    """
    Return the overlap area between two display-space bounding boxes.

    Parameters
    ----------
    first_bbox : Bbox
        First bounding box.
    second_bbox : Bbox
        Second bounding box.

    Returns
    -------
    float
        Overlap area in square pixels, or ``0.0`` when the boxes do not intersect.
    """
    overlap_x0 = max(first_bbox.x0, second_bbox.x0)
    overlap_y0 = max(first_bbox.y0, second_bbox.y0)
    overlap_x1 = min(first_bbox.x1, second_bbox.x1)
    overlap_y1 = min(first_bbox.y1, second_bbox.y1)
    if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
        return 0.0
    return float((overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0))


def bbox_outside_area(candidate_bbox: Bbox, axis_bbox: Bbox) -> float:
    """
    Return the area of a label bbox that would fall outside the visible axis box.

    Parameters
    ----------
    candidate_bbox : Bbox
        Candidate label bounding box in display coordinates.
    axis_bbox : Bbox
        Axis bounding box in display coordinates.

    Returns
    -------
    float
        Area outside the visible axis box in square pixels.
    """
    constrained_axis_bbox = Bbox.from_extents(
        axis_bbox.x0 + ANNOTATION_AXIS_MARGIN_PX,
        axis_bbox.y0 + ANNOTATION_AXIS_MARGIN_PX,
        axis_bbox.x1 - ANNOTATION_AXIS_MARGIN_PX,
        axis_bbox.y1 - ANNOTATION_AXIS_MARGIN_PX,
    )
    intersection_area = bbox_overlap_area(candidate_bbox, constrained_axis_bbox)
    candidate_area = max(0.0, candidate_bbox.width) * max(0.0, candidate_bbox.height)
    return max(0.0, candidate_area - intersection_area)
