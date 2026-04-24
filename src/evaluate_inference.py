"""
Batch runner for comparing retained inference variants.

This script executes `src/infer.py` for multiple retained variant configurations, extracts
runtime/evaluation metrics from each run's output, writes a
consolidated CSV with all metrics, and creates a plot (no ML logic).
As a result, this file is mainly generated using AI assistance.
If you want to change the plot style, this is the file to edit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from utils.benchmark_cmd_stage_plot import (
    CMD_STAGE_PLOT_FILENAME,
    CMDStagePlotStyle,
    plot_cmd_stage_trajectories,
    plot_cmd_stage_trajectories_three_panel,
)
from utils.benchmark_results import (
    BENCHMARK_PLOT_BASENAME,
    BENCHMARK_RESULTS_CSV_NAME,
    SCATTER_MARKER_SIZE,
    PlotRenderContext,
    annotate_points,
    apply_current_style_guide,
    build_legend_handles,
    build_plot_render_context,
    build_three_panel_rows,
    filter_rows_by_variant_kinds,
    marker_for_row,
    resolve_frame_width_ms,
    resolve_results_csv,
    should_annotate_points,
)
from utils.benchmarking import AggregatedBenchmarkMetrics, aggregate_metrics_json
from utils.model_variants import (
    RetainedVariantSpec,
    expand_target_selectors,
    supported_selector_ids,
)
from utils.utils import select_memory_display_unit

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class ModelResult:
    """
    Consolidated benchmark metrics for one retained variant.

    Parameters
    ----------
    variant_id : str
        Exact retained variant identifier.
    family_id : str
        Family identifier used for grouping variants.
    family_label : str
        Human-readable family label for legends and annotations.
    variant_kind : str
        Variant kind, for example ``dist`` or ``e2e_upfroz``.
    infer_model_name : str
        Runtime model class name used by ``src/infer.py``.
    family_colour : str
        Colour assigned to the family.
    run_dir : Path
        Directory containing the raw full-evaluation inference artefacts.
    metrics_json_path : Path
        Path to the raw full-evaluation ``metrics_*.json`` file.
    normalised_metrics_json_path : Path | None
        Path to the normalised runtime-only ``metrics_*.json`` file, if produced.
    files_processed : int
        Number of files processed in the raw full-evaluation run.
    total_frames : int
        Number of frames processed in the raw full-evaluation run.
    frame_width_ms : float
        Frame width in milliseconds used for the evaluated benchmark run.
    latency_per_frame_ms : float
        End-to-end latency per frame in milliseconds.
    gflops_per_frame : float
        End-to-end GFLOPs per frame.
    lam_latency_per_frame_ms : float
        LAM-only latency per frame in milliseconds.
    lam_gflops_per_frame : float
        LAM-only GFLOPs per frame.
    memory_peak_max_mb : float
        Worst-case raw end-to-end peak memory across files.
    memory_peak_median_mb : float
        Median raw end-to-end peak memory across files.
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
    normalised_memory_peak_max_mb : float
        Worst-case normalised end-to-end peak memory across files.
    normalised_memory_peak_median_mb : float
        Median normalised end-to-end peak memory across files.
    normalised_memory_duration_sec : float
        Canonical duration used for the normalised memory benchmark.
    localisation_error_deg : float
        Full-evaluation localisation error in degrees.
    localisation_recall : float
        Full-evaluation localisation recall on the native 0..1 scale.
    total_params : int
        Total number of model parameters.
    """

    variant_id: str
    family_id: str
    family_label: str
    variant_kind: str
    infer_model_name: str
    family_colour: str
    run_dir: Path
    metrics_json_path: Path
    normalised_metrics_json_path: Path | None
    files_processed: int
    total_frames: int
    frame_width_ms: float
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
    normalised_memory_peak_max_mb: float
    normalised_memory_peak_median_mb: float
    normalised_memory_duration_sec: float
    localisation_error_deg: float
    localisation_recall: float
    total_params: int


@dataclass(frozen=True)
class ScatterSpec:
    """
    Configuration for generating one scatter plot image.

    Parameters
    ----------
    y_values : list[float]
        Y-axis values for the scatter plot.
    y_label : str
        Label for the y-axis.
    title : str
        Figure title.
    output_path : Path
        Output image path.
    lam_baseline : float | None, optional
        Optional LAM baseline y-value.
    lam_baseline_x : float | None, optional
        Optional LAM baseline x-value used for a vertical guide line.
    lam_baseline_colour : str, optional
        Colour used to render the optional LAM baseline line and y-axis marker.
    lam_baseline_label : str, optional
        Label for the optional LAM baseline.
    """

    y_values: list[float]
    y_label: str
    title: str
    output_path: Path
    lam_baseline: float | None = None
    lam_baseline_x: float | None = None
    lam_baseline_colour: str = "black"
    lam_baseline_label: str = "LAM-only baseline"


@dataclass(frozen=True)
class BrokenAxisGap:
    """
    Adjacent y-value gap that should trigger a broken-axis companion plot.

    Parameters
    ----------
    lower_value : float
        Lower value of the adjacent pair.
    upper_value : float
        Upper value of the adjacent pair.
    ratio : float
        Ratio ``upper_value / lower_value`` that triggered the break.
    """

    lower_value: float
    upper_value: float
    ratio: float


@dataclass(frozen=True)
class BrokenAxisSegment:
    """
    One visible y-axis segment in a broken-axis companion plot.

    Parameters
    ----------
    min_value : float
        Minimum plotted data value inside the segment.
    max_value : float
        Maximum plotted data value inside the segment.
    lower_limit : float
        Lower axis limit for the visible segment.
    upper_limit : float
        Upper axis limit for the visible segment.
    point_count : int
        Number of plotted points inside the visible segment.
    """

    min_value: float
    max_value: float
    lower_limit: float
    upper_limit: float
    point_count: int


@dataclass(frozen=True)
class XAxisSpec:
    """
    X-axis configuration for one family of scatter plots.

    Parameters
    ----------
    x_values : list[float]
        X-axis values.
    label : str
        X-axis label.
    title_suffix : str
        Title suffix appended after ``vs``.
    output_suffix : str
        File-name suffix used for plot outputs.
    """

    x_values: list[float]
    label: str
    title_suffix: str
    output_suffix: str


@dataclass(frozen=True)
class PlotScope:
    """
    Plot scope describing one subset of retained variant kinds.

    Parameters
    ----------
    scope_id : str
        Stable identifier used in output file names.
    title_suffix : str
        Human-readable suffix appended to plot titles.
    variant_kinds : frozenset[str] | None
        Variant kinds included in the scope. ``None`` means every plotted variant kind.
    """

    scope_id: str
    title_suffix: str
    variant_kinds: frozenset[str] | None


@dataclass(frozen=True)
class MemoryPlotTemplate:
    """
    Shared metadata for one memory-oriented scatter metric.

    Parameters
    ----------
    column_name : str
        Source CSV column used for y-values.
    output_stem : str
        Output suffix used in directory names.
    title : str
        Base figure title before adding the x-axis suffix.
    y_label_prefix : str
        Prefix for the y-axis label before appending display units.
    scale : float
        Value scaling factor applied to raw memory values.
    unit : str
        Unit label corresponding to ``scale``.
    lam_baseline : float | None
        Optional LAM baseline in scaled units.
    """

    column_name: str
    output_stem: str
    title: str
    y_label_prefix: str
    scale: float
    unit: str
    lam_baseline: float | None


@dataclass(frozen=True)
class ThreePanelSubset:
    """
    Fully prepared data for one panel in the three-panel composite plot.

    Parameters
    ----------
    variant_kind : str
        Variant kind represented by the panel.
    title : str
        Panel title.
    rows : list[dict[str, str]]
        CSV rows represented in the panel.
    x_values : list[float]
        Panel-specific x-values.
    y_values : list[float]
        Panel-specific y-values.
    """

    variant_kind: str
    title: str
    rows: list[dict[str, str]]
    x_values: list[float]
    y_values: list[float]


DEFAULT_TARGET_SELECTORS = [
    "bicubiclam",
    "uplam",
    "srcnnlam",
    "imdnlam",
    "safmnlam",
    "ganlam",
    "ainnlam",
]
LOCATA_TARGET_SELECTORS = ["lam", *DEFAULT_TARGET_SELECTORS]
PLOT_MODE_LATEX_FONT_ONLY = "latex-font-only"
PLOT_MODE_NO_TITLE = "no-title"
PLOT_MODE_NO_FRAME_CONTEXT = "no-frame-context"
PLOT_MODE_SAVE_EPS = "save-eps"
PLOT_MODE_SAVE_SVG = "save-svg"
PLOT_MODE_THREE_PANEL_CIRCLE_MARKERS = "three-panel-circle-markers"
VALID_PLOT_MODES = [
    PLOT_MODE_LATEX_FONT_ONLY,
    PLOT_MODE_NO_TITLE,
    PLOT_MODE_NO_FRAME_CONTEXT,
    PLOT_MODE_SAVE_EPS,
    PLOT_MODE_SAVE_SVG,
    PLOT_MODE_THREE_PANEL_CIRCLE_MARKERS,
]
DEFAULT_BROKEN_Y_THRESHOLD = 1.5
DEFAULT_MEMORY_BROKEN_Y_THRESHOLD = 1.25
MIN_BROKEN_Y_VALUE_COUNT = 2
ANNOTATION_PRIORITY = {"dist": 0, "e2e_auxdis": 1, "e2e_upfroz": 2}
PLOTTED_VARIANT_KINDS = frozenset({"dist", "e2e_auxdis", "e2e_upfroz"})
BROKEN_AXIS_OUTER_PADDING_RATIO = 0.12
BROKEN_AXIS_INNER_PADDING_RATIO = 0.18
BROKEN_AXIS_MIN_PADDING_RATIO = 0.03
BROKEN_AXIS_MAX_GAP_PADDING_RATIO = 0.25
BROKEN_AXIS_BREAK_MARKER_HALF_HEIGHT = 0.55
BROKEN_AXIS_BREAK_MARKER_SIZE = 12
BROKEN_AXIS_MIN_HEIGHT_RATIO = 1.25
BROKEN_AXIS_MAX_HEIGHT_RATIO = 3.0
BROKEN_AXIS_HEIGHT_PER_POINT = 0.18
LOCALISATION_ERROR_X_LABEL = "Localisation Error (deg)"
LOCALISATION_RECALL_X_LABEL = "Localisation Recall"
PLOT_WIDTH_IN = 7.5
PLOT_HEIGHT_IN = 5.5
THREE_PANEL_PLOT_WIDTH_IN = 13.8
THREE_PANEL_FILENAME = "three_panel.png"
THREE_PANEL_BROKEN_Y_FILENAME = "three_panel_broken_y.png"
DEFAULT_PNG_DPI = 1200
PLOT_FONT_SIZE = 15
SHARED_X_AXIS_PADDING_RATIO = 0.05
PLOT_SCOPES = (
    PlotScope(scope_id="combined", title_suffix="Combined", variant_kinds=None),
    PlotScope(scope_id="dist", title_suffix="Distinct", variant_kinds=frozenset({"dist"})),
    PlotScope(scope_id="e2e", title_suffix="E2E", variant_kinds=frozenset({"e2e_auxdis"})),
    PlotScope(
        scope_id="e2e_upfroz",
        title_suffix="Frozen Upsampler",
        variant_kinds=frozenset({"e2e_upfroz"}),
    ),
)


def _parse_broken_y_threshold(raw_value: str) -> float:
    """
    Parse and validate the broken-y threshold CLI value.

    Parameters
    ----------
    raw_value : str
        The raw string value provided for the broken-y threshold.

    Returns
    -------
    float
        The parsed broken-y threshold as a float. Must be greater than 1.0.
    """
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Broken-y threshold must be a floating-point number, got {raw_value!r}."
        ) from exc
    if value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Broken-y threshold must be greater than 1.0, got {value:g}."
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace        Parsed arguments with attributes:
        - base_config: Path to the base YAML config template.
        - device: Device string for infer.py (e.g. cpu, mps, cuda).
        - targets: List of family selectors or exact variant IDs to evaluate.
        - mode: List of optional plot modes (e.g. 'latex-font-only', 'no-title').
        - broken_y_threshold: Minimum adjacent y-ratio that triggers a broken-y companion plot.
        - output_csv: Path to save the consolidated comparison CSV.
        - output_plot: Base path for comparison plots written with metric/x-axis suffixes.
    """
    parser = argparse.ArgumentParser(description="Run multi-variant inference and compare metrics.")
    parser.add_argument(
        "--base-config",
        default="config/inference_config.yaml",
        type=str,
        help="Base inference config YAML used as template.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device passed to infer.py (e.g. cpu, mps, cuda).",
    )
    parser.add_argument(
        "--results",
        "--result",
        dest="results",
        type=str,
        default=None,
        help=(
            "Replay plot generation from an existing benchmark-comparison directory or an "
            f"explicit {BENCHMARK_RESULTS_CSV_NAME} file."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Family selectors and/or exact retained variant IDs to evaluate. "
            "If omitted, uses inference.benchmark_targets from config (or defaults)."
        ),
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=VALID_PLOT_MODES,
        default=None,
        help=(
            "Optional plot modes. Available values: "
            "'latex-font-only', 'no-title', 'no-frame-context', 'save-eps', 'save-svg', "
            "and 'three-panel-circle-markers'. "
            "You may pass any combination."
        ),
    )
    parser.add_argument(
        "--override-style-guide",
        action="store_true",
        help=(
            "When plotting from consolidated CSV data, override stored family labels and colours "
            "with the current style guide from utils/model_variants.py."
        ),
    )
    parser.add_argument(
        "--broken-y-threshold",
        type=_parse_broken_y_threshold,
        default=DEFAULT_BROKEN_Y_THRESHOLD,
        help=("Minimum adjacent y-value ratio that for a broken-y companion plot."),
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=DEFAULT_PNG_DPI,
        help=(
            f"Resolution in DPI for saved PNG files. Defaults to {DEFAULT_PNG_DPI}. "
            "Vector formats (EPS, SVG) are resolution-independent and unaffected."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Path to save consolidated comparison CSV. "
            "Defaults to output/benchmark-comparison-*/."
        ),
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help=(
            "Base path for comparison plots. One subdirectory is generated per metric/x-axis "
            "combination, each containing scope-specific PNGs. "
            "Defaults to output/benchmark-comparison-*/."
        ),
    )
    args = parser.parse_args(argv)
    if args.results and args.targets:
        parser.error("--results cannot be combined with --targets.")
    if args.results and args.output_csv:
        parser.error("--results cannot be combined with --output-csv.")
    return args


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    """
    Resolve a path string relative to repository root when needed.

    Parameters
    ----------
    repo_root : Path
        The root directory of the repository to resolve relative paths against.
    raw_path : str
        The raw path string to resolve. If this is an absolute path, it is returned as-is.
        If it is a relative path, it is resolved against the repo_root.

    Returns
    -------
    Path
    The resolved absolute path.
    """
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def load_base_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate base YAML config.

    Parameters
    ----------
    config_path : Path
        Path to the base YAML config file.

    Returns
    -------
    dict[str, Any]
        The loaded config as a dictionary. Must contain a top-level "inference" key.
    """
    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected dict in YAML config, got {type(config).__name__}")
    if "inference" not in config:
        raise ValueError("Config is missing required top-level key: inference")
    return config


def setup_logging(inference_config: dict[str, Any]) -> str | None:
    """
    Set up logging configuration from a provided inference config dictionary.

    This mirrors `src/infer.py` so benchmark runs share the same logging behaviour.
    """
    log_filename = None
    if "logging" in inference_config:
        logging_config = inference_config["logging"]

        logger = logging.getLogger()
        min_level = logging.DEBUG
        if "handlers" in logging_config:
            handler_levels = [
                getattr(logging, handler.get("level", "DEBUG"))
                for handler in logging_config.get("handlers", [])
            ]
            if handler_levels:
                min_level = min(handler_levels)
        logger.setLevel(min_level)
        logger.handlers.clear()

        formatter = logging.Formatter(logging_config.get("format", "%(levelname)s - %(message)s"))

        for handler_config in logging_config.get("handlers", []):
            if handler_config["type"] == "file":
                filename = handler_config["filename"]
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


def build_model_config(base_config: dict[str, Any], variant: RetainedVariantSpec) -> dict[str, Any]:  # type: ignore[no-any-unimported]
    """
    Build per-variant config by overriding the retained target key.

    Parameters
    ----------
    base_config : dict[str, Any]
        The base configuration dictionary.
    variant : RetainedVariantSpec
        The variant specification for which to build a config.

    Returns
    -------
    dict[str, Any]
        The built config dictionary for the specified variant.
    """
    config = copy.deepcopy(base_config)
    inference = config.setdefault("inference", {})
    if not isinstance(inference, dict):
        raise ValueError("Config key 'inference' must be a dictionary")

    for key in ("model_name", "model_checkpoint", "lam_checkpoint"):
        inference.pop(key, None)

    inference["model_variant"] = variant.variant_id
    inference["collect_metrics"] = True
    inference["benchmark_runtime_only"] = False
    inference.setdefault("latency_warmup_runs", 2)
    inference.setdefault("latency_measurement_runs", 5)
    inference.setdefault("memory_warmup_runs", 1)
    inference.setdefault("memory_measurement_runs", 3)
    inference.setdefault("memory_poll_interval_ms", 1.0)
    inference.setdefault("normalised_memory_enabled", True)
    inference.setdefault("normalised_memory_duration_sec", 10.0)
    inference.setdefault("normalised_memory_pad_short_files", True)
    return config


def build_normalised_model_config(  # type: ignore[no-any-unimported]
    base_config: dict[str, Any],
    variant: RetainedVariantSpec,
    *,
    selected_file_ids: tuple[str, ...],
) -> dict[str, Any]:
    """
    Build a runtime-only config for the normalised memory benchmark.

    Parameters
    ----------
    base_config : dict[str, Any]
        Base configuration dictionary.
    variant : RetainedVariantSpec
        Variant specification for the benchmark run.
    selected_file_ids : tuple[str, ...]
        Exact file identifiers from the raw full-evaluation run.

    Returns
    -------
    dict[str, Any]
        Config dictionary for the normalised runtime-only benchmark pass.

    Raises
    ------
    ValueError
        If the configured normalised duration is not positive.
    """
    config = build_model_config(base_config, variant)
    inference = config["inference"]
    normalised_duration_sec = float(inference.get("normalised_memory_duration_sec", 10.0) or 0.0)
    if normalised_duration_sec <= 0.0:
        raise ValueError(
            "Config key inference.normalised_memory_duration_sec must be positive when "
            "normalised memory benchmarking is enabled."
        )
    inference["benchmark_runtime_only"] = True
    inference["max_audio_length_sec"] = normalised_duration_sec
    inference["selected_files"] = list(selected_file_ids)
    return config


def collect_metrics_files(output_root: Path, run_id: str) -> set[Path]:
    """
    Collect existing metrics JSON paths for a given retained run identifier.

    Parameters
    ----------
    output_root : Path
        The root directory where inference runs output their metrics JSON files.
    run_id : str
        The run identifier for which to collect metrics files.

    Returns
    -------
    set[Path]
        A set of resolved paths to the collected metrics JSON files.
    """
    pattern = f"{run_id}-*/metrics_*.json"
    return {path.resolve() for path in output_root.glob(pattern)}


def select_metrics_path(
    before: set[Path],
    after: set[Path],
    run_id: str,
) -> Path:
    """
    Select the metrics JSON produced by the latest run.

    Parameters
    ----------
    before : set[Path]
        Set of metrics JSON paths that existed before the inference run.
    after : set[Path]
        Set of metrics JSON paths that exist after the inference run.
    run_id : str
        The run identifier for which to select metrics.

    Returns
    -------
    Path
    The path to the metrics JSON file that was produced by the inference run.
    """
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new_files:
        return new_files[-1]

    # Fallback if run overwrote an existing location unexpectedly.
    candidates = sorted(after, key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No metrics file found for run '{run_id}'. Ensure collect_metrics=true in config."
    )


def run_inference_once(  # type: ignore[no-any-unimported]
    repo_root: Path,
    output_root: Path,
    variant: RetainedVariantSpec,
    config_payload: dict[str, Any],
    device: str,
) -> Path:
    """
    Run `infer.py` once for a retained variant and return produced metrics JSON path.

    Parameters
    ----------
    repo_root : Path
        The root directory of the repository where `infer.py` is located.
    output_root : Path
        The root directory where inference runs output their metrics JSON files.
    variant : RetainedVariantSpec
        The variant specification for which to run inference.
    config_payload : dict[str, Any]
        The configuration dictionary to write to a temporary YAML file and pass to `infer.py`.
    device : str
        The device string to pass to `infer.py` (e.g. "cpu", "mps", "cuda").

    Returns
    -------
    Path
        The path to the metrics JSON file produced by the inference run.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{variant.variant_id}_inference.yaml",
        encoding="utf-8",
        delete=False,
    ) as tmp_file:
        yaml.safe_dump(config_payload, tmp_file, sort_keys=False)
        tmp_config_path = Path(tmp_file.name)

    before = collect_metrics_files(output_root, variant.variant_id)
    infer_script = (repo_root / "src" / "infer.py").resolve()
    command = [
        sys.executable,
        str(infer_script),
        "--config",
        str(tmp_config_path),
        "--device",
        device,
    ]
    result = subprocess.run(  # noqa: S603
        command,
        cwd=repo_root,
        capture_output=False,
        text=True,
        check=False,
    )
    tmp_config_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"infer.py failed for variant '{variant.variant_id}' (exit code {result.returncode})."
        )

    after = collect_metrics_files(output_root, variant.variant_id)
    return select_metrics_path(before, after, variant.variant_id)


def aggregate_metrics(metrics_path: Path) -> AggregatedBenchmarkMetrics:  # type: ignore[no-any-unimported]
    """
    Aggregate scientific benchmark metrics from one metrics JSON file.

    Parameters
    ----------
    metrics_path : Path
        Path to the metrics JSON file produced by an inference run.

    Returns
    -------
    AggregatedBenchmarkMetrics
        Aggregated benchmark metrics for the run.
    """
    return aggregate_metrics_json(metrics_path)


def _format_optional_float(value: float) -> str:
    """
    Format a floating-point value for CSV output.

    Parameters
    ----------
    value : float
        Value to format.

    Returns
    -------
    str
        Fixed-width decimal string for finite values, otherwise an empty string.
    """
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


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


def write_comparison_csv(output_csv: Path, results: list[ModelResult]) -> None:
    """
    Write consolidated model comparison metrics to CSV.

    Parameters
    ----------
    output_csv : Path
        The path where the consolidated comparison CSV should be saved.
    results : list[ModelResult]
        A list of ModelResult instances containing the aggregated metrics for each model run.

    Returns
    -------
    None
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_id",
        "family_id",
        "family_label",
        "variant_kind",
        "family_colour",
        "infer_model_name",
        "files_processed",
        "total_frames",
        "frame_width_ms",
        "latency_per_frame_ms",
        "gflops_per_frame",
        "lam_latency_per_frame_ms",
        "lam_gflops_per_frame",
        "memory_peak_max_mb",
        "memory_peak_median_mb",
        "cmd_reference_to_upsampler_median",
        "cmd_upsampler_to_lam_median",
        "cmd_reference_to_lam_median",
        "cmd_reference_to_lam_denoise1_median",
        "cmd_reference_to_lam_denoise2_median",
        "cmd_reference_to_lam_denoise3_median",
        "cmd_reference_to_lam_denoise4_median",
        "normalised_memory_peak_max_mb",
        "normalised_memory_peak_median_mb",
        "normalised_memory_duration_sec",
        "localisation_error_deg",
        "localisation_recall",
        "total_params",
        "run_dir",
        "metrics_json_path",
        "normalised_metrics_json_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "variant_id": result.variant_id,
                    "family_id": result.family_id,
                    "family_label": result.family_label,
                    "variant_kind": result.variant_kind,
                    "family_colour": result.family_colour,
                    "infer_model_name": result.infer_model_name,
                    "files_processed": result.files_processed,
                    "total_frames": result.total_frames,
                    "frame_width_ms": _format_optional_float(result.frame_width_ms),
                    "latency_per_frame_ms": f"{result.latency_per_frame_ms:.6f}",
                    "gflops_per_frame": f"{result.gflops_per_frame:.6f}",
                    "lam_latency_per_frame_ms": f"{result.lam_latency_per_frame_ms:.6f}",
                    "lam_gflops_per_frame": f"{result.lam_gflops_per_frame:.6f}",
                    "memory_peak_max_mb": _format_optional_float(result.memory_peak_max_mb),
                    "memory_peak_median_mb": _format_optional_float(result.memory_peak_median_mb),
                    "cmd_reference_to_upsampler_median": _format_optional_float(
                        result.cmd_reference_to_upsampler_median
                    ),
                    "cmd_upsampler_to_lam_median": _format_optional_float(
                        result.cmd_upsampler_to_lam_median
                    ),
                    "cmd_reference_to_lam_median": _format_optional_float(
                        result.cmd_reference_to_lam_median
                    ),
                    "cmd_reference_to_lam_denoise1_median": _format_optional_float(
                        result.cmd_reference_to_lam_denoise1_median
                    ),
                    "cmd_reference_to_lam_denoise2_median": _format_optional_float(
                        result.cmd_reference_to_lam_denoise2_median
                    ),
                    "cmd_reference_to_lam_denoise3_median": _format_optional_float(
                        result.cmd_reference_to_lam_denoise3_median
                    ),
                    "cmd_reference_to_lam_denoise4_median": _format_optional_float(
                        result.cmd_reference_to_lam_denoise4_median
                    ),
                    "normalised_memory_peak_max_mb": _format_optional_float(
                        result.normalised_memory_peak_max_mb
                    ),
                    "normalised_memory_peak_median_mb": _format_optional_float(
                        result.normalised_memory_peak_median_mb
                    ),
                    "normalised_memory_duration_sec": _format_optional_float(
                        result.normalised_memory_duration_sec
                    ),
                    "localisation_error_deg": _format_optional_float(result.localisation_error_deg),
                    "localisation_recall": _format_optional_float(result.localisation_recall),
                    "total_params": result.total_params,
                    "run_dir": str(result.run_dir),
                    "metrics_json_path": str(result.metrics_json_path),
                    "normalised_metrics_json_path": (
                        str(result.normalised_metrics_json_path)
                        if result.normalised_metrics_json_path is not None
                        else ""
                    ),
                }
            )


def load_comparison_csv(comparison_csv: Path) -> list[dict[str, str]]:
    """
    Load rows from the consolidated comparison CSV.

    Parameters
    ----------
    comparison_csv : Path
        The path to the consolidated comparison CSV file to load.

    Returns
    -------
    list[dict[str, str]]
        A list of dictionaries representing each row in the CSV, where keys are column names.
    """
    with open(comparison_csv, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        return [
            {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}
            for row in reader
        ]


def _build_annotation_points(
    plot_rows: list[dict[str, str]],
    y_values: list[float],
    x_values: list[float],
) -> tuple[list[float], list[float], list[str], list[str]]:
    """
    Select one annotation anchor per family in priority order.

    For each family, select the variant with the highest annotation priority that has valid x and y
    values. This ensures that each family is represented by at most one annotation, and that the
    selected annotation is the most important one available for that family. The priority order is
    defined in the ANNOTATION_PRIORITY dictionary.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        The list of dictionaries representing each row in the CSV, where keys are column names.
    y_values : list[float]
        The list of y values corresponding to each row in plot_rows.
    x_values : list[float]
        The list of x values corresponding to each row in plot_rows.

    Returns
    -------
    tuple[list[float], list[float], list[str], list[str]]
        A tuple containing:
        - annotation_x: List of x coordinates for annotations.
        - annotation_y: List of y coordinates for annotations.
        - annotation_labels: List of labels for annotations.
        - annotation_colours: List of colours for annotations.
    """
    selected_indices: dict[str, int] = {}
    for index, row in enumerate(plot_rows):
        if math.isnan(x_values[index]) or math.isnan(y_values[index]):
            continue
        family_id = row["family_id"]
        current_index = selected_indices.get(family_id)
        if current_index is None:
            selected_indices[family_id] = index
            continue
        current_priority = ANNOTATION_PRIORITY.get(
            plot_rows[current_index]["variant_kind"],
            len(ANNOTATION_PRIORITY),
        )
        candidate_priority = ANNOTATION_PRIORITY.get(
            row["variant_kind"],
            len(ANNOTATION_PRIORITY),
        )
        if candidate_priority < current_priority:
            selected_indices[family_id] = index

    annotation_x: list[float] = []
    annotation_y: list[float] = []
    annotation_labels: list[str] = []
    annotation_colours: list[str] = []
    for index in selected_indices.values():
        annotation_x.append(x_values[index])
        annotation_y.append(y_values[index])
        annotation_labels.append(plot_rows[index]["family_label"])
        annotation_colours.append(plot_rows[index]["family_colour"])
    return annotation_x, annotation_y, annotation_labels, annotation_colours


def _build_rc_params(modes: set[str]) -> dict[str, Any]:
    """
    Build temporary Matplotlib rc parameters for the active plot modes.

    Parameters
    ----------
    modes : set[str]
        Active plot mode identifiers.

    Returns
    -------
    dict[str, Any]
        Rc parameter overrides to use inside ``matplotlib.rc_context``.
    """
    if PLOT_MODE_LATEX_FONT_ONLY not in modes:
        return {
            "font.size": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "figure.titlesize": PLOT_FONT_SIZE,
            "figure.labelsize": PLOT_FONT_SIZE,
        }
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE,
        "figure.titlesize": PLOT_FONT_SIZE,
        "figure.labelsize": PLOT_FONT_SIZE,
    }


def _scatter_points(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    axis: Any,
    x_values: list[float],
    y_values: list[float],
    plot_rows: list[dict[str, str]],
    render_context: PlotRenderContext,
    *,
    three_panel: bool,
) -> None:
    """
    Render all scatter points for one axis.

    Parameters
    ----------
    axis : Any
        Matplotlib axis receiving the scatter points.
    x_values : list[float]
        X-axis values.
    y_values : list[float]
        Y-axis values.
    plot_rows : list[dict[str, str]]
        CSV rows corresponding to the plotted values.
    render_context : PlotRenderContext
        Active plot-style overrides for the current render.
    three_panel : bool
        Whether the marker selection is for a three-panel plot.
    """
    for x_value, y_value, row in zip(x_values, y_values, plot_rows, strict=False):
        axis.scatter(
            [x_value],
            [y_value],
            c=[row["family_colour"]],
            marker=marker_for_row(row, render_context, three_panel=three_panel),
            s=SCATTER_MARKER_SIZE,
            zorder=2,
        )


def _add_lam_baseline(
    axis: Any,
    lam_baseline: float | None,
    lam_baseline_x: float | None,
    lam_baseline_colour: str,
) -> bool:
    """
    Add the LAM baseline guide line when available.

    Parameters
    ----------
    axis : Any
        Matplotlib axis receiving the baseline.
    lam_baseline : float | None
        Baseline y-value, or ``None`` when unavailable.
    lam_baseline_x : float | None
        Baseline x-value for the vertical guide line, or ``None`` when unavailable.
    lam_baseline_colour : str
        Colour used for baseline guide lines.

    Returns
    -------
    bool
        ``True`` when at least one baseline guide was drawn, otherwise ``False``.
    """
    drew_baseline = False
    if lam_baseline is not None and math.isfinite(lam_baseline):
        axis.axhline(
            y=lam_baseline,
            color=lam_baseline_colour,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=0.6,
        )
        drew_baseline = True

    if lam_baseline_x is not None and math.isfinite(lam_baseline_x):
        axis.axvline(
            x=lam_baseline_x,
            color=lam_baseline_colour,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=0.6,
        )
        drew_baseline = True

    return drew_baseline


def _add_axis_legend(  # type: ignore[no-any-unimported]
    axis: Any,
    plot_rows: list[dict[str, str]],
    render_context: PlotRenderContext,
) -> None:
    """
    Attach the single-axis legend when the active mode requires one.

    Parameters
    ----------
    axis : Any
        Matplotlib axis receiving the legend.
    plot_rows : list[dict[str, str]]
        Benchmark row data for the current plot.
    render_context : PlotRenderContext
        Current plot render context.

    Returns
    -------
    None
    """
    handles = build_legend_handles(plot_rows, render_context, three_panel=False)
    if handles:
        axis.legend(handles=handles, loc="upper right")


def _find_adjacent_y_gaps(
    y_values: list[float],
    threshold: float,
) -> list[BrokenAxisGap]:
    """
    Return all adjacent y-value gaps that meet the configured threshold.

    Parameters
    ----------
    y_values : list[float]
        Candidate y-axis values.
    threshold : float
        Minimum adjacent ratio that triggers a broken-axis gap.

    Returns
    -------
    list[BrokenAxisGap]
        Detected adjacent gaps that justify a broken-axis companion plot.
    """
    finite_positive_values = sorted(
        value for value in y_values if math.isfinite(value) and value > 0.0
    )
    if len(finite_positive_values) < MIN_BROKEN_Y_VALUE_COUNT:
        return []

    gaps: list[BrokenAxisGap] = []
    for lower_value, upper_value in zip(
        finite_positive_values,
        finite_positive_values[1:],
        strict=False,
    ):
        ratio = upper_value / lower_value
        if ratio < threshold:
            continue
        gaps.append(
            BrokenAxisGap(
                lower_value=lower_value,
                upper_value=upper_value,
                ratio=ratio,
            )
        )
    return gaps


def _resolve_cluster_padding(value: float, span: float) -> float:
    """
    Resolve baseline padding for one broken-axis segment.

    Parameters
    ----------
    value : float
        Reference value for the segment edge.
    span : float
        Span of values inside the segment.

    Returns
    -------
    float
        Padding to apply around the segment edge.
    """
    return max(
        span * BROKEN_AXIS_OUTER_PADDING_RATIO,
        abs(value) * BROKEN_AXIS_MIN_PADDING_RATIO,
        1e-9,
    )


def _split_values_into_broken_axis_segments(
    finite_values: list[float],
    gaps: list[BrokenAxisGap],
) -> list[list[float]]:
    """
    Split sorted y-values into segments separated by threshold-breaking gaps.

    Parameters
    ----------
    finite_values : list[float]
        Sorted finite y-values.
    gaps : list[BrokenAxisGap]
        Detected gaps that separate the visible segments.

    Returns
    -------
    list[list[float]]
        Y-values grouped into one list per visible segment.
    """
    sorted_gaps = sorted(gaps, key=lambda gap: gap.upper_value)
    segment_values: list[list[float]] = []
    current_segment: list[float] = []
    gap_index = 0
    for value in finite_values:
        while (
            gap_index < len(sorted_gaps)
            and current_segment
            and value >= sorted_gaps[gap_index].upper_value
        ):
            segment_values.append(current_segment)
            current_segment = []
            gap_index += 1
        current_segment.append(value)
    if current_segment:
        segment_values.append(current_segment)
    return segment_values


def _build_broken_axis_segment(
    values: list[float],
    lower_neighbour_max: float | None,
    upper_neighbour_min: float | None,
) -> BrokenAxisSegment:
    """
    Build one broken-axis segment with inner and outer padding.

    Parameters
    ----------
    values : list[float]
        Y-values contained in the segment.
    lower_neighbour_max : float | None
        Upper edge of the previous segment, if present.
    upper_neighbour_min : float | None
        Lower edge of the next segment, if present.

    Returns
    -------
    BrokenAxisSegment
        Resolved visible segment with padded axis limits.
    """
    segment_min = min(values)
    segment_max = max(values)
    magnitude = max(abs(segment_min), abs(segment_max))
    segment_span = max(
        segment_max - segment_min,
        magnitude * BROKEN_AXIS_MIN_PADDING_RATIO,
        1e-9,
    )
    lower_padding = _resolve_cluster_padding(segment_min, segment_span)
    upper_padding = _resolve_cluster_padding(segment_max, segment_span)

    if lower_neighbour_max is not None:
        lower_gap_span = segment_min - lower_neighbour_max
        if lower_gap_span <= 0.0:
            raise ValueError(
                "Broken-axis plot requires positive spacing between adjacent y-value segments."
            )
        lower_padding = min(
            max(
                segment_span * BROKEN_AXIS_INNER_PADDING_RATIO,
                abs(segment_min) * BROKEN_AXIS_MIN_PADDING_RATIO,
                1e-9,
            ),
            lower_gap_span * BROKEN_AXIS_MAX_GAP_PADDING_RATIO,
        )
    if upper_neighbour_min is not None:
        upper_gap_span = upper_neighbour_min - segment_max
        if upper_gap_span <= 0.0:
            raise ValueError(
                "Broken-axis plot requires positive spacing between adjacent y-value segments."
            )
        upper_padding = min(
            max(
                segment_span * BROKEN_AXIS_INNER_PADDING_RATIO,
                abs(segment_max) * BROKEN_AXIS_MIN_PADDING_RATIO,
                1e-9,
            ),
            upper_gap_span * BROKEN_AXIS_MAX_GAP_PADDING_RATIO,
        )

    lower_limit = segment_min - lower_padding
    if segment_min >= 0.0:
        lower_limit = max(0.0, lower_limit)
    return BrokenAxisSegment(
        min_value=segment_min,
        max_value=segment_max,
        lower_limit=lower_limit,
        upper_limit=segment_max + upper_padding,
        point_count=len(values),
    )


def _build_broken_axis_segments(
    y_values: list[float],
    gaps: list[BrokenAxisGap],
) -> list[BrokenAxisSegment]:
    """
    Resolve visible broken-axis segments from threshold-breaking adjacent gaps.

    Parameters
    ----------
    y_values : list[float]
        Y-axis values for the full plot.
    gaps : list[BrokenAxisGap]
        Detected gaps that justify a broken axis.

    Returns
    -------
    list[BrokenAxisSegment]
        Visible segments used by the broken-axis companion plot.
    """
    finite_values = sorted(value for value in y_values if math.isfinite(value))
    if not finite_values:
        raise ValueError("Broken-axis plot requires at least one finite y-value.")
    if not gaps:
        raise ValueError("Broken-axis plot requires at least one threshold-breaking gap.")

    segment_values = _split_values_into_broken_axis_segments(finite_values, gaps)
    if len(segment_values) != len(gaps) + 1:
        raise ValueError("Broken-axis segmentation failed to separate all detected y-value gaps.")

    return [
        _build_broken_axis_segment(
            values=values,
            lower_neighbour_max=segment_values[index - 1][-1] if index > 0 else None,
            upper_neighbour_min=segment_values[index + 1][0]
            if index < len(segment_values) - 1
            else None,
        )
        for index, values in enumerate(segment_values)
    ]


def _build_broken_axis_height_ratios(segments: list[BrokenAxisSegment]) -> list[float]:
    """
    Build per-segment height ratios for a broken-axis figure.

    Parameters
    ----------
    segments : list[BrokenAxisSegment]
        Visible segments in the broken-axis figure.

    Returns
    -------
    list[float]
        Height ratios for the Matplotlib gridspec configuration.
    """
    return [
        min(
            BROKEN_AXIS_MAX_HEIGHT_RATIO,
            max(
                BROKEN_AXIS_MIN_HEIGHT_RATIO,
                BROKEN_AXIS_MIN_HEIGHT_RATIO + segment.point_count * BROKEN_AXIS_HEIGHT_PER_POINT,
            ),
        )
        for segment in segments
    ]


def _split_annotation_points_by_segments(
    annotation_x: list[float],
    annotation_y: list[float],
    annotation_labels: list[str],
    annotation_colours: list[str],
    segments: list[BrokenAxisSegment],
) -> list[tuple[list[float], list[float], list[str], list[str]]]:
    """
    Split annotation points across the visible broken-axis segments.

    Parameters
    ----------
    annotation_x : list[float]
        Annotation x coordinates.
    annotation_y : list[float]
        Annotation y coordinates.
    annotation_labels : list[str]
        Annotation labels.
    annotation_colours : list[str]
        Annotation colours.
    segments : list[BrokenAxisSegment]
        Visible broken-axis segments.

    Returns
    -------
    list[tuple[list[float], list[float], list[str], list[str]]]
        Annotation groups aligned with the visible segments.
    """
    grouped_annotations: list[tuple[list[float], list[float], list[str], list[str]]] = [
        ([], [], [], []) for _ in segments
    ]
    for x_value, y_value, label, colour in zip(
        annotation_x,
        annotation_y,
        annotation_labels,
        annotation_colours,
        strict=False,
    ):
        assigned = False
        for index, segment in enumerate(segments):
            if segment.lower_limit <= y_value <= segment.upper_limit:
                grouped_annotations[index][0].append(x_value)
                grouped_annotations[index][1].append(y_value)
                grouped_annotations[index][2].append(label)
                grouped_annotations[index][3].append(colour)
                assigned = True
                break
        if assigned:
            continue
        closest_index = min(
            range(len(segments)),
            key=lambda index: min(
                abs(y_value - segments[index].lower_limit),
                abs(y_value - segments[index].upper_limit),
            ),
        )
        grouped_annotations[closest_index][0].append(x_value)
        grouped_annotations[closest_index][1].append(y_value)
        grouped_annotations[closest_index][2].append(label)
        grouped_annotations[closest_index][3].append(colour)
    return grouped_annotations


def _draw_broken_axis_marks(upper_axis: Any, lower_axis: Any) -> None:
    """
    Draw diagonal marks indicating a broken y-axis.

    Parameters
    ----------
    upper_axis : Any
        Axis above the break.
    lower_axis : Any
        Axis below the break.
    """
    kwargs = {
        "marker": [
            (-1, -BROKEN_AXIS_BREAK_MARKER_HALF_HEIGHT),
            (1, BROKEN_AXIS_BREAK_MARKER_HALF_HEIGHT),
        ],
        "markersize": BROKEN_AXIS_BREAK_MARKER_SIZE,
        "linestyle": "none",
        "color": "black",
        "mec": "black",
        "mew": 1.0,
        "clip_on": False,
    }
    upper_axis.plot([0, 1], [0, 0], transform=upper_axis.transAxes, **kwargs)
    lower_axis.plot([0, 1], [1, 1], transform=lower_axis.transAxes, **kwargs)


def _broken_y_output_path(output_path: Path) -> Path:
    """
    Build the companion output path for a broken-y plot.

    Parameters
    ----------
    output_path : Path
        Output path of the standard scatter plot.

    Returns
    -------
    Path
        Output path for the broken-y companion plot.
    """
    return output_path.with_name(f"{output_path.stem}_broken_y{output_path.suffix}")


def _rows_for_plot_scope(
    plot_rows: list[dict[str, str]],
    scope: PlotScope,
) -> list[dict[str, str]]:
    """
    Filter plotted rows to one comparison scope.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Rows already filtered to plotted variant kinds.
    scope : PlotScope
        Plot scope describing which variant kinds to include.

    Returns
    -------
    list[dict[str, str]]
        Rows that belong to the requested scope.
    """
    return filter_rows_by_variant_kinds(  # type: ignore[no-any-return]
        plot_rows,
        scope.variant_kinds,
        include_lam_variant=True,
    )


def _x_axis_specs_for_rows(plot_rows: list[dict[str, str]]) -> list[XAxisSpec]:
    """
    Build x-axis specifications for a given row subset.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Rows included in the current plot scope.

    Returns
    -------
    list[XAxisSpec]
        X-axis specifications for localisation error and localisation recall.
    """
    return [
        XAxisSpec(
            x_values=[
                _parse_optional_float(
                    row.get("localisation_error_deg", row.get("localization_error_deg", ""))
                )
                for row in plot_rows
            ],
            label=LOCALISATION_ERROR_X_LABEL,
            title_suffix="Localisation Error",
            output_suffix="localisation_error",
        ),
        XAxisSpec(
            x_values=[
                _parse_optional_float(
                    row.get("localisation_recall", row.get("localization_recall", ""))
                )
                for row in plot_rows
            ],
            label=LOCALISATION_RECALL_X_LABEL,
            title_suffix="Localisation Recall",
            output_suffix="localisation_recall",
        ),
    ]


def _x_axis_spec_for_suffix(
    plot_rows: list[dict[str, str]],
    output_suffix: str,
) -> XAxisSpec:
    """
    Resolve one x-axis spec by its stable output suffix.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Candidate rows used to construct x-axis values.
    output_suffix : str
        Stable x-axis output suffix.

    Returns
    -------
    XAxisSpec
        Matching x-axis specification.
    """
    for spec in _x_axis_specs_for_rows(plot_rows):
        if spec.output_suffix == output_suffix:
            return spec
    raise ValueError(f"Unknown x-axis output suffix: {output_suffix}")


def _resolve_lam_x_baseline(
    lam_row: dict[str, str] | None,
    x_axis_output_suffix: str,
) -> float | None:
    """
    Resolve the LAM x-value baseline for the requested x-axis metric.

    Parameters
    ----------
    lam_row : dict[str, str] | None
        Optional LAM row from the consolidated CSV.
    x_axis_output_suffix : str
        Stable x-axis suffix (for example ``localisation_error``).

    Returns
    -------
    float | None
        Finite LAM x-value for the requested axis, or ``None`` when unavailable.
    """
    if lam_row is None:
        return None

    if x_axis_output_suffix == "localisation_error":
        value = _parse_optional_float(
            lam_row.get("localisation_error_deg", lam_row.get("localization_error_deg", ""))
        )
    elif x_axis_output_suffix == "localisation_recall":
        value = _parse_optional_float(
            lam_row.get("localisation_recall", lam_row.get("localization_recall", ""))
        )
    else:
        return None

    if not math.isfinite(value):
        return None
    return value


def _build_memory_plot_templates(
    plot_rows: list[dict[str, str]],
    normalised_duration_sec: float,
    lam_row: dict[str, str] | None,
) -> list[MemoryPlotTemplate]:
    """
    Build memory metric templates shared by all scopes and panel subsets.

    Parameters
    ----------
    plot_rows : list[dict[str, str]]
        Rows included in plotted variant kinds.
    normalised_duration_sec : float
        Duration label used for normalised memory metrics.
    lam_row : dict[str, str] | None
        Optional LAM row for baseline extraction.

    Returns
    -------
    list[MemoryPlotTemplate]
        Resolved memory metric templates with output units and optional LAM baseline values.
    """
    memory_plot_definitions = [
        (
            "memory_peak_max_mb",
            "raw_memory_peak_max",
            "Worst-case End-to-End Peak Memory Delta",
            "Worst-case End-to-End Peak Memory Delta",
        ),
        (
            "memory_peak_median_mb",
            "raw_memory_peak_median",
            "Median End-to-End Peak Memory Delta",
            "Median End-to-End Peak Memory Delta",
        ),
        (
            "normalised_memory_peak_max_mb",
            "normalised_memory_peak_max",
            f"Worst-case Normalised {normalised_duration_sec:.1f} s Peak Memory Delta",
            "Worst-case End-to-End Peak Memory Delta",
        ),
        (
            "normalised_memory_peak_median_mb",
            "normalised_memory_peak_median",
            f"Median Normalised {normalised_duration_sec:.1f} s Peak Memory Delta",
            "Median End-to-End Peak Memory Delta",
        ),
    ]

    templates: list[MemoryPlotTemplate] = []
    for column_name, output_stem, title_prefix, y_label_prefix in memory_plot_definitions:
        all_raw_values = [_parse_optional_float(row.get(column_name)) for row in plot_rows]
        if not any(math.isfinite(value) for value in all_raw_values):
            continue

        memory_scale, memory_unit = select_memory_display_unit(all_raw_values)
        lam_baseline = None
        if lam_row is not None:
            lam_value = _parse_optional_float(lam_row.get(column_name))
            lam_baseline = lam_value * memory_scale if math.isfinite(lam_value) else float("nan")

        templates.append(
            MemoryPlotTemplate(
                column_name=column_name,
                output_stem=output_stem,
                title=title_prefix,
                y_label_prefix=y_label_prefix,
                scale=memory_scale,
                unit=memory_unit,
                lam_baseline=lam_baseline,
            )
        )
    return templates


def _build_scope_plot_specs(
    scope_rows: list[dict[str, str]],
    frame_context: str,
    lam_row: dict[str, str] | None,
    memory_templates: list[MemoryPlotTemplate],
    modes: set[str],
) -> list[ScatterSpec]:
    """
    Build all scatter specs for one row subset.

    Parameters
    ----------
    scope_rows : list[dict[str, str]]
        Rows represented by the current scope or panel subset.
    frame_context : str
        Frame-width context used in latency and GFLOPs labels.
    lam_row : dict[str, str] | None
        Optional LAM row for baseline extraction.
    memory_templates : list[MemoryPlotTemplate]
        Shared memory metric templates.

    Returns
    -------
    list[ScatterSpec]
        Scatter specs for latency, GFLOPs, and all available memory metrics.
    """
    lam_baseline_colour = "black"
    if lam_row is not None:
        lam_baseline_colour_candidate = str(lam_row.get("family_colour", "")).strip()
        if lam_baseline_colour_candidate:
            lam_baseline_colour = lam_baseline_colour_candidate

    latency_label = "Latency per Frame (ms)"
    gflops_label = "GFLOPs per Frame"
    if PLOT_MODE_NO_FRAME_CONTEXT not in modes:
        latency_label = f"Latency per Frame (ms, {frame_context})"
        gflops_label = f"GFLOPs per Frame ({frame_context})"

    scope_plot_specs: list[ScatterSpec] = [
        ScatterSpec(
            y_values=[_parse_optional_float(row.get("latency_per_frame_ms")) for row in scope_rows],
            y_label=latency_label,
            title="End-to-End Latency",
            output_path=Path("latency"),
            lam_baseline=(
                _parse_optional_float(lam_row.get("latency_per_frame_ms")) if lam_row else None
            ),
            lam_baseline_colour=lam_baseline_colour,
        ),
        ScatterSpec(
            y_values=[_parse_optional_float(row.get("gflops_per_frame")) for row in scope_rows],
            y_label=gflops_label,
            title="End-to-End GFLOPs",
            output_path=Path("gflops"),
            lam_baseline=(
                _parse_optional_float(lam_row.get("gflops_per_frame")) if lam_row else None
            ),
            lam_baseline_colour=lam_baseline_colour,
        ),
        ScatterSpec(
            y_values=[_parse_optional_float(row.get("total_params")) / 1e6 for row in scope_rows],
            y_label="Parameters (M)",
            title="Total Parameters",
            output_path=Path("total_params"),
            lam_baseline=(
                _parse_optional_float(lam_row.get("total_params")) / 1e6
                if lam_row and math.isfinite(_parse_optional_float(lam_row.get("total_params")))
                else None
            ),
            lam_baseline_colour=lam_baseline_colour,
        ),
    ]

    for template in memory_templates:
        scope_raw_values = [
            _parse_optional_float(row.get(template.column_name)) for row in scope_rows
        ]
        scaled_values = [
            value * template.scale if math.isfinite(value) else float("nan")
            for value in scope_raw_values
        ]
        scope_plot_specs.append(
            ScatterSpec(
                y_values=scaled_values,
                y_label=f"{template.y_label_prefix} ({template.unit})",
                title=template.title,
                output_path=Path(template.output_stem),
                lam_baseline=template.lam_baseline,
                lam_baseline_colour=lam_baseline_colour,
            )
        )

    return scope_plot_specs


def _scatter_specs_by_output_stem(specs: list[ScatterSpec]) -> dict[str, ScatterSpec]:
    """
    Build a lookup map from output stem to scatter spec.

    Parameters
    ----------
    specs : list[ScatterSpec]
        Scatter specs to index.

    Returns
    -------
    dict[str, ScatterSpec]
        Mapping keyed by ``spec.output_path.name``.
    """
    return {spec.output_path.name: spec for spec in specs}


def _shared_x_limits_for_panels(
    panels: list[ThreePanelSubset],
) -> tuple[float, float] | None:
    """
    Resolve one shared x-axis span that covers all finite panel x-values.

    Parameters
    ----------
    panels : list[ThreePanelSubset]
        Panel data used in the three-panel composite.

    Returns
    -------
    tuple[float, float] | None
        Shared ``(x_min, x_max)`` limits including small padding, or ``None`` when all panel
        x-values are non-finite.
    """
    finite_x_values = [
        x_value for panel in panels for x_value in panel.x_values if math.isfinite(x_value)
    ]
    if not finite_x_values:
        return None

    x_min = min(finite_x_values)
    x_max = max(finite_x_values)
    span = x_max - x_min
    if span <= 0.0:
        padding = max(abs(x_min) * SHARED_X_AXIS_PADDING_RATIO, 1e-6)
    else:
        padding = max(span * SHARED_X_AXIS_PADDING_RATIO, 1e-6)
    return x_min - padding, x_max + padding


def _save_vector_formats(fig: plt.Figure, png_path: Path, modes: set[str]) -> None:
    """Save EPS and/or SVG companions next to *png_path* when the respective mode is active."""
    if PLOT_MODE_SAVE_EPS in modes:
        _ps_logger = logging.getLogger("matplotlib.backends.backend_ps")
        _prev_level = _ps_logger.level
        _ps_logger.setLevel(logging.ERROR)
        try:
            fig.savefig(png_path.with_suffix(".eps"))
        finally:
            _ps_logger.setLevel(_prev_level)
    if PLOT_MODE_SAVE_SVG in modes:
        fig.savefig(png_path.with_suffix(".svg"))


def _plot_scatter(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    x_values: list[float],
    plot_rows: list[dict[str, str]],
    spec: ScatterSpec,
    x_label: str,
    render_context: PlotRenderContext,
    modes: set[str],
    png_dpi: int = DEFAULT_PNG_DPI,
) -> None:
    """
    Create one scatter plot and save it as PNG.

    Parameters
    ----------
    x_values : list[float]
        X-axis values.
    plot_rows : list[dict[str, str]]
        CSV rows corresponding to the plotted variants.
    spec : ScatterSpec
        Y-axis and output configuration.
    x_label : str
        X-axis label.
    modes : set[str]
        Active plot presentation modes.
    """
    hide_title = PLOT_MODE_NO_TITLE in modes
    with plt.rc_context(rc=_build_rc_params(modes)):
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN),
            constrained_layout=True,
        )
        _scatter_points(
            axis,
            x_values,
            spec.y_values,
            plot_rows,
            render_context,
            three_panel=False,
        )

        if not hide_title:
            axis.set_title(spec.title)
        axis.set_xlabel(x_label)
        fig.supylabel(spec.y_label)
        axis.grid(True, linestyle="--", alpha=0.35)

        _add_lam_baseline(
            axis,
            spec.lam_baseline,
            spec.lam_baseline_x,
            spec.lam_baseline_colour,
        )

        if should_annotate_points(render_context, three_panel=False):
            (
                annotation_x,
                annotation_y,
                annotation_labels,
                annotation_colours,
            ) = _build_annotation_points(plot_rows, spec.y_values, x_values)
            annotate_points(
                axis,
                annotation_x,
                annotation_y,
                annotation_labels,
                annotation_colours,
                obstacle_x_values=x_values,
                obstacle_y_values=spec.y_values,
            )

        _add_axis_legend(axis, plot_rows, render_context)

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(spec.output_path, dpi=png_dpi)
        _save_vector_formats(fig, spec.output_path, modes)
        plt.close(fig)


def _plot_scatter_broken_y(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    x_values: list[float],
    plot_rows: list[dict[str, str]],
    spec: ScatterSpec,
    x_label: str,
    render_context: PlotRenderContext,
    modes: set[str],
    gaps: list[BrokenAxisGap],
    png_dpi: int = DEFAULT_PNG_DPI,
) -> Path:
    """
    Create one broken-y scatterplot companion and save it as PNG.

    Parameters
    ----------
    x_values : list[float]
        X-axis values for the scatter plot.
    plot_rows : list[dict[str, str]]
        CSV rows corresponding to the plotted variants.
    spec : ScatterSpec
        Y-axis and output configuration for the plot.
    x_label : str
        Label for the shared x-axis.
    modes : set[str]
        Active plot modes.
    gaps : list[BrokenAxisGap]
        Detected adjacent y-value gaps that justify the broken axis.

    Returns
    -------
    Path
        Output path of the generated broken-y companion plot.
    """
    hide_title = PLOT_MODE_NO_TITLE in modes
    output_path = _broken_y_output_path(spec.output_path)
    segments = list(reversed(_build_broken_axis_segments(spec.y_values, gaps)))
    height_ratios = _build_broken_axis_height_ratios(segments)
    with plt.rc_context(rc=_build_rc_params(modes)):
        fig, axes = plt.subplots(
            len(segments),
            1,
            sharex=True,
            figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN),
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes_list = list(axes if isinstance(axes, (list, tuple)) else axes.flat)
        for axis, segment in zip(axes_list, segments, strict=False):
            _scatter_points(
                axis,
                x_values,
                spec.y_values,
                plot_rows,
                render_context,
                three_panel=False,
            )
            axis.grid(True, linestyle="--", alpha=0.35)
            axis.set_ylim(segment.lower_limit, segment.upper_limit)

        for index, axis in enumerate(axes_list):
            if index > 0:
                axis.spines["top"].set_visible(False)
                axis.tick_params(axis="x", which="both", top=False)
            if index < len(axes_list) - 1:
                axis.spines["bottom"].set_visible(False)
                axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        axes_list[-1].set_xlabel(x_label)
        fig.supylabel(spec.y_label)
        if not hide_title:
            axes_list[0].set_title(spec.title)

        for axis in axes_list:
            _add_lam_baseline(
                axis,
                spec.lam_baseline,
                spec.lam_baseline_x,
                spec.lam_baseline_colour,
            )

        if should_annotate_points(render_context, three_panel=False):
            (
                annotation_x,
                annotation_y,
                annotation_labels,
                annotation_colours,
            ) = _build_annotation_points(plot_rows, spec.y_values, x_values)
            annotation_groups = _split_annotation_points_by_segments(
                annotation_x,
                annotation_y,
                annotation_labels,
                annotation_colours,
                segments,
            )
            for axis, annotation_group in zip(axes_list, annotation_groups, strict=False):
                annotate_points(
                    axis,
                    *annotation_group,
                    obstacle_x_values=x_values,
                    obstacle_y_values=spec.y_values,
                )
        for upper_axis, lower_axis in zip(axes_list, axes_list[1:], strict=False):
            _draw_broken_axis_marks(upper_axis, lower_axis)

        _add_axis_legend(axes_list[0], plot_rows, render_context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=png_dpi)
        _save_vector_formats(fig, output_path, modes)
        plt.close(fig)
    return output_path


def _plot_three_panel_scatter(  # type: ignore[no-any-unimported]  # noqa: PLR0913
    panels: list[ThreePanelSubset],
    spec: ScatterSpec,
    x_label: str,
    render_context: PlotRenderContext,
    modes: set[str],
    png_dpi: int = DEFAULT_PNG_DPI,
) -> None:
    """
    Create one shared three-panel scatter plot and save it as PNG.

    Parameters
    ----------
    panels : list[ThreePanelSubset]
        Ordered panel subsets to render left-to-right.
    spec : ScatterSpec
        Shared y-axis and output configuration.
    x_label : str
        X-axis label for each panel.
    modes : set[str]
        Active plot presentation modes.
    """
    if not panels:
        raise ValueError("Three-panel plot requires at least one panel.")

    hide_title = PLOT_MODE_NO_TITLE in modes
    shared_x_limits = _shared_x_limits_for_panels(panels)
    with plt.rc_context(rc=_build_rc_params(modes)):
        fig, axes = plt.subplots(
            1,
            len(panels),
            sharey=True,
            sharex=True,
            figsize=(THREE_PANEL_PLOT_WIDTH_IN, PLOT_HEIGHT_IN),
            constrained_layout=True,
        )
        axes_list = list(axes if isinstance(axes, (list, tuple)) else axes.flat)

        for index, (axis, panel) in enumerate(zip(axes_list, panels, strict=False)):
            _scatter_points(
                axis,
                panel.x_values,
                panel.y_values,
                panel.rows,
                render_context,
                three_panel=True,
            )
            axis.grid(True, linestyle="--", alpha=0.35)
            axis.set_xlabel(x_label)
            axis.set_title(panel.title)
            if shared_x_limits is not None:
                axis.set_xlim(*shared_x_limits)

            axis.set_ylabel("")
            if index > 0:
                axis.tick_params(axis="y", which="both", left=False, labelleft=False)

            _add_lam_baseline(
                axis,
                spec.lam_baseline,
                spec.lam_baseline_x,
                spec.lam_baseline_colour,
            )

            if should_annotate_points(render_context, three_panel=True):
                (
                    annotation_x,
                    annotation_y,
                    annotation_labels,
                    annotation_colours,
                ) = _build_annotation_points(panel.rows, panel.y_values, panel.x_values)
                annotate_points(
                    axis,
                    annotation_x,
                    annotation_y,
                    annotation_labels,
                    annotation_colours,
                    obstacle_x_values=panel.x_values,
                    obstacle_y_values=panel.y_values,
                )

        if not hide_title:
            fig.suptitle(spec.title)
        fig.supylabel(spec.y_label)

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(spec.output_path, dpi=png_dpi)
        _save_vector_formats(fig, spec.output_path, modes)
        plt.close(fig)


def _plot_three_panel_scatter_broken_y(  # type: ignore[no-any-unimported]  # noqa: C901, PLR0912, PLR0913
    panels: list[ThreePanelSubset],
    spec: ScatterSpec,
    x_label: str,
    render_context: PlotRenderContext,
    modes: set[str],
    gaps: list[BrokenAxisGap],
    png_dpi: int = DEFAULT_PNG_DPI,
) -> Path:
    """
    Create one broken-y three-panel companion plot and save it as PNG.

    Parameters
    ----------
    panels : list[ThreePanelSubset]
        Ordered panel subsets to render left-to-right.
    spec : ScatterSpec
        Shared y-axis and output configuration.
    x_label : str
        X-axis label for each panel.
    modes : set[str]
        Active plot presentation modes.
    gaps : list[BrokenAxisGap]
        Detected adjacent y-value gaps for the union of all panel y-values.

    Returns
    -------
    Path
        Output path of the generated broken-y companion plot.
    """
    if not panels:
        raise ValueError("Three-panel broken-y plot requires at least one panel.")

    hide_title = PLOT_MODE_NO_TITLE in modes
    output_path = _broken_y_output_path(spec.output_path)
    all_y_values = [value for panel in panels for value in panel.y_values]
    shared_x_limits = _shared_x_limits_for_panels(panels)
    segments = list(reversed(_build_broken_axis_segments(all_y_values, gaps)))
    height_ratios = _build_broken_axis_height_ratios(segments)

    annotation_groups_by_panel = []
    if should_annotate_points(render_context, three_panel=True):
        for panel in panels:
            (
                annotation_x,
                annotation_y,
                annotation_labels,
                annotation_colours,
            ) = _build_annotation_points(panel.rows, panel.y_values, panel.x_values)
            annotation_groups_by_panel.append(
                _split_annotation_points_by_segments(
                    annotation_x,
                    annotation_y,
                    annotation_labels,
                    annotation_colours,
                    segments,
                )
            )

    with plt.rc_context(rc=_build_rc_params(modes)):
        fig, axes = plt.subplots(
            len(segments),
            len(panels),
            sharex=True,
            sharey="row",
            figsize=(THREE_PANEL_PLOT_WIDTH_IN, PLOT_HEIGHT_IN),
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes_rows = [list(row) for row in axes]

        for row_index, segment in enumerate(segments):
            for col_index, panel in enumerate(panels):
                axis = axes_rows[row_index][col_index]
                _scatter_points(
                    axis,
                    panel.x_values,
                    panel.y_values,
                    panel.rows,
                    render_context,
                    three_panel=True,
                )
                axis.grid(True, linestyle="--", alpha=0.35)
                axis.set_ylim(segment.lower_limit, segment.upper_limit)
                if shared_x_limits is not None:
                    axis.set_xlim(*shared_x_limits)

                if row_index == 0:
                    axis.set_title(panel.title)
                if row_index > 0:
                    axis.spines["top"].set_visible(False)
                    axis.tick_params(axis="x", which="both", top=False)
                if row_index < len(segments) - 1:
                    axis.spines["bottom"].set_visible(False)
                    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                else:
                    axis.set_xlabel(x_label)

                if col_index == 0:
                    axis.tick_params(axis="y", which="both", left=True, labelleft=True)
                else:
                    axis.set_ylabel("")
                    axis.tick_params(axis="y", which="both", left=False, labelleft=False)

                _add_lam_baseline(
                    axis,
                    spec.lam_baseline,
                    spec.lam_baseline_x,
                    spec.lam_baseline_colour,
                )

                if annotation_groups_by_panel:
                    annotate_points(
                        axis,
                        *annotation_groups_by_panel[col_index][row_index],
                        obstacle_x_values=panel.x_values,
                        obstacle_y_values=panel.y_values,
                    )

        for row_index in range(len(axes_rows) - 1):
            for col_index in range(len(panels)):
                _draw_broken_axis_marks(
                    axes_rows[row_index][col_index],
                    axes_rows[row_index + 1][col_index],
                )

        fig.supylabel(spec.y_label)

        if not hide_title:
            fig.suptitle(spec.title)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=png_dpi)
        _save_vector_formats(fig, output_path, modes)
        plt.close(fig)
    return output_path


def plot_from_csv(  # noqa: C901, PLR0912, PLR0913, PLR0915
    comparison_csv: Path,
    output_plot_prefix: Path,
    frame_width_ms: float,
    modes: set[str],
    override_style_guide: bool = False,
    broken_y_threshold: float = DEFAULT_BROKEN_Y_THRESHOLD,
    png_dpi: int = DEFAULT_PNG_DPI,
) -> list[Path]:
    """
    Generate scientific benchmark scatter plots from the consolidated CSV.

    Parameters
    ----------
    comparison_csv : Path
        Path to the consolidated benchmark CSV.
    output_plot_prefix : Path
        Prefix used for generated plot file names.
    frame_width_ms : float
        Fallback frame width in milliseconds for latency and GFLOPs labels when the CSV does not
        yet contain persisted ``frame_width_ms`` metadata.
    modes : set[str]
        Active plot presentation modes.
    override_style_guide : bool, optional
        Whether to override stored family labels and colours with the current style guide from
        ``utils.model_variants``.
    broken_y_threshold : float, optional
        Adjacent y-ratio threshold for generating broken-axis companion plots.

    Returns
    -------
    list[Path]
        Paths to all generated plot files.
    """
    rows = load_comparison_csv(comparison_csv)
    if not rows:
        raise ValueError(f"No rows found in CSV: {comparison_csv}")
    if override_style_guide:
        rows = apply_current_style_guide(rows)
    render_context = build_plot_render_context(
        force_three_panel_circle_markers=PLOT_MODE_THREE_PANEL_CIRCLE_MARKERS in modes,
    )
    frame_width_ms = resolve_frame_width_ms(
        rows,
        fallback_frame_width_ms=frame_width_ms,
    )

    filtered_out = [
        row for row in rows if str(row.get("variant_kind", "")).strip() not in PLOTTED_VARIANT_KINDS
    ]
    if filtered_out:
        logging.info(
            "Filtering %d non-plotted comparison row(s) from the scatter plots: %s",
            len(filtered_out),
            ", ".join(sorted({str(row.get("variant_kind", "")).strip() for row in filtered_out})),
        )
    plot_rows = [
        row for row in rows if str(row.get("variant_kind", "")).strip() in PLOTTED_VARIANT_KINDS
    ]
    if not plot_rows:
        logging.warning(
            "Skipping plot generation because the consolidated CSV contains only non-plotted "
            "variant kinds."
        )
        return []

    frame_context = f"frame = {frame_width_ms:g} ms"
    lam_row = next((row for row in rows if str(row.get("variant_id", "")).strip() == "lam"), None)

    normalised_duration_values = [
        _parse_optional_float(row.get("normalised_memory_duration_sec")) for row in rows
    ]
    normalised_duration_sec = next(
        (value for value in normalised_duration_values if math.isfinite(value)),
        10.0,
    )
    memory_templates = _build_memory_plot_templates(
        plot_rows,
        normalised_duration_sec,
        lam_row,
    )

    output_paths: list[Path] = []
    cmd_stage_plot_directory = output_plot_prefix.parent / (
        f"{output_plot_prefix.name}_cmd_stage_trajectories"
    )
    cmd_stage_plot_path = plot_cmd_stage_trajectories(
        plot_rows,
        cmd_stage_plot_directory / CMD_STAGE_PLOT_FILENAME,
        render_context,
        CMDStagePlotStyle(
            rc_params=_build_rc_params(modes),
            hide_title=PLOT_MODE_NO_TITLE in modes,
            save_vector_formats=_save_vector_formats,
            modes=modes,
            png_dpi=png_dpi,
        ),
    )
    if cmd_stage_plot_path is not None:
        output_paths.append(cmd_stage_plot_path)
    cmd_stage_three_panel_path = plot_cmd_stage_trajectories_three_panel(
        build_three_panel_rows(plot_rows),
        cmd_stage_plot_directory / THREE_PANEL_FILENAME,
        render_context,
        CMDStagePlotStyle(
            rc_params=_build_rc_params(modes),
            hide_title=PLOT_MODE_NO_TITLE in modes,
            save_vector_formats=_save_vector_formats,
            modes=modes,
            png_dpi=png_dpi,
        ),
    )
    if cmd_stage_three_panel_path is not None:
        output_paths.append(cmd_stage_three_panel_path)

    for scope in PLOT_SCOPES:
        scope_rows = _rows_for_plot_scope(plot_rows, scope)
        if not scope_rows:
            logging.info("Skipping empty plot scope '%s'.", scope.scope_id)
            continue
        scope_plot_specs = _build_scope_plot_specs(
            scope_rows,
            frame_context,
            lam_row,
            memory_templates,
            modes,
        )

        for x_axis_spec in _x_axis_specs_for_rows(scope_rows):
            lam_baseline_x = _resolve_lam_x_baseline(
                lam_row,
                x_axis_spec.output_suffix,
            )
            if not any(math.isfinite(value) for value in x_axis_spec.x_values):
                logging.info(
                    "Skipping %s plots for scope '%s' because the consolidated CSV contains no "
                    "finite x-values.",
                    x_axis_spec.output_suffix,
                    scope.scope_id,
                )
                continue
            for base_spec in scope_plot_specs:
                plot_directory = output_plot_prefix.parent / (
                    f"{output_plot_prefix.name}_{base_spec.output_path.name}_vs_"
                    f"{x_axis_spec.output_suffix}"
                )
                output_path = plot_directory / f"{scope.scope_id}.png"
                spec = ScatterSpec(
                    y_values=base_spec.y_values,
                    y_label=base_spec.y_label,
                    title=(
                        f"{base_spec.title} vs {x_axis_spec.title_suffix} "
                        f"({scope.title_suffix})"
                    ),
                    output_path=output_path,
                    lam_baseline=base_spec.lam_baseline,
                    lam_baseline_x=lam_baseline_x,
                    lam_baseline_colour=base_spec.lam_baseline_colour,
                    lam_baseline_label=base_spec.lam_baseline_label,
                )
                _plot_scatter(
                    x_values=x_axis_spec.x_values,
                    plot_rows=scope_rows,
                    spec=spec,
                    x_label=x_axis_spec.label,
                    render_context=render_context,
                    modes=modes,
                    png_dpi=png_dpi,
                )
                output_paths.append(output_path)
                effective_broken_y_threshold = (
                    DEFAULT_MEMORY_BROKEN_Y_THRESHOLD
                    if "memory_peak" in base_spec.output_path.name
                    else broken_y_threshold
                )
                gaps = _find_adjacent_y_gaps(spec.y_values, effective_broken_y_threshold)
                if not gaps:
                    continue
                broken_output_path = _plot_scatter_broken_y(
                    x_values=x_axis_spec.x_values,
                    plot_rows=scope_rows,
                    spec=spec,
                    x_label=x_axis_spec.label,
                    render_context=render_context,
                    modes=modes,
                    gaps=gaps,
                    png_dpi=png_dpi,
                )
                logging.debug(
                    "Generated broken-y companion plot for %s/%s "
                    "(%d adjacent gap(s) >= %.3f, largest ratio %.3f).",
                    plot_directory.name,
                    output_path.name,
                    len(gaps),
                    effective_broken_y_threshold,
                    max(gap.ratio for gap in gaps),
                )
                output_paths.append(broken_output_path)

    three_panel_scope_specs: list[
        tuple[str, str, list[dict[str, str]], dict[str, ScatterSpec]]
    ] = []
    for variant_kind, panel_title, panel_rows in build_three_panel_rows(plot_rows):
        panel_specs = _build_scope_plot_specs(
            panel_rows,
            frame_context,
            lam_row,
            memory_templates,
            modes,
        )
        three_panel_scope_specs.append(
            (
                variant_kind,
                panel_title,
                panel_rows,
                _scatter_specs_by_output_stem(panel_specs),
            )
        )

    combined_plot_specs = _build_scope_plot_specs(
        plot_rows,
        frame_context,
        lam_row,
        memory_templates,
        modes,
    )
    combined_specs_by_output_stem = _scatter_specs_by_output_stem(combined_plot_specs)

    for x_axis_spec in _x_axis_specs_for_rows(plot_rows):
        lam_baseline_x = _resolve_lam_x_baseline(
            lam_row,
            x_axis_spec.output_suffix,
        )
        panel_x_specs = [
            _x_axis_spec_for_suffix(panel_rows, x_axis_spec.output_suffix)
            for _, _, panel_rows, _ in three_panel_scope_specs
        ]
        if not any(
            any(math.isfinite(value) for value in panel_x_spec.x_values)
            for panel_x_spec in panel_x_specs
        ):
            logging.info(
                "Skipping three-panel %s plots because the consolidated CSV contains no "
                "finite x-values across panel subsets.",
                x_axis_spec.output_suffix,
            )
            continue

        for output_stem, base_spec in combined_specs_by_output_stem.items():
            plot_directory = output_plot_prefix.parent / (
                f"{output_plot_prefix.name}_{output_stem}_vs_{x_axis_spec.output_suffix}"
            )
            output_path = plot_directory / THREE_PANEL_FILENAME

            panels: list[ThreePanelSubset] = []
            for panel_x_spec, panel_data in zip(
                panel_x_specs,
                three_panel_scope_specs,
                strict=False,
            ):
                variant_kind, panel_title, panel_rows, panel_spec_lookup = panel_data
                panel_metric_spec = panel_spec_lookup[output_stem]
                panels.append(
                    ThreePanelSubset(
                        variant_kind=variant_kind,
                        title=panel_title,
                        rows=panel_rows,
                        x_values=panel_x_spec.x_values,
                        y_values=panel_metric_spec.y_values,
                    )
                )

            all_panel_y_values = [value for panel in panels for value in panel.y_values]
            spec = ScatterSpec(
                y_values=all_panel_y_values,
                y_label=base_spec.y_label,
                title=(f"{base_spec.title} vs {x_axis_spec.title_suffix} "),
                output_path=output_path,
                lam_baseline=base_spec.lam_baseline,
                lam_baseline_x=lam_baseline_x,
                lam_baseline_colour=base_spec.lam_baseline_colour,
                lam_baseline_label=base_spec.lam_baseline_label,
            )
            _plot_three_panel_scatter(
                panels=panels,
                spec=spec,
                x_label=x_axis_spec.label,
                render_context=render_context,
                modes=modes,
                png_dpi=png_dpi,
            )
            output_paths.append(output_path)

            effective_broken_y_threshold = (
                DEFAULT_MEMORY_BROKEN_Y_THRESHOLD
                if "memory_peak" in output_stem
                else broken_y_threshold
            )
            gaps = _find_adjacent_y_gaps(all_panel_y_values, effective_broken_y_threshold)
            if not gaps:
                continue
            broken_output_path = _plot_three_panel_scatter_broken_y(
                panels=panels,
                spec=spec,
                x_label=x_axis_spec.label,
                render_context=render_context,
                modes=modes,
                gaps=gaps,
                png_dpi=png_dpi,
            )
            logging.debug(
                "Generated broken-y three-panel companion plot for %s/%s "
                "(%d adjacent gap(s) >= %.3f, largest ratio %.3f).",
                plot_directory.name,
                output_path.name,
                len(gaps),
                effective_broken_y_threshold,
                max(gap.ratio for gap in gaps),
            )
            output_paths.append(broken_output_path)

    return output_paths


def resolve_target_selectors(
    cli_targets: list[str] | None,
    base_config: dict[str, Any],
) -> list[str]:
    """
    Resolve target selectors from CLI, config, or hard-coded defaults.

    Priority order:
    1. CLI arguments (if provided)
    2. Config key `inference.benchmark_targets` (if valid)
    3. Hard-coded defaults based on `inference.data_set` (LOCATA vs others)

    Parameters
    ----------
    cli_targets : list[str] | None
        List of target selectors provided via CLI.
    base_config : dict[str, Any]
        Base configuration dictionary.

    Returns
    -------
    list[str]
        List of resolved target selectors.
    """
    if cli_targets:
        return cli_targets

    inference = base_config.get("inference")
    if not isinstance(inference, dict):
        return DEFAULT_TARGET_SELECTORS

    data_set = str(inference.get("data_set", "")).strip().lower()
    default_targets = LOCATA_TARGET_SELECTORS if data_set == "locata" else DEFAULT_TARGET_SELECTORS

    config_targets = inference.get("benchmark_targets")
    if config_targets is None:
        return default_targets
    if not isinstance(config_targets, list) or not all(
        isinstance(target, str) for target in config_targets
    ):
        raise ValueError(
            "Config key inference.benchmark_targets must be a list of strings, "
            "e.g. ['bicubiclam', 'uplam', 'ainnlam_e2e_auxen']."
        )
    if not config_targets:
        raise ValueError("Config key inference.benchmark_targets is empty.")
    return config_targets


def resolve_plot_modes(cli_modes: list[str] | None) -> set[str]:
    """
    Resolve plot modes from CLI values.

    Parameters
    ----------
    cli_modes : list[str] | None
        Plot mode values passed via the command line.

    Returns
    -------
    set[str]
        Normalised set of active plot mode identifiers.
    """
    if not cli_modes:
        return set()
    return set(cli_modes)


def resolve_output_plot_prefix(
    repo_root: Path,
    raw_output_plot: str | None,
    *,
    default_dir: Path,
) -> Path:
    """
    Resolve the output plot prefix, applying the repository-root defaulting rules.

    Parameters
    ----------
    repo_root : Path
        Repository root for resolving relative paths.
    raw_output_plot : str | None
        Raw CLI value from ``--output-plot``.
    default_dir : Path
        Default directory used when ``--output-plot`` is omitted.

    Returns
    -------
    Path
        Plot output prefix without a filename suffix.
    """
    output_plot = (
        _resolve_path(repo_root, raw_output_plot)
        if raw_output_plot
        else default_dir / BENCHMARK_PLOT_BASENAME
    )
    return output_plot.with_suffix("")


def log_comparison_outputs(output_csv: Path, plot_paths: list[Path]) -> None:
    """
    Log the directories containing the comparison CSV and generated plots.

    Parameters
    ----------
    output_csv : Path
        Path to the consolidated benchmark CSV.
    plot_paths : list[Path]
        Generated plot paths.
    """
    logging.info("Comparison artefacts written:")
    logging.info("- CSV directory: %s", output_csv.parent)
    logging.debug("- CSV file: %s", output_csv)

    plot_directories = sorted({plot_path.parent for plot_path in plot_paths})
    if plot_directories:
        logging.info("- Plot directories:")
        for plot_directory in plot_directories:
            logging.info("  %s", plot_directory)
    else:
        logging.info("- Plot directories: none")

    broken_y_paths = [plot_path for plot_path in plot_paths if plot_path.stem.endswith("_broken_y")]
    broken_y_directories = sorted({plot_path.parent for plot_path in broken_y_paths})
    if broken_y_directories:
        logging.info("- Broken-y plot directories:")
        for plot_directory in broken_y_directories:
            logging.info("  %s", plot_directory)
    else:
        logging.info("- Broken-y plot directories: none")

    three_panel_paths = sorted(
        plot_path for plot_path in plot_paths if plot_path.name == THREE_PANEL_FILENAME
    )
    if three_panel_paths:
        logging.info("- Three-panel plot files:")
        for plot_path in three_panel_paths:
            logging.info("  %s", plot_path)
    else:
        logging.info("- Three-panel plot files: none")

    three_panel_broken_paths = sorted(
        plot_path for plot_path in plot_paths if plot_path.name == THREE_PANEL_BROKEN_Y_FILENAME
    )
    if three_panel_broken_paths:
        logging.info("- Broken-y three-panel plot files:")
        for plot_path in three_panel_broken_paths:
            logging.info("  %s", plot_path)
    else:
        logging.info("- Broken-y three-panel plot files: none")


def run_benchmark_comparison(  # noqa: PLR0913
    *,
    repo_root: Path,
    base_config: dict[str, Any],
    target_selectors: list[str],
    device: str,
    output_csv: Path,
    output_plot_prefix: Path,
    plot_modes: set[str],
    override_style_guide: bool,
    broken_y_threshold: float,
    png_dpi: int = DEFAULT_PNG_DPI,
) -> tuple[Path, list[Path]]:
    """
    Execute fresh benchmark inference runs and generate the consolidated artefacts.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    base_config : dict[str, Any]
        Base inference configuration.
    target_selectors : list[str]
        Target selectors resolved from CLI/config defaults.
    device : str
        Device forwarded to ``src/infer.py``.
    output_csv : Path
        Destination for the consolidated benchmark CSV.
    output_plot_prefix : Path
        Prefix for generated plot directories.
    plot_modes : set[str]
        Active plot presentation modes.
    override_style_guide : bool
        Whether to override stored family labels and colours with the current style guide when
        plotting from the consolidated CSV.
    broken_y_threshold : float
        Broken-axis threshold for non-memory plots.

    Returns
    -------
    tuple[Path, list[Path]]
        Consolidated CSV path and generated plot paths.
    """
    logging.info("Requested targets: %s", ", ".join(target_selectors))
    logging.info("Supported selectors: %s", ", ".join(supported_selector_ids()))
    variants = expand_target_selectors(
        target_selectors,
        repo_root=repo_root,
        logger=logging.getLogger(),
    )
    logging.info("Resolved runnable variants: %s", ", ".join(v.variant_id for v in variants))

    inference_config = base_config["inference"]
    output_root_raw = str(inference_config.get("output_path", "output"))
    output_root = _resolve_path(repo_root, output_root_raw)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_width_ms = float(inference_config.get("frame_width_ms", 100.0))

    results: list[ModelResult] = []
    for variant in variants:
        logging.info(
            "[%s] running inference for family=%s (%s), kind=%s...",
            variant.variant_id,
            variant.family_id,
            variant.family_label,
            variant.variant_kind,
        )
        model_config = build_model_config(base_config, variant)
        metrics_path = run_inference_once(
            repo_root=repo_root,
            output_root=output_root,
            variant=variant,
            config_payload=model_config,
            device=device,
        )
        raw_metrics = aggregate_metrics(metrics_path)

        normalised_metrics_path: Path | None = None
        normalised_metrics = AggregatedBenchmarkMetrics(
            files_processed=0,
            total_frames=0,
            latency_per_frame_ms=float("nan"),
            gflops_per_frame=float("nan"),
            lam_latency_per_frame_ms=float("nan"),
            lam_gflops_per_frame=float("nan"),
            memory_peak_max_mb=float("nan"),
            memory_peak_median_mb=float("nan"),
            cmd_reference_to_upsampler_median=float("nan"),
            cmd_upsampler_to_lam_median=float("nan"),
            cmd_reference_to_lam_median=float("nan"),
            cmd_reference_to_lam_denoise1_median=float("nan"),
            cmd_reference_to_lam_denoise2_median=float("nan"),
            cmd_reference_to_lam_denoise3_median=float("nan"),
            cmd_reference_to_lam_denoise4_median=float("nan"),
            localisation_error_deg=float("nan"),
            localisation_recall=float("nan"),
            total_params=0,
            file_ids=(),
        )
        normalised_enabled = bool(model_config["inference"].get("normalised_memory_enabled", True))
        normalised_duration_sec = float(
            model_config["inference"].get("normalised_memory_duration_sec", 10.0) or 0.0
        )
        if normalised_enabled:
            logging.info(
                "[%s] running normalised %.3f s runtime-only memory benchmark...",
                variant.variant_id,
                normalised_duration_sec,
            )
            normalised_config = build_normalised_model_config(
                base_config,
                variant,
                selected_file_ids=raw_metrics.file_ids,
            )
            normalised_metrics_path = run_inference_once(
                repo_root=repo_root,
                output_root=output_root,
                variant=variant,
                config_payload=normalised_config,
                device=device,
            )
            normalised_metrics = aggregate_metrics(normalised_metrics_path)
        else:
            logging.info(
                "[%s] normalised memory benchmark disabled in the config.",
                variant.variant_id,
            )

        result = ModelResult(
            variant_id=variant.variant_id,
            family_id=variant.family_id,
            family_label=variant.family_label,
            variant_kind=variant.variant_kind,
            infer_model_name=variant.infer_model_name,
            family_colour=variant.colour,
            run_dir=metrics_path.parent,
            metrics_json_path=metrics_path,
            normalised_metrics_json_path=normalised_metrics_path,
            files_processed=raw_metrics.files_processed,
            total_frames=raw_metrics.total_frames,
            frame_width_ms=frame_width_ms,
            latency_per_frame_ms=raw_metrics.latency_per_frame_ms,
            gflops_per_frame=raw_metrics.gflops_per_frame,
            lam_latency_per_frame_ms=raw_metrics.lam_latency_per_frame_ms,
            lam_gflops_per_frame=raw_metrics.lam_gflops_per_frame,
            memory_peak_max_mb=raw_metrics.memory_peak_max_mb,
            memory_peak_median_mb=raw_metrics.memory_peak_median_mb,
            cmd_reference_to_upsampler_median=raw_metrics.cmd_reference_to_upsampler_median,
            cmd_upsampler_to_lam_median=raw_metrics.cmd_upsampler_to_lam_median,
            cmd_reference_to_lam_median=raw_metrics.cmd_reference_to_lam_median,
            cmd_reference_to_lam_denoise1_median=raw_metrics.cmd_reference_to_lam_denoise1_median,
            cmd_reference_to_lam_denoise2_median=raw_metrics.cmd_reference_to_lam_denoise2_median,
            cmd_reference_to_lam_denoise3_median=raw_metrics.cmd_reference_to_lam_denoise3_median,
            cmd_reference_to_lam_denoise4_median=raw_metrics.cmd_reference_to_lam_denoise4_median,
            normalised_memory_peak_max_mb=normalised_metrics.memory_peak_max_mb,
            normalised_memory_peak_median_mb=normalised_metrics.memory_peak_median_mb,
            normalised_memory_duration_sec=(
                normalised_duration_sec if normalised_enabled else float("nan")
            ),
            localisation_error_deg=raw_metrics.localisation_error_deg,
            localisation_recall=raw_metrics.localisation_recall,
            total_params=raw_metrics.total_params,
        )
        results.append(result)

    write_comparison_csv(output_csv, results)
    plot_paths = plot_from_csv(
        output_csv,
        output_plot_prefix,
        frame_width_ms=frame_width_ms,
        modes=plot_modes,
        override_style_guide=override_style_guide,
        broken_y_threshold=broken_y_threshold,
        png_dpi=png_dpi,
    )
    return output_csv, plot_paths


def replay_benchmark_plots(  # noqa: PLR0913
    *,
    comparison_csv: Path,
    output_plot_prefix: Path,
    fallback_frame_width_ms: float,
    plot_modes: set[str],
    override_style_guide: bool,
    broken_y_threshold: float,
    png_dpi: int = DEFAULT_PNG_DPI,
) -> tuple[Path, list[Path]]:
    """
    Rebuild plots from an existing consolidated benchmark CSV.

    Parameters
    ----------
    comparison_csv : Path
        Existing consolidated benchmark CSV.
    output_plot_prefix : Path
        Prefix for regenerated plot directories.
    fallback_frame_width_ms : float
        Base-config fallback used for legacy CSVs.
    plot_modes : set[str]
        Active plot presentation modes.
    override_style_guide : bool
        Whether to override stored family labels and colours with the current style guide when
        plotting from the consolidated CSV.
    broken_y_threshold : float
        Broken-axis threshold for non-memory plots.

    Returns
    -------
    tuple[Path, list[Path]]
        Consolidated CSV path and generated plot paths.
    """
    plot_paths = plot_from_csv(
        comparison_csv,
        output_plot_prefix,
        frame_width_ms=fallback_frame_width_ms,
        modes=plot_modes,
        override_style_guide=override_style_guide,
        broken_y_threshold=broken_y_threshold,
        png_dpi=png_dpi,
    )
    return comparison_csv, plot_paths


def main() -> None:  # noqa: PLR0915
    """Run benchmark inference for selected targets and generate comparison artefacts."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = _resolve_path(repo_root, args.base_config)
    base_config = load_base_config(base_config_path)
    inference_config = base_config["inference"]
    setup_logging(inference_config)

    logging.info("Starting benchmark evaluation...")
    logging.info("Using base config: %s", base_config_path.resolve())
    plot_modes = resolve_plot_modes(args.mode)
    if plot_modes:
        logging.info("Active plot modes: %s", ", ".join(sorted(plot_modes)))
    if args.override_style_guide:
        logging.info("Style-guide override enabled for plot labels and colours.")
    fallback_frame_width_ms = float(inference_config.get("frame_width_ms", 100.0))

    if args.results:
        results_dir, comparison_csv = resolve_results_csv(repo_root, args.results)
        logging.info("Replay mode active: reusing existing comparison CSV: %s", comparison_csv)
        logging.info(
            "Replay mode active: skipping target resolution, inference runs, and CSV writing."
        )
        logging.info("Replay mode active: ignoring --device=%s.", args.device)
        output_plot_prefix = resolve_output_plot_prefix(
            repo_root,
            args.output_plot,
            default_dir=results_dir,
        )
        output_csv, plot_paths = replay_benchmark_plots(
            comparison_csv=comparison_csv,
            output_plot_prefix=output_plot_prefix,
            fallback_frame_width_ms=fallback_frame_width_ms,
            plot_modes=plot_modes,
            override_style_guide=args.override_style_guide,
            broken_y_threshold=args.broken_y_threshold,
            png_dpi=args.png_dpi,
        )
        log_comparison_outputs(output_csv, plot_paths)
        return

    target_selectors = resolve_target_selectors(args.targets, base_config)
    output_root_raw = str(inference_config.get("output_path", "output"))
    output_root = _resolve_path(repo_root, output_root_raw)
    output_root.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_out_dir = output_root / f"benchmark-comparison-{run_timestamp}"
    output_csv = (
        _resolve_path(repo_root, args.output_csv)
        if args.output_csv
        else default_out_dir / BENCHMARK_RESULTS_CSV_NAME
    )
    output_plot_prefix = resolve_output_plot_prefix(
        repo_root,
        args.output_plot,
        default_dir=default_out_dir,
    )
    output_csv, plot_paths = run_benchmark_comparison(
        repo_root=repo_root,
        base_config=base_config,
        target_selectors=target_selectors,
        device=args.device,
        output_csv=output_csv,
        output_plot_prefix=output_plot_prefix,
        plot_modes=plot_modes,
        override_style_guide=args.override_style_guide,
        broken_y_threshold=args.broken_y_threshold,
        png_dpi=args.png_dpi,
    )
    log_comparison_outputs(output_csv, plot_paths)


if __name__ == "__main__":
    main()
