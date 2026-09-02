"""
Evaluation utilities for DCASE-format prediction files.

Adapted from the original LAM repository and shared across dataset-specific
ground truth loaders that return DoaEvent objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from data.doa_event import DoaEvent
from lam_min import SELD_evaluation_metrics

DCASE_METRIC_NAMES = (
    "error_rate",
    "f_score",
    "localisation_error",
    "localisation_recall",
    "seld_score",
)


class GroundTruthLoader(Protocol):
    """Dataset-independent DoA ground-truth loader."""

    def load(self, file_id: str) -> DoaEvent:
        """Load one recording's events.

        Parameters
        ----------
        file_id : str
            Recording identifier.

        Returns
        -------
        DoaEvent
            Frame-level direction events.
        """
        ...


def load_output_format_file(output_format_file: str | Path) -> dict[int, list[list[float]]]:
    """
    Load DCASE output format csv file and returns it in dictionary format.

    Parameters
    ----------
    _output_format_file: str or Path
        DCASE output format CSV.

    Returns
    -------
    _output_dict: dict[int, list[list[float]]]
        Dictionary with frame index as keys and list of DoA parameters as values.
    """
    output_dict: dict[int, list[list[float]]] = {}
    with open(output_format_file, "r", encoding="utf-8") as fid:
        for line in fid:
            words = line.strip().split(",")
            frame_ind = int(words[0])
            if frame_ind not in output_dict:
                output_dict[frame_ind] = []

            if len(words) == 5:
            # frame, class idx, source_id, polar coordinates(2)
            # no distance data, for example in synthetic data fold 1 and
                output_dict[frame_ind].append(
                    [int(words[1]),
                     int(words[2]),
                     float(words[3]),
                     float(words[4])]
                )
            elif len(words) == 6:
            # frame, class idx, source_id, cartesian coordinates(3)
            # # no distance data
                output_dict[frame_ind].append(
                    [int(words[1]),
                     int(words[2]),
                     float(words[3]),
                     float(words[4]),
                     float(words[5])]
                )
            elif len(words) >= 7:
            # frame, class idx, source_id, cartesian coordinates(3), distance
                output_dict[frame_ind].append(
                    [int(words[1]),
                     int(words[2]),
                     float(words[3]),
                     float(words[4]),
                     float(words[5])]
                )

    return output_dict


def starss_event_to_output_dict(
    event: DoaEvent,
    class_agnostic: bool = False,
) -> dict[int, list[list[float]]]:
    """
    Convert a STARSS DoA event to an output dictionary with frame-indexed DoA information.

    Transforms a DoaEvent object into a dictionary mapping frame indices to lists of
    DoA parameters (class, source ID, azimuth, elevation). Distance information is ignored
    during this conversion.

    Parameters
    ----------
    event: DoaEvent
        The input STARSS DoA event containing frame, class, source, azimuth,
        and elevation information.
    class_agnostic: bool, optional
        If True, all events are assigned class ID 0, ignoring the original class information.
        If False, preserves the original class IDs from the event.
        Defaults to False.

    Returns
    -------
    : dict[int, list[list[float]]]
        A dictionary mapping frame indices (int) to lists of DoA parameters.
        Each parameter list contains:
        `[class_id (int),
        source_id (int),
        azimuth (float),
        elevation (float)]`

    Example:
        >>> output = starss_event_to_output_dict(event, class_agnostic=False)
        >>> # output[0] might be: [[1, 0, 45.5, 30.2], [1, 1, 120.3, 15.8]]
    """
    output_dict: dict[int, list[list[float]]] = {}

    frames = event.frame.tolist()
    class_ids = event.active_class_index.tolist()
    source_ids = event.source_number_index.tolist()
    azimuths = event.azimuth.tolist()
    elevations = event.elevation.tolist()

    for frame, class_id, source_id, azimuth, elevation in zip(
        frames, class_ids, source_ids, azimuths, elevations, strict=False
    ):
        frame_index = int(frame)
        if frame_index not in output_dict:
            output_dict[frame_index] = []
        out_class = 0 if class_agnostic else int(class_id)
        output_dict[frame_index].append(
            [out_class,
             int(source_id),
             float(azimuth),
             float(elevation)])

    return output_dict


def segment_labels(pred_dict: dict[int, list[list[float]]],
                   max_frames: int) -> dict[int, dict[int, list[list]]]:
    """
    Collect class-wise sound event location information in 1s segments.

    Returns dictionary_name[segment_index][class_index] = list(frame-cnt-within-segment, doa values)
    """
    nb_label_frames_1s = 10  # 10 frames per second
    nb_blocks = int(np.ceil(max_frames / float(nb_label_frames_1s)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, max_frames, nb_label_frames_1s):
        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // nb_label_frames_1s
        loc_dict: dict[int, dict[int, list[list[float]]]] = {}
        for audio_frame in range(frame_cnt, frame_cnt + nb_label_frames_1s):
            if audio_frame not in pred_dict:
                continue
            for value in pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {} # type: ignore

                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]: # type: ignore
                    loc_dict[value[0]][block_frame] = [] # type: ignore
                loc_dict[value[0]][block_frame].append(value[1:]) # type: ignore

        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict


def convert_output_format_cartesian_to_polar(
    in_dict: dict[int, list[list[float]]]
) -> dict[int, list[list[float]]]:
    """
    Convert cartesian DCASE output dict to polar (azimuth, elevation in degrees).
    """
    out_dict: dict[int, list[list[float]]] = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]
                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
    return out_dict


def compute_seld_metrics_for_files(
    pred_files_path: Path,
    gt_loader: GroundTruthLoader,
    file_ids: Iterable[str],
    num_classes: int = 13,
    doa_threshold: int = 20,
    average: str = "macro",
    use_polar_format: bool = True,
    class_agnostic: bool = False,
    file_id_mapping: dict[str, str] | None = None,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """
    Compute SELD (Sound Event Localization and Detection) metrics for a set of prediction files against ground truth data.

    Parameters
    ----------
    pred_files_path: Path
        Path to the directory containing prediction CSV files.
    gt_loader: GroundTruthLoader
        Loader object for ground truth events.
    file_ids: Iterable[str]
        Iterable of prediction file identifiers (without .csv extension).
    num_classes: int, optional
        Number of sound event classes.
        Defaults to 13.
    doa_threshold: int, optional
        DOA (Direction of Arrival) threshold in degrees for localization accuracy.
        Defaults to 20.
    average: str, optional
        Averaging method for metrics ('macro', 'micro', etc.). Defaults to "macro".
    use_polar_format: bool, optional
        Whether to convert predictions to polar coordinates.
        Defaults to True.
    class_agnostic: bool, optional
        If True, evaluate in a class-agnostic manner.
        Defaults to False.
    file_id_mapping: dict[str, str] | None, optional
        Mapping from prediction file IDs to ground truth file IDs.
        Use when prediction filenames differ from ground truth (e.g., with timestamps).
        If None, uses the same file_id for both. Defaults to None.

    Returns
    -------
    : tuple[float, float, float, float, float, np.ndarray]
        A tuple containing SELD metrics:
            - Error rate (ER)
            - F-score (F)
            - Localization error (LE)
            - Localization recall (LR)
            - SELD score
            - Per-class metric array
    """
    report = compute_seld_metrics_report_for_files(
        pred_files_path=pred_files_path,
        gt_loader=gt_loader,
        file_ids=file_ids,
        num_classes=num_classes,
        doa_threshold=doa_threshold,
        average=average,
        use_polar_format=use_polar_format,
        class_agnostic=class_agnostic,
        file_id_mapping=file_id_mapping,
    )
    metrics = report["metrics"]
    return (
        *(float(metrics[name]) for name in DCASE_METRIC_NAMES),
        np.asarray(report["classwise"]),
    )


def compute_seld_metrics_report_for_files(
    pred_files_path: Path,
    gt_loader: GroundTruthLoader,
    file_ids: Iterable[str],
    num_classes: int = 13,
    doa_threshold: int = 20,
    average: str = "macro",
    use_polar_format: bool = True,
    class_agnostic: bool = False,
    file_id_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Compute pooled and file-level SELD evaluation statistics.

    Parameters
    ----------
    pred_files_path : Path
        Directory containing DCASE prediction CSV files.
    gt_loader : GroundTruthLoader
        Ground-truth event loader.
    file_ids : Iterable[str]
        Prediction file identifiers without the ``.csv`` suffix.
    num_classes : int, optional
        Number of sound event classes.
    doa_threshold : int, optional
        Localisation threshold in degrees.
    average : str, optional
        SELD class-averaging method.
    use_polar_format : bool, optional
        Convert Cartesian predictions to polar coordinates.
    class_agnostic : bool, optional
        Evaluate all events as class zero.
    file_id_mapping : dict[str, str] | None, optional
        Prediction-to-ground-truth identifier mapping.

    Returns
    -------
    dict[str, Any]
        Pooled metrics, event counts, prediction/reference ratio, per-file metrics, and
        file-level sample statistics calculated with ``ddof=1``.
    """
    if class_agnostic:
        num_classes = 1

    eval_metrics = SELD_evaluation_metrics.SELDMetrics(
        nb_classes=num_classes, doa_threshold=doa_threshold, average=average
    )
    file_metrics: list[dict[str, float | str]] = []
    reference_event_count = predicted_event_count = 0

    for file_id in file_ids:
        pred_file = pred_files_path / f"{file_id}.csv"
        if not pred_file.exists():
            continue

        pred_dict = load_output_format_file(pred_file)
        if use_polar_format:
            pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
        if class_agnostic:
            # Convert class indices to 0 for class-agnostic evaluation
            for frame in pred_dict:
                for value in pred_dict[frame]:
                    value[0] = 0  # Set class index to 0 for class-agnostic

        # Use mapping to get ground truth file ID, or use same file_id
        gt_file_id = file_id_mapping.get(file_id, file_id) if file_id_mapping else file_id
        try:
            gt_event = gt_loader.load(gt_file_id)
        except FileNotFoundError:
            continue
        gt_dict = starss_event_to_output_dict(gt_event, class_agnostic=class_agnostic)
        if not gt_dict:
            continue

        nb_ref_frames = max(gt_dict) + 1
        pred_labels = segment_labels(pred_dict, nb_ref_frames)
        gt_labels = segment_labels(gt_dict, nb_ref_frames)
        eval_metrics.update_seld_scores(pred_labels, gt_labels)
        file_evaluator = SELD_evaluation_metrics.SELDMetrics(
            nb_classes=num_classes, doa_threshold=doa_threshold, average=average
        )
        file_evaluator.update_seld_scores(pred_labels, gt_labels)
        values = file_evaluator.compute_seld_scores()[:5]
        file_metrics.append(
            {"file_id": gt_file_id, **dict(zip(DCASE_METRIC_NAMES, map(float, values), strict=True))}
        )
        reference_event_count += sum(map(len, gt_dict.values()))
        predicted_event_count += sum(map(len, pred_dict.values()))

    ER, F, LE, LR, seld_scr, classwise = eval_metrics.compute_seld_scores()
    if not isinstance(classwise, np.ndarray):
        classwise = np.asarray(classwise)
    matrix = np.asarray(
        [[row[name] for name in DCASE_METRIC_NAMES] for row in file_metrics], dtype=float
    )
    means = np.mean(matrix, axis=0) if len(matrix) else np.full(len(DCASE_METRIC_NAMES), np.nan)
    variances = (
        np.var(matrix, axis=0, ddof=1)
        if len(matrix) > 1
        else np.full(len(DCASE_METRIC_NAMES), np.nan)
    )
    file_level_summary = {
        name: {
            "estimate": float(means[index]),
            "sample_variance": (
                float(variances[index]) if np.isfinite(variances[index]) else None
            ),
            "sample_standard_deviation": (
                float(np.sqrt(variances[index])) if np.isfinite(variances[index]) else None
            ),
        }
        for index, name in enumerate(DCASE_METRIC_NAMES)
    }
    return {
        "metrics": dict(zip(DCASE_METRIC_NAMES, map(float, (ER, F, LE, LR, seld_scr)), strict=True)),
        "classwise": classwise,
        "reference_event_count": reference_event_count,
        "predicted_event_count": predicted_event_count,
        "prediction_to_reference_ratio": (
            predicted_event_count / reference_event_count
            if reference_event_count
            else float("nan")
        ),
        "files_evaluated": len(file_metrics),
        "file_metrics": file_metrics,
        "file_level_summary": file_level_summary,
    }
