"""Tests for benchmark metric aggregation."""

import json
from pathlib import Path

import pytest

from utils.benchmarking import aggregate_metrics_json
from utils.utils import _print_metrics_summary


def test_aggregate_metrics_keeps_prediction_ratio_and_standard_deviations(
    tmp_path: Path,
) -> None:
    """Carry evaluation distribution metrics into consolidated benchmark values."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {"file_id": "sample", "num_frames": 2, "total_time_ms": 4},
                {
                    "localisation_error": 10,
                    "localisation_recall": 0.75,
                    "prediction_to_reference_ratio": 1.25,
                    "file_level_summary": {
                        "localisation_error": {"sample_standard_deviation": 2.5},
                        "localisation_recall": {"sample_standard_deviation": 0.1},
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = aggregate_metrics_json(metrics_path)

    assert metrics.prediction_to_reference_ratio == pytest.approx(1.25)
    assert metrics.localisation_error_sample_standard_deviation_deg == pytest.approx(2.5)
    assert metrics.localisation_recall_sample_standard_deviation == pytest.approx(0.1)


def test_metrics_summary_preserves_pooled_dcase_estimates() -> None:
    """Keep historical pooled scores while displaying file-level spread."""
    output = _print_metrics_summary(
        [
            {"file_id": "sample", "num_frames": 1},
            {
                "seld_score": 0.4,
                "f_score": 0.8,
                "error_rate": 0.2,
                "localisation_error": 10.0,
                "localisation_recall": 0.75,
                "file_level_summary": {
                    "seld_score": {
                        "estimate": 0.9,
                        "sample_standard_deviation": 0.1,
                    }
                },
            },
        ],
        {"data_set": "test", "frame_width_ms": 100},
        n_files=1,
        device="cpu",
    )

    assert "0.4000 (file SD 0.1000)" in output
    assert "0.9000" not in output
