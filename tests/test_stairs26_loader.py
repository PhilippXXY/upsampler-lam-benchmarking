"""Tests for the STAIRS26 inference loader."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from data.stairs26_loader import Stairs26AudioDataset, Stairs26GroundTruthLoader
from lam_min.doa_metrics import (
    compute_seld_metrics_for_files,
    compute_seld_metrics_report_for_files,
)

SAMPLE_RATE = 24_000
TWO_FILES = 2


def _write_wav(path: Path, channels: int) -> None:
    """Write a short multi-channel test recording.

    Parameters
    ----------
    path : Path
        Output WAV path.
    channels : int
        Number of audio channels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.zeros((24, channels), dtype=np.float32), SAMPLE_RATE)


def test_discovers_nested_wavs(tmp_path: Path) -> None:
    """Discover archive trees recursively."""
    _write_wav(tmp_path / "audio" / "first" / "fold1_room1_mix001.wav", 32)
    _write_wav(tmp_path / "audio" / "second" / "fold2_room1_mix002.wav", 32)

    expected_files = 2
    dataset = Stairs26AudioDataset(tmp_path)
    assert len(dataset) == expected_files
    assert dataset[0]["file_id"] == "fold1_room1_mix001"
    assert dataset[0]["audio"].shape == (24, 32)
    assert dataset[0]["sample_rate"] == SAMPLE_RATE


def test_accepts_official_four_channel_evaluation_audio(tmp_path: Path) -> None:
    """Accept the official four-channel evaluation recording format."""
    _write_wav(tmp_path / "mix001.wav", 4)
    expected_channels = 4
    assert Stairs26AudioDataset(tmp_path)[0]["audio"].shape[1] == expected_channels


def test_rejects_invalid_channel_count(tmp_path: Path) -> None:
    """Reject non-STAIRS channel counts."""
    _write_wav(tmp_path / "invalid.wav", 8)
    with pytest.raises(ValueError, match="Expected 4 or 32 channels"):
        Stairs26AudioDataset(tmp_path)[0]


def test_ground_truth_loader_converts_peak_pixels_to_doa(tmp_path: Path) -> None:
    """Convert the strongest mask pixels using STAIRS26 image coordinates."""
    metadata = tmp_path / "labels" / "nested"
    metadata.mkdir(parents=True)
    (metadata / "sample_std.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "metadata_frame_index": 496,
                        "instance_id": 1,
                        "category_id": 10,
                        "distance": 218.0,
                        "segmentation": [[[236.0, 96.0, 0.2], [241.0, 96.0, 0.8]]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    event = Stairs26GroundTruthLoader(tmp_path).load("sample")

    assert torch.equal(event.frame, torch.tensor([496]))
    assert torch.allclose(event.t_sec, torch.tensor([49.6]))
    assert torch.equal(event.active_class_index, torch.tensor([10]))
    assert torch.equal(event.source_number_index, torch.tensor([1]))
    assert torch.allclose(event.azimuth, torch.tensor([-61.0]))
    assert torch.allclose(event.elevation, torch.tensor([-6.0]))


def test_ground_truth_loader_requires_matching_json(tmp_path: Path) -> None:
    """Reject metadata roots without a matching recording label."""
    with pytest.raises(FileNotFoundError, match="sample_std.json"):
        Stairs26GroundTruthLoader(tmp_path).load("sample")


def test_ground_truth_loader_supports_seld_evaluation(tmp_path: Path) -> None:
    """Evaluate peak-converted STAIRS26 labels through the shared SELD path."""
    metadata = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    metadata.mkdir()
    predictions.mkdir()
    (metadata / "sample_std.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "metadata_frame_index": 10,
                        "instance_id": 0,
                        "category_id": 3,
                        "segmentation": [[[180.0, 90.0, 1.0]]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (predictions / "sample.csv").write_text("10,0,0,0.0,0.0\n", encoding="utf-8")

    error_rate, f_score, localisation_error, localisation_recall, seld_score, _ = (
        compute_seld_metrics_for_files(
            pred_files_path=predictions,
            gt_loader=Stairs26GroundTruthLoader(metadata),
            file_ids=["sample"],
            num_classes=1,
            use_polar_format=False,
            class_agnostic=True,
        )
    )

    assert error_rate == pytest.approx(0.0)
    assert f_score == pytest.approx(1.0)
    assert localisation_error == pytest.approx(0.0)
    assert localisation_recall == pytest.approx(1.0)
    assert seld_score == pytest.approx(0.0)


def test_seld_report_includes_prediction_ratio_and_file_standard_deviation(
    tmp_path: Path,
) -> None:
    """Report prediction/reference ratio and sample spread across files."""
    metadata = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    metadata.mkdir()
    predictions.mkdir()
    annotation = {
        "annotations": [
            {
                "metadata_frame_index": 10,
                "instance_id": 0,
                "category_id": 3,
                "segmentation": [[[180.0, 90.0, 1.0]]],
            }
        ]
    }
    for file_id in ("matched", "missed"):
        (metadata / f"{file_id}_std.json").write_text(json.dumps(annotation), encoding="utf-8")
    (predictions / "matched.csv").write_text("10,0,0,0.0,0.0\n", encoding="utf-8")
    (predictions / "missed.csv").write_text("", encoding="utf-8")

    report = compute_seld_metrics_report_for_files(
        pred_files_path=predictions,
        gt_loader=Stairs26GroundTruthLoader(metadata),
        file_ids=["matched", "missed"],
        num_classes=1,
        use_polar_format=False,
        class_agnostic=True,
    )

    assert report["prediction_to_reference_ratio"] == pytest.approx(0.5)
    assert report["files_evaluated"] == TWO_FILES
    assert report["file_level_summary"]["localisation_recall"]["estimate"] == pytest.approx(
        0.5
    )
    assert report["file_level_summary"]["localisation_recall"][
        "sample_standard_deviation"
    ] == pytest.approx(np.sqrt(0.5))
