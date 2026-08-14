"""Synthetic smoke tests for variable-channel training integration."""

from __future__ import annotations

import sys
import types
from typing import Any

import torch
from torch.utils.data import Dataset

basemap_module = types.ModuleType("mpl_toolkits.basemap")
basemap_module.Basemap = object  # type: ignore[attr-defined]
sys.modules.setdefault("mpl_toolkits.basemap", basemap_module)

from lam_min.model.VariableSRCNNLAM import VariableSRCNNLAM  # noqa: E402
from train_upsamplers import run_epoch as run_upsampler_epoch  # noqa: E402
from training.end_to_end import build_lam_loss  # noqa: E402
from training.end_to_end import run_epoch as run_e2e_epoch  # noqa: E402
from upsampler.srcnn import VariableSRCNNUpsampler  # noqa: E402
from utils.training_utils import build_train_loader  # noqa: E402


class TinyTrainingDataset(Dataset[dict[str, Any]]):
    """Return deterministic full-channel training CSMs."""

    def __init__(self, size: int) -> None:
        """Store the requested dataset size."""
        self.size = size

    def __len__(self) -> int:
        """Return the sample count."""
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one positive-semidefinite nine-band CSM."""
        generator = torch.Generator().manual_seed(index)
        vectors = torch.randn(1, 9, 32, 3, generator=generator, dtype=torch.complex64)
        return {"S_high": vectors @ vectors.transpose(-1, -2).conj(), "file_id": str(index)}


def test_standalone_epoch_reports_every_microphone_count() -> None:
    """Train one synthetic pass and expose count-specific losses."""
    counts = (4, 8, 16, 24, 32)
    loader = build_train_loader(
        [TinyTrainingDataset(5)],
        0,
        torch.device("cpu"),
        "proportional",
        variable_channel_counts=counts,
        seed=3,
    )
    model = VariableSRCNNUpsampler(feature_channels=2, mapping_channels=2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = run_upsampler_epoch(model, loader, torch.device("cpu"), 1, optimiser, epoch=1)
    assert all(f"loss_{count}ch" in metrics for count in counts)
    assert metrics["loss_32ch"] == 0.0


def test_end_to_end_epoch_accepts_observed_indices() -> None:
    """Run the variable wrapper through the shared LAM validation path."""
    loader = build_train_loader(
        [TinyTrainingDataset(1)],
        0,
        torch.device("cpu"),
        "proportional",
        variable_channel_counts=(4,),
        seed=3,
        shuffle=False,
    )
    model = VariableSRCNNLAM(
        num_bands=9,
        feature_channels=2,
        mapping_channels=2,
        variable_input_channel_counts=(4,),
        freeze_lam=False,
    )
    lam_loss = build_lam_loss(
        {"lam_method": "original_msetv", "lam_tv_weight": 1e-5},
        device=torch.device("cpu"),
    )
    metrics = run_e2e_epoch(
        model,
        loader=loader,
        device=torch.device("cpu"),
        frame_batch_size=1,
        lam_loss=lam_loss,
        aux_enabled=True,
        aux_weight_config=0.1,
        effective_aux_weight=0.1,
        aux_baseline_ratio=1.0,
        use_model_specific_aux_loss=True,
        optimiser=None,
        epoch=1,
    )
    assert metrics["loss_4ch"] == metrics["loss_total"]
    assert metrics["loss_aux_raw"] > 0


def test_32ch_end_to_end_updates_lam_with_zero_completion_loss() -> None:
    """Keep full input unchanged while allowing the end-to-end LAM branch to train."""
    loader = build_train_loader(
        [TinyTrainingDataset(1)],
        0,
        torch.device("cpu"),
        "proportional",
        variable_channel_counts=(32,),
        seed=5,
        shuffle=False,
    )
    model = VariableSRCNNLAM(
        num_bands=9,
        feature_channels=2,
        mapping_channels=2,
        variable_input_channel_counts=(32,),
        freeze_lam=False,
    )
    lam_loss = build_lam_loss(
        {"lam_method": "original_msetv", "lam_tv_weight": 1e-5},
        device=torch.device("cpu"),
    )
    before = model.lam.D.detach().clone()
    metrics = run_e2e_epoch(
        model,
        loader=loader,
        device=torch.device("cpu"),
        frame_batch_size=1,
        lam_loss=lam_loss,
        aux_enabled=True,
        aux_weight_config=0.1,
        effective_aux_weight=0.1,
        aux_baseline_ratio=1.0,
        use_model_specific_aux_loss=True,
        optimiser=torch.optim.Adam(model.parameters(), lr=1e-5),
        epoch=1,
    )
    assert metrics["loss_aux_raw"] == 0
    assert not torch.equal(before, model.lam.D)
