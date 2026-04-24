"""Materialise on-disk CSM caches for training datasets."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from utils.training_utils import build_dataset_list, load_conf, setup_logging

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main() -> None:  # noqa: D103
    """
    Precompute CSM tensors for training datasets.

    This improves training speed by avoiding on-the-fly CSM computation during training.
    """
    parser = argparse.ArgumentParser("Precompute training CSM tensors")
    parser.add_argument("--config", type=str, default="config/train_end_to_end.yaml")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to materialise.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional per-dataset file cap for partial cache generation.",
    )
    args = parser.parse_args()

    config = load_conf(Path(args.config))
    output_root = Path(config.get("training", {}).get("output_root", "output"))
    setup_logging(
        output_root.joinpath("logs"),
        log_stem="precompute_training_csm",
        timestamp=timestamp,
    )

    for split in args.splits:
        datasets = build_dataset_list(config=config, split=split, max_files=args.max_files)
        for dataset in datasets:
            cache_dir = getattr(dataset, "precomputed_csm_dir", None)
            if cache_dir is None:
                raise ValueError(
                    f"Dataset '{type(dataset).__name__}' has no precomputed_csm_root configured."
                )
            logging.info(
                "Precomputing %s split for %s into %s (%d files)",
                split,
                type(dataset).__name__,
                cache_dir,
                len(dataset),
            )
            for index in tqdm(
                range(len(dataset)), ncols=100, desc=f"{split}:{type(dataset).__name__}"
            ):
                dataset[index]


if __name__ == "__main__":
    main()
