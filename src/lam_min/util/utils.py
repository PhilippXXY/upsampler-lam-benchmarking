"""
Inference-only utilities for model loading and coordinate conversion.

Trimmed from Roman et al. repository to retain only essential inference functions:
    - Checkpoint loading for pretrained LAM/UpLAM models
    - Dynamic module instantiation from configuration dictionaries
    - Polar-to-Cartesian coordinate conversion for output formatting
    - Execution timing helper

All training, plotting, and mathematical table utilities have been removed
for this minimal inference implementation.

References
----------
.. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import time
from typing import Any

import numpy as np
import torch

SAFMN_DEFAULT_FEATURE_CHANNELS = 36
SAFMN_DEFAULT_N_BLOCKS = 8
SAFMN_DEFAULT_FFN_SCALE = 2.0
SAFMN_DEFAULT_N_LEVELS = 4


def _load_lam_from_state_dict(lam_state: dict[str, Any], model: Any) -> None:
    """
    Load LAM state from a checkpoint state dictionary into a model's LAM submodule.

    Parameters
    ----------
    lam_state : dict[str, Any]
        State dictionary containing LAM parameters
    model : Any
        Model instance with a "lam" attribute (e.g., BicubicLAM, IMDNLAM, SRCNNLAM, SAFMNLAM)
        to load state into
    """
    if any(key.startswith("lam.") for key in lam_state):
        lam_state = {
            key[len("lam.") :]: value for key, value in lam_state.items() if key.startswith("lam.")
        }
    missing, unexpected = model.lam.load_state_dict(lam_state, strict=False)
    if missing:
        logging.warning("Missing LAM keys when loading checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected LAM keys when loading checkpoint: %s", unexpected)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """
    Load a pretrained model checkpoint and return state dictionary.

    Supports both .pth and .tar checkpoint formats. For .tar files,
    extracts the model state from the 'model' key.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar)
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA)

    Returns
    -------
    dict[str, Any]
        Model state dictionary suitable for model.load_state_dict()

    Raises
    ------
    ValueError
        If checkpoint extension is not .pth or .tar

    Examples
    --------
    >>> device = torch.device("cuda:0")
    >>> state = load_checkpoint("checkpoints/UpLAM.pth", device)
    >>> model.load_state_dict(state)
    """
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    if ext not in (".pth", ".tar"):
        raise ValueError("Only support .pth and .tar checkpoints.")

    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        return model_checkpoint
    # .tar
    return model_checkpoint["model"]


def _extract_safmn_upsampler_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract SAFMN upsampler keys from a checkpoint state dict.

    Supports both raw SAFMN upsampler checkpoints (`to_feat.*`, `feats.*`, ...)
    and wrapper checkpoints prefixed with `upsampler.`.

    Parameters
    ----------
    state : dict[str, Any]
        Raw checkpoint state dictionary loaded from file.

    Returns
    -------
    dict[str, Any]
        Extracted state dictionary containing only upsampler keys,
        with any wrapper prefixes removed.
        If no upsampler keys are found, returns the original state dict.
    """
    if any(key.startswith("upsampler.") for key in state):
        extracted = {
            key[len("upsampler.") :]: value
            for key, value in state.items()
            if key.startswith("upsampler.")
        }
        if extracted:
            return extracted
    return state


def infer_safmn_architecture_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> dict[str, int | float] | None:
    """
    Infer SAFMN architecture parameters from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file.
    device : torch.device
        Device used for checkpoint loading.

    Returns
    -------
    dict[str, int | float] | None
        Inferred values for `feature_channels`, `n_blocks`, `ffn_scale`,
        and `n_levels`, or None when inference is not possible.
    """
    try:
        raw_state = load_checkpoint(checkpoint_path, device)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logging.warning("Could not inspect SAFMN checkpoint '%s': %s", checkpoint_path, exc)
        return None

    state = _extract_safmn_upsampler_state(raw_state)

    to_feat_weight = state.get("to_feat.weight")
    if to_feat_weight is None or to_feat_weight.ndim < 1:
        return None
    feature_channels = int(to_feat_weight.shape[0])

    block_pattern = re.compile(r"^feats\.(\d+)\.")
    level_pattern = re.compile(r"^feats\.\d+\.safm\.mfr\.(\d+)\.")

    block_indices = {
        int(match.group(1)) for key in state if (match := block_pattern.match(key)) is not None
    }
    level_indices = {
        int(match.group(1)) for key in state if (match := level_pattern.match(key)) is not None
    }

    n_blocks = (max(block_indices) + 1) if block_indices else SAFMN_DEFAULT_N_BLOCKS
    n_levels = (max(level_indices) + 1) if level_indices else SAFMN_DEFAULT_N_LEVELS

    ffn_scale = SAFMN_DEFAULT_FFN_SCALE
    ccm0_weight = state.get("feats.0.ccm.ccm.0.weight")
    if ccm0_weight is not None and ccm0_weight.ndim > 1 and int(ccm0_weight.shape[1]) > 0:
        ffn_scale = float(ccm0_weight.shape[0] / ccm0_weight.shape[1])

    return {
        "feature_channels": feature_channels,
        "n_blocks": int(n_blocks),
        "ffn_scale": float(ffn_scale),
        "n_levels": int(n_levels),
    }


def resolve_safmn_inference_architecture(
    inference_config: dict[str, Any], checkpoint_path: str, device: torch.device
) -> tuple[dict[str, int | float], dict[str, int | float] | None]:
    """
    Resolve SAFMN architecture for inference from config and checkpoint metadata.

    Config values override inferred checkpoint values. If neither is available,
    SAFMN defaults are used.

    Parameters
    ----------
    inference_config : dict[str, Any]
        Inference configuration dictionary, potentially containing SAFMN architecture parameters.
    checkpoint_path : str
        Path to SAFMN checkpoint for potential architecture inference.
    device : torch.device
        Device used for checkpoint loading during inference.

    Returns
    -------
    tuple[dict[str, int | float], dict[str, int | float] | None]
        A tuple containing:
        - resolved: Dictionary of resolved SAFMN architecture parameters to use for model
        instantiation.
        - inferred: Dictionary of SAFMN architecture parameters inferred from checkpoint,
        or None if inference failed
    """
    inferred = infer_safmn_architecture_from_checkpoint(checkpoint_path, device)

    resolved = {
        "feature_channels": int(
            inference_config.get(
                "feature_channels",
                inferred["feature_channels"] if inferred else SAFMN_DEFAULT_FEATURE_CHANNELS,
            )
        ),
        "n_blocks": int(
            inference_config.get(
                "n_blocks",
                inferred["n_blocks"] if inferred else SAFMN_DEFAULT_N_BLOCKS,
            )
        ),
        "ffn_scale": float(
            inference_config.get(
                "ffn_scale",
                inferred["ffn_scale"] if inferred else SAFMN_DEFAULT_FFN_SCALE,
            )
        ),
        "n_levels": int(
            inference_config.get(
                "n_levels",
                inferred["n_levels"] if inferred else SAFMN_DEFAULT_N_LEVELS,
            )
        ),
    }

    return resolved, inferred


def load_bicubic_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into a BicubicLAM model.

    Parameters
    ----------
    model : Any
        BicubicLAM model instance to load state into
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA)
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading an upsampler-only checkpoint

    Returns
    -------
    None
        Loads state into model in-place; does not return anything
    """
    checkpoint_path = checkpoint_path.strip()

    if not checkpoint_path:
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            return
        raise FileNotFoundError(
            "BicubicLAM does not use an upsampler checkpoint and requires a valid "
            f"'lam_checkpoint'. Received empty checkpoint path and invalid lam_checkpoint: "
            f"{lam_checkpoint!r}"
        )

    if not os.path.exists(checkpoint_path):
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            return
        raise FileNotFoundError(
            "Bicubic checkpoint not found and no valid lam_checkpoint provided. "
            f"Missing: {checkpoint_path}"
        )

    state = load_checkpoint(checkpoint_path, device)

    # If checkpoint contains prefixed keys for either LAM or upsampler,
    # load directly with non-strict matching
    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_raw_lam = "D" in state or "tau" in state

    # BicubicLAM only needs LAM weights. Wrapper checkpoints therefore either
    # expose `lam.*` keys or contain the raw LAM state dict.
    if has_prefixed_lam:
        _load_lam_from_state_dict(state, model)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    # Bicubic upsampling has no learnable parameters. Empty checkpoints are
    # therefore valid only as placeholders and still require a separate LAM
    # checkpoint for actual model initialisation.
    if len(state) == 0:
        if lam_checkpoint is not None:
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded an empty BicubicLAM checkpoint but no 'lam_checkpoint' provided. "
                "LAM remains randomly initialised."
            )
        return

    if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
        logging.warning(
            "Ignoring incompatible BicubicLAM checkpoint '%s'. Bicubic upsampling has no "
            "learnable weights, and this file does not contain LAM parameters. "
            "Loading '%s' instead.",
            checkpoint_path,
            lam_checkpoint,
        )
        lam_state = load_checkpoint(lam_checkpoint, device)
        _load_lam_from_state_dict(lam_state, model)
        return

    logging.warning(
        "Ignoring incompatible BicubicLAM checkpoint '%s'. Bicubic upsampling has no "
        "learnable weights, and no valid 'lam_checkpoint' was provided. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def load_uplam_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into an UpLAM model.

    Parameters
    ----------
    model : Any
        UpLAM model instance to load state into.
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state.
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA).
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading CDBPN-only weights
        or when warming only the LAM branch.
    """
    checkpoint_path = checkpoint_path.strip()

    if not checkpoint_path:
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            logging.warning(
                "Initialising UpLAM without a CDBPN checkpoint. "
                "LAM was loaded from '%s' and CDBPN remains randomly initialised.",
                lam_checkpoint,
            )
            return
        raise FileNotFoundError(
            "UpLAM requires either a valid checkpoint_path or lam_checkpoint. "
            f"Received empty checkpoint path and invalid lam_checkpoint: {lam_checkpoint!r}"
        )

    if not os.path.exists(checkpoint_path):
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            logging.warning(
                "UpLAM checkpoint '%s' was not found. Loaded LAM from '%s' instead and left "
                "CDBPN randomly initialised.",
                checkpoint_path,
                lam_checkpoint,
            )
            return
        raise FileNotFoundError(
            "UpLAM checkpoint not found and no valid lam_checkpoint provided. "
            f"Missing: {checkpoint_path}"
        )

    state = load_checkpoint(checkpoint_path, device)

    raw_upsampler_prefixes = (
        "feat0_rel.",
        "feat0_imag.",
        "feat1_rel.",
        "feat1_imag.",
        "up1_rel.",
        "up1_imag.",
        "down1_rel.",
        "down1_imag.",
        "up2_rel.",
        "up2_imag.",
        "output_conv_rel.",
        "output_conv_imag.",
    )
    prefixed_upsampler_prefixes = tuple(f"cdbpn.{prefix}" for prefix in raw_upsampler_prefixes)

    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_prefixed_upsampler = any(key.startswith(prefixed_upsampler_prefixes) for key in state)
    has_raw_lam = "D" in state or "tau" in state
    has_raw_upsampler = any(key.startswith(raw_upsampler_prefixes) for key in state)

    if has_prefixed_lam or has_prefixed_upsampler:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing UpLAM keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected UpLAM keys when loading checkpoint: %s", unexpected)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    if has_raw_upsampler:
        missing, unexpected = model.cdbpn.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing CDBPN keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected CDBPN keys when loading checkpoint: %s", unexpected)
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded CDBPN-only UpLAM checkpoint but no valid 'lam_checkpoint' was provided. "
                "LAM remains randomly initialised."
            )
        return

    if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
        logging.warning(
            "Ignoring incompatible BicubicLAM checkpoint '%s'. Bicubic upsampling has no "
            "learnable weights, and this file does not contain LAM parameters. "
            "Loading '%s' instead.",
            checkpoint_path,
            lam_checkpoint,
        )
        lam_state = load_checkpoint(lam_checkpoint, device)
        _load_lam_from_state_dict(lam_state, model)
        return

    logging.warning(
        "Could not infer checkpoint format for UpLAM from %s. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def load_imdn_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into an IMDNLAM model.

    Parameters
    ----------
    model : Any
        IMDNLAM model instance to load state into
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA)
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading an upsampler-only checkpoint

    Returns
    -------
    None
        Loads state into model in-place; does not return anything
    """
    if not os.path.exists(checkpoint_path):
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            return
        raise FileNotFoundError(
            "IMDN checkpoint not found and no valid lam_checkpoint provided. "
            f"Missing: {checkpoint_path}"
        )

    state = load_checkpoint(checkpoint_path, device)

    # IMDNUpsampler has an internal submodule called `upsampler`, so raw IMDN
    # checkpoints contain keys like `upsampler.0.*`. We therefore cannot treat
    # every `upsampler.*` key as wrapper-prefixed IMDNLAM keys.
    raw_upsampler_prefixes = (
        "fea_conv.",
        "IMDB1.",
        "IMDB2.",
        "IMDB3.",
        "IMDB4.",
        "IMDB5.",
        "IMDB6.",
        "c.",
        "LR_conv.",
        "upsampler.",
    )
    prefixed_upsampler_prefixes = tuple(f"upsampler.{prefix}" for prefix in raw_upsampler_prefixes)

    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_prefixed_upsampler = any(key.startswith(prefixed_upsampler_prefixes) for key in state)
    has_raw_lam = "D" in state or "tau" in state
    has_raw_upsampler = any(key.startswith(raw_upsampler_prefixes) for key in state)

    if has_prefixed_lam or has_prefixed_upsampler:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing IMDNLAM keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected IMDNLAM keys when loading checkpoint: %s", unexpected)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    # Raw IMDN upsampler checkpoint.
    if has_raw_upsampler:
        missing, unexpected = model.upsampler.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing upsampler keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected upsampler keys when loading checkpoint: %s", unexpected)
        if lam_checkpoint is not None:
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded upsampler-only IMDN checkpoint but no 'lam_checkpoint' provided. "
                "LAM remains randomly initialised."
            )
        return

    logging.warning(
        "Could not infer checkpoint format for IMDNLAM from %s. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def load_safmn_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into a SAFMNLAM model.

    Parameters
    ----------
    model : Any
        SAFMNLAM model instance to load state into
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA)
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading an upsampler-only checkpoint

    Returns
    -------
    None
        Loads state into model in-place; does not return anything
    """
    if not os.path.exists(checkpoint_path):
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            return
        raise FileNotFoundError(
            "SAFMN checkpoint not found and no valid lam_checkpoint provided. "
            f"Missing: {checkpoint_path}"
        )

    state = load_checkpoint(checkpoint_path, device)

    raw_upsampler_prefixes = ("to_feat.", "feats.", "to_img.")
    prefixed_upsampler_prefixes = tuple(f"upsampler.{prefix}" for prefix in raw_upsampler_prefixes)

    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_prefixed_upsampler = any(key.startswith(prefixed_upsampler_prefixes) for key in state)
    has_raw_lam = "D" in state or "tau" in state
    has_raw_upsampler = any(key.startswith(raw_upsampler_prefixes) for key in state)

    if has_prefixed_lam or has_prefixed_upsampler:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing SAFMNLAM keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected SAFMNLAM keys when loading checkpoint: %s", unexpected)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    # Raw SAFMN upsampler checkpoint.
    if has_raw_upsampler:
        missing, unexpected = model.upsampler.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing upsampler keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected upsampler keys when loading checkpoint: %s", unexpected)
        if lam_checkpoint is not None:
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded upsampler-only SAFMN checkpoint but no 'lam_checkpoint' provided. "
                "LAM remains randomly initialised."
            )
        return

    logging.warning(
        "Could not infer checkpoint format for SAFMNLAM from %s. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def load_srcnn_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into an SRCNNLAM model.

    Parameters
    ----------
    model : Any
        SRCNNLAM model instance to load state into
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA)
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading an upsampler-only checkpoint

    Returns
    -------
    None
        Loads state into model in-place; does not return anything
    """
    state = load_checkpoint(checkpoint_path, device)

    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_prefixed_upsampler = any(key.startswith("upsampler.") for key in state)
    has_raw_lam = "D" in state or "tau" in state
    has_raw_upsampler = any(key.startswith("srcnn.") for key in state)

    # Prefer direct wrapper loading if prefixed keys are present.
    if has_prefixed_lam or has_prefixed_upsampler:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing SRCNNLAM keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected SRCNNLAM keys when loading checkpoint: %s", unexpected)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    # Raw SRCNN upsampler checkpoint.
    if has_raw_upsampler:
        missing, unexpected = model.upsampler.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing upsampler keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected upsampler keys when loading checkpoint: %s", unexpected)
        if lam_checkpoint is not None:
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded upsampler-only SRCNN checkpoint but no 'lam_checkpoint' provided. "
                "LAM remains randomly initialised."
            )
        return

    logging.warning(
        "Could not infer checkpoint format for SRCNNLAM from %s. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def load_gan_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into a GANLAM model.

    Parameters
    ----------
    model : Any
        GANLAM model instance to load state into
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA)
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading an upsampler-only checkpoint

    Returns
    -------
    None
        Loads state into model in-place; does not return anything
    """
    if not os.path.exists(checkpoint_path):
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            return
        raise FileNotFoundError(
            "GAN checkpoint not found and no valid lam_checkpoint provided. "
            f"Missing: {checkpoint_path}"
        )

    state = load_checkpoint(checkpoint_path, device)

    raw_upsampler_prefixes = (
        "generator.",
        "discriminator.",
    )
    prefixed_upsampler_prefixes = tuple(f"upsampler.{prefix}" for prefix in raw_upsampler_prefixes)

    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_prefixed_upsampler = any(key.startswith(prefixed_upsampler_prefixes) for key in state)
    has_raw_lam = "D" in state or "tau" in state
    has_raw_upsampler = any(key.startswith(raw_upsampler_prefixes) for key in state)

    if has_prefixed_lam or has_prefixed_upsampler:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing GANLAM keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected GANLAM keys when loading checkpoint: %s", unexpected)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    if has_raw_upsampler:
        missing, unexpected = model.upsampler.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing upsampler keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected upsampler keys when loading checkpoint: %s", unexpected)
        if lam_checkpoint is not None:
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded upsampler-only GAN checkpoint but no 'lam_checkpoint' provided. "
                "LAM remains randomly initialised."
            )
        return

    logging.warning(
        "Could not infer checkpoint format for GANLAM from %s. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def load_ainn_lam_state(  # noqa: C901
    model: Any, checkpoint_path: str, device: torch.device, lam_checkpoint: str | None = None
) -> None:
    """
    Load checkpoint state into an AINNLAM model.

    Parameters
    ----------
    model : Any
        AINNLAM model instance to load state into.
    checkpoint_path : str
        Path to checkpoint file (.pth or .tar) containing model state.
    device : torch.device
        Device to map checkpoint tensors to (CPU or CUDA).
    lam_checkpoint : str | None, optional
        Optional path to separate LAM checkpoint if loading an upsampler-only checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        if lam_checkpoint is not None and os.path.exists(lam_checkpoint):
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
            return
        raise FileNotFoundError(
            "AINN checkpoint not found and no valid lam_checkpoint provided. "
            f"Missing: {checkpoint_path}"
        )

    state = load_checkpoint(checkpoint_path, device)

    raw_upsampler_prefixes = (
        "model_layers.",
        "output_layer.",
    )
    prefixed_upsampler_prefixes = tuple(f"upsampler.{prefix}" for prefix in raw_upsampler_prefixes)

    has_prefixed_lam = any(key.startswith("lam.") for key in state)
    has_prefixed_upsampler = any(key.startswith(prefixed_upsampler_prefixes) for key in state)
    has_raw_lam = "D" in state or "tau" in state
    has_raw_upsampler = any(key.startswith(raw_upsampler_prefixes) for key in state)

    if has_prefixed_lam or has_prefixed_upsampler:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing AINNLAM keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected AINNLAM keys when loading checkpoint: %s", unexpected)
        return

    if has_raw_lam:
        _load_lam_from_state_dict(state, model)
        return

    if has_raw_upsampler:
        missing, unexpected = model.upsampler.load_state_dict(state, strict=False)
        if missing:
            logging.warning("Missing upsampler keys when loading checkpoint: %s", missing)
        if unexpected:
            logging.warning("Unexpected upsampler keys when loading checkpoint: %s", unexpected)
        if lam_checkpoint is not None:
            lam_state = load_checkpoint(lam_checkpoint, device)
            _load_lam_from_state_dict(lam_state, model)
        else:
            logging.warning(
                "Loaded upsampler-only AINN checkpoint but no 'lam_checkpoint' provided. "
                "LAM remains randomly initialised."
            )
        return

    logging.warning(
        "Could not infer checkpoint format for AINNLAM from %s. "
        "Proceeding with model initial weights.",
        checkpoint_path,
    )


def initialize_config(module_cfg: dict[str, Any], pass_args: bool = True) -> Any:
    """
    Dynamically instantiate Python modules from JSON-like configuration.

    Loads a module by name and instantiates a specified class or function
    with provided arguments. Useful for creating models from configuration files.

    Parameters
    ----------
    module_cfg : dict[str, Any]
        Configuration dictionary with keys:
        - "module": str, full module path (e.g., "lam_min.model.LAM")
        - "main": str, class/function name to instantiate (e.g., "LAM")
        - "args": dict, keyword arguments to pass to constructor
    pass_args : bool, optional
        Whether to pass "args" to the constructor (default: True)
        If False, instantiates without arguments

    Returns
    -------
    Any
        Instantiated object (model, dataset, etc.)

    Examples
    --------
    >>> config = {
    ...     "module": "lam_min.model.UpLAM",
    ...     "main": "UpLAM",
    ...     "args": {"num_bands": 9, "scale_factor": 8}
    ... }
    >>> model = initialize_config(config)
    """
    module = importlib.import_module(module_cfg["module"])
    obj = getattr(module, module_cfg["main"])
    return obj(**module_cfg["args"]) if pass_args else obj


def convert_polar_to_cartesian(azi_polar: float, ele_polar: float) -> tuple[float, float, float]:
    """
    Convert spherical polar coordinates to Cartesian coordinates on unit sphere.

    Transforms azimuth and elevation angles to (x, y, z) coordinates,
    assuming unit radius. Used for converting DoA estimates to DCASE
    challenge output format.

    Parameters
    ----------
    azi_polar : float
        Azimuth angle in degrees [-180, 180]
        Measured from positive x-axis, counter-clockwise when viewed from above
    ele_polar : float
        Elevation angle in degrees [-90, 90]
        Measured from xy-plane, positive towards +z

    Returns
    -------
    x : float
        X coordinate on unit sphere
    y : float
        Y coordinate on unit sphere
    z : float
        Z coordinate on unit sphere

    Examples
    --------
    >>> x, y, z = convert_polar_to_cartesian(0.0, 0.0)
    >>> print(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    x=1.000, y=0.000, z=0.000

    >>> x, y, z = convert_polar_to_cartesian(90.0, 0.0)
    >>> print(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    x=0.000, y=1.000, z=0.000
    """
    ele_rad = ele_polar * np.pi / 180.0
    azi_rad = azi_polar * np.pi / 180.0
    tmp = np.cos(ele_rad)
    x = np.cos(azi_rad) * tmp
    y = np.sin(azi_rad) * tmp
    z = np.sin(ele_rad)
    return float(x), float(y), float(z)


class ExecutionTime:
    """
    Simple wall-clock timer for performance profiling.

    Tracks elapsed time from instantiation for basic timing measurements
    during inference or processing operations.

    Attributes
    ----------
    start_time : float
        Wall-clock time at instantiation (from time.time())

    Examples
    --------
    >>> timer = ExecutionTime()
    >>> # ... perform operations ...
    >>> print(f"Elapsed: {timer.duration()} seconds")
    """

    def __init__(self) -> None:
        """Initialise timer and record start time."""
        self.start_time = time.time()

    def duration(self) -> int:
        """
        Calculate elapsed time since initialisation.

        Returns
        -------
        int
            Elapsed time in seconds (rounded down to nearest integer)
        """
        return int(time.time() - self.start_time)


def get_field(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Retrieve spherical field-of-view sampling coordinates.

    Wrapper function that imports and calls get_field from trainer.utils,
    avoiding circular import dependencies.

    Parameters
    ----------
    *args : Any
        Positional arguments passed to trainer.utils.get_field
    **kwargs : Any
        Keyword arguments passed to trainer.utils.get_field

    Returns
    -------
    np.ndarray
        Spatial sampling coordinates of shape (3, N_px)

    See Also
    --------
    lam_min.trainer.utils.get_field : Core implementation
    """
    from lam_min.trainer.utils import get_field as _get_field  # noqa: PLC0415

    return _get_field(*args, **kwargs)
