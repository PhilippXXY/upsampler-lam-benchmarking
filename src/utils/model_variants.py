"""Retained model variant registry for inference and benchmarking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# Set of supported variant kinds, which determine plot markers and checkpoint requirements.
# "dist" variants mean distinctly trained upsampler and LAM models
# "e2e_auxdis" variants mean e2e trained models with the auxiliary loss of the upsampler disabled
# "e2e_upfroz" variants mean e2e trained models with the upsampler fixed from the `_dist` path
# and only the LAM branch updated
# "e2e_auxen" variants mean e2e trained models with the auxiliary loss of the upsampler enabled
VARIANT_KIND_DIST = "dist"
VARIANT_KIND_E2E_AUXDIS = "e2e_auxdis"
VARIANT_KIND_E2E_UPFROZ = "e2e_upfroz"
VARIANT_KIND_E2E_AUXEN = "e2e_auxen"
VARIANT_KIND_ORDER = (
    VARIANT_KIND_DIST,
    VARIANT_KIND_E2E_AUXDIS,
    VARIANT_KIND_E2E_UPFROZ,
    VARIANT_KIND_E2E_AUXEN,
)
# Mapping from variant kind to plot marker style.
MARKER_BY_VARIANT_KIND: dict[str, str] = {
    VARIANT_KIND_DIST: "o",
    VARIANT_KIND_E2E_AUXDIS: "s",
    VARIANT_KIND_E2E_UPFROZ: "^",
    VARIANT_KIND_E2E_AUXEN: "v",
}
# Mapping from family identifier to human-readable label for plotting and logging.
FAMILY_LABELS: dict[str, str] = {
    "lam": "LAM (No upsampler)",
    "uplam": "CDBPN",
    "bicubiclam": "Bicubic",
    "srcnnlam": "SRCNN",
    "srcnnlam_variable": "Variable SRCNN",
    "imdnlam": "IMDN",
    "safmnlam": "SAMFN",
    "ganlam": "GAN",
    "ainnlam": "AINN",
}
# Mapping from family identifier to plot colour.
FAMILY_COLOURS: dict[str, str] = {
    "lam": "#5c5c5c",
    "bicubiclam": "#898e96",
    "uplam": "#ad5f55",
    "srcnnlam": "#c59647",
    "srcnnlam_variable": "#d08a32",
    "imdnlam": "#9d6387",
    "safmnlam": "#8a6a5a",
    "ganlam": "#5d82aa",
    "ainnlam": "#5f8f72",
}
# Mapping from family identifier to the canonical model name used in checkpoint naming.
FAMILY_MODEL_NAMES: dict[str, str] = {
    "lam": "LAM",
    "uplam": "UpLAM",
    "bicubiclam": "BicubicLAM",
    "srcnnlam": "SRCNNLAM",
    "srcnnlam_variable": "VariableSRCNNLAM",
    "imdnlam": "IMDNLAM",
    "safmnlam": "SAFMNLAM",
    "ganlam": "GANLAM",
    "ainnlam": "AINNLAM",
}
# Hardcoded checkpoint paths for all retained variants.
CHECKPOINT_LAM = "src/lam_min/checkpoints/LAM.pth"
CHECKPOINT_UPLAM = "src/lam_min/checkpoints/UpLAM.pth"
CHECKPOINT_UPLAM_E2E_AUXDIS = "src/lam_min/checkpoints/e2e/uplam_e2e_auxdis.pth"
CHECKPOINT_UPLAM_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/uplam_e2e_upfroz.pth"
CHECKPOINT_UPLAM_E2E_AUXEN = "src/lam_min/checkpoints/e2e/uplam_e2e_auxen.pth"
CHECKPOINT_BICUBIC_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/bicubiclam_e2e_upfroz.pth"
CHECKPOINT_SRCNN = "src/upsampler/srcnn/checkpoints/srcnn.pth"
CHECKPOINT_SRCNN_E2E_AUXDIS = "src/lam_min/checkpoints/e2e/srcnnlam_e2e_auxdis.pth"
CHECKPOINT_SRCNN_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/srcnnlam_e2e_upfroz.pth"
CHECKPOINT_SRCNN_E2E_AUXEN = "src/lam_min/checkpoints/e2e/srcnnlam_e2e_auxen.pth"
CHECKPOINT_VARIABLE_SRCNN = "src/upsampler/srcnn/checkpoints/srcnn_variable.pth"
CHECKPOINT_VARIABLE_SRCNN_E2E_AUXDIS = (
    "src/lam_min/checkpoints/e2e/srcnnlam_variable_e2e_auxdis.pth"
)
CHECKPOINT_VARIABLE_SRCNN_E2E_UPFROZ = (
    "src/lam_min/checkpoints/e2e/srcnnlam_variable_e2e_upfroz.pth"
)
CHECKPOINT_VARIABLE_SRCNN_E2E_AUXEN = "src/lam_min/checkpoints/e2e/srcnnlam_variable_e2e_auxen.pth"
CHECKPOINT_IMDN = "src/upsampler/imdn/checkpoints/imdn.pth"
CHECKPOINT_IMDN_E2E_AUXDIS = "src/lam_min/checkpoints/e2e/imdnlam_e2e_auxdis.pth"
CHECKPOINT_IMDN_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/imdnlam_e2e_upfroz.pth"
CHECKPOINT_IMDN_E2E_AUXEN = "src/lam_min/checkpoints/e2e/imdnlam_e2e_auxen.pth"
CHECKPOINT_SAFMN = "src/upsampler/safmn/checkpoints/safmn.pth"
CHECKPOINT_SAFMN_E2E_AUXDIS = "src/lam_min/checkpoints/e2e/safmnlam_e2e_auxdis.pth"
CHECKPOINT_SAFMN_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/safmnlam_e2e_upfroz.pth"
CHECKPOINT_SAFMN_E2E_AUXEN = "src/lam_min/checkpoints/e2e/safmnlam_e2e_auxen.pth"
CHECKPOINT_GAN = "src/upsampler/gan/checkpoints/gan.pth"
CHECKPOINT_GAN_E2E_AUXDIS = "src/lam_min/checkpoints/e2e/ganlam_e2e_auxdis.pth"
CHECKPOINT_GAN_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/ganlam_e2e_upfroz.pth"
CHECKPOINT_GAN_E2E_AUXEN = "src/lam_min/checkpoints/e2e/ganlam_e2e_auxen.pth"
CHECKPOINT_AINN = "src/upsampler/ainn/checkpoints/ainn.pth"
CHECKPOINT_AINN_E2E_AUXDIS = "src/lam_min/checkpoints/e2e/ainnlam_e2e_auxdis.pth"
CHECKPOINT_AINN_E2E_UPFROZ = "src/lam_min/checkpoints/e2e/ainnlam_e2e_upfroz.pth"
CHECKPOINT_AINN_E2E_AUXEN = "src/lam_min/checkpoints/e2e/ainnlam_e2e_auxen.pth"


@dataclass(frozen=True)
class RetainedVariantSpec:
    """Resolved metadata for one retained benchmark/inference variant."""

    variant_id: str
    family_id: str
    family_label: str
    variant_kind: str
    infer_model_name: str
    checkpoint: str
    lam_checkpoint: str | None
    colour: str

    @property
    def marker(self) -> str:
        """Return the plot marker for this variant."""
        return MARKER_BY_VARIANT_KIND[self.variant_kind]

    def required_paths(self) -> tuple[str, ...]:
        """Return checkpoint paths that must exist for this variant to run."""
        required = []
        if self.checkpoint:
            required.append(self.checkpoint)
        if self.lam_checkpoint:
            required.append(self.lam_checkpoint)
        return tuple(required)

    def missing_paths(self, repo_root: Path) -> tuple[Path, ...]:
        """
        Return missing required checkpoint paths resolved against the repo root.

        Parameters
        ----------
        repo_root : Path
            Repository root path to resolve variant checkpoint paths against.

        Returns
        -------
        tuple[Path, ...]
            Tuple of resolved checkpoint paths that are required for this variant but do not exist.

        """
        missing: list[Path] = []
        for raw_path in self.required_paths():
            path = resolve_variant_path(repo_root, raw_path)
            if not path.exists():
                missing.append(path)
        return tuple(missing)

    def is_available(self, repo_root: Path) -> bool:
        """
        Return whether all required checkpoint paths exist.

        Parameters
        ----------
        repo_root : Path
            Repository root path to resolve variant checkpoint paths against.

        Returns
        -------
        bool
            True if all required checkpoint paths exist and the variant is available to run,
            False if any required checkpoint is missing.
        """
        return not self.missing_paths(repo_root)


def resolve_variant_path(repo_root: Path, raw_path: str) -> Path:
    """
    Resolve a variant checkpoint path relative to the repository root.

    Absolute paths are returned as-is, while relative paths are resolved against the repo root.

    Parameters
    ----------
    repo_root : Path
        Repository root path to resolve relative variant checkpoint paths against.
    raw_path : str
        The raw checkpoint path specified in the variant registry, which may be absolute or
        relative.

    Returns
    -------
    Path
        The resolved checkpoint path as a Path object, which is guaranteed to be absolute.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root.joinpath(path)).resolve()


def _build_variant_specs() -> tuple[RetainedVariantSpec, ...]:
    """
    Build the static retained variant registry.

    Variants are manually curated based on available checkpoints and intended coverage of model
    families and training loss types.

    Returns
    -------
    tuple[RetainedVariantSpec, ...]
        Tuple of resolved retained variant specifications.
    """
    entries = [
        RetainedVariantSpec(
            variant_id="lam",
            family_id="lam",
            family_label=FAMILY_LABELS["lam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["lam"],
            checkpoint=CHECKPOINT_LAM,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["lam"],
        ),
        RetainedVariantSpec(
            variant_id="uplam_dist",
            family_id="uplam",
            family_label=FAMILY_LABELS["uplam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["uplam"],
            checkpoint=CHECKPOINT_UPLAM,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["uplam"],
        ),
        RetainedVariantSpec(
            variant_id="uplam_e2e_auxdis",
            family_id="uplam",
            family_label=FAMILY_LABELS["uplam"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["uplam"],
            checkpoint=CHECKPOINT_UPLAM_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["uplam"],
        ),
        RetainedVariantSpec(
            variant_id="uplam_e2e_upfroz",
            family_id="uplam",
            family_label=FAMILY_LABELS["uplam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["uplam"],
            checkpoint=CHECKPOINT_UPLAM_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["uplam"],
        ),
        RetainedVariantSpec(
            variant_id="uplam_e2e_auxen",
            family_id="uplam",
            family_label=FAMILY_LABELS["uplam"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["uplam"],
            checkpoint=CHECKPOINT_UPLAM_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["uplam"],
        ),
        RetainedVariantSpec(
            variant_id="bicubiclam_dist",
            family_id="bicubiclam",
            family_label=FAMILY_LABELS["bicubiclam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["bicubiclam"],
            checkpoint="",
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["bicubiclam"],
        ),
        RetainedVariantSpec(
            variant_id="bicubiclam_e2e_upfroz",
            family_id="bicubiclam",
            family_label=FAMILY_LABELS["bicubiclam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["bicubiclam"],
            checkpoint=CHECKPOINT_BICUBIC_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["bicubiclam"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_dist",
            family_id="srcnnlam",
            family_label=FAMILY_LABELS["srcnnlam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam"],
            checkpoint=CHECKPOINT_SRCNN,
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["srcnnlam"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_e2e_auxdis",
            family_id="srcnnlam",
            family_label=FAMILY_LABELS["srcnnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam"],
            checkpoint=CHECKPOINT_SRCNN_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["srcnnlam"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_e2e_upfroz",
            family_id="srcnnlam",
            family_label=FAMILY_LABELS["srcnnlam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam"],
            checkpoint=CHECKPOINT_SRCNN_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["srcnnlam"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_e2e_auxen",
            family_id="srcnnlam",
            family_label=FAMILY_LABELS["srcnnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam"],
            checkpoint=CHECKPOINT_SRCNN_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["srcnnlam"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_variable_dist",
            family_id="srcnnlam_variable",
            family_label=FAMILY_LABELS["srcnnlam_variable"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam_variable"],
            checkpoint=CHECKPOINT_VARIABLE_SRCNN,
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["srcnnlam_variable"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_variable_e2e_auxdis",
            family_id="srcnnlam_variable",
            family_label=FAMILY_LABELS["srcnnlam_variable"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam_variable"],
            checkpoint=CHECKPOINT_VARIABLE_SRCNN_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["srcnnlam_variable"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_variable_e2e_upfroz",
            family_id="srcnnlam_variable",
            family_label=FAMILY_LABELS["srcnnlam_variable"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam_variable"],
            checkpoint=CHECKPOINT_VARIABLE_SRCNN_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["srcnnlam_variable"],
        ),
        RetainedVariantSpec(
            variant_id="srcnnlam_variable_e2e_auxen",
            family_id="srcnnlam_variable",
            family_label=FAMILY_LABELS["srcnnlam_variable"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["srcnnlam_variable"],
            checkpoint=CHECKPOINT_VARIABLE_SRCNN_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["srcnnlam_variable"],
        ),
        RetainedVariantSpec(
            variant_id="imdnlam_dist",
            family_id="imdnlam",
            family_label=FAMILY_LABELS["imdnlam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["imdnlam"],
            checkpoint=CHECKPOINT_IMDN,
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["imdnlam"],
        ),
        RetainedVariantSpec(
            variant_id="imdnlam_e2e_auxdis",
            family_id="imdnlam",
            family_label=FAMILY_LABELS["imdnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["imdnlam"],
            checkpoint=CHECKPOINT_IMDN_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["imdnlam"],
        ),
        RetainedVariantSpec(
            variant_id="imdnlam_e2e_upfroz",
            family_id="imdnlam",
            family_label=FAMILY_LABELS["imdnlam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["imdnlam"],
            checkpoint=CHECKPOINT_IMDN_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["imdnlam"],
        ),
        RetainedVariantSpec(
            variant_id="imdnlam_e2e_auxen",
            family_id="imdnlam",
            family_label=FAMILY_LABELS["imdnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["imdnlam"],
            checkpoint=CHECKPOINT_IMDN_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["imdnlam"],
        ),
        RetainedVariantSpec(
            variant_id="safmnlam_dist",
            family_id="safmnlam",
            family_label=FAMILY_LABELS["safmnlam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["safmnlam"],
            checkpoint=CHECKPOINT_SAFMN,
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["safmnlam"],
        ),
        RetainedVariantSpec(
            variant_id="safmnlam_e2e_auxdis",
            family_id="safmnlam",
            family_label=FAMILY_LABELS["safmnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["safmnlam"],
            checkpoint=CHECKPOINT_SAFMN_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["safmnlam"],
        ),
        RetainedVariantSpec(
            variant_id="safmnlam_e2e_upfroz",
            family_id="safmnlam",
            family_label=FAMILY_LABELS["safmnlam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["safmnlam"],
            checkpoint=CHECKPOINT_SAFMN_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["safmnlam"],
        ),
        RetainedVariantSpec(
            variant_id="safmnlam_e2e_auxen",
            family_id="safmnlam",
            family_label=FAMILY_LABELS["safmnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["safmnlam"],
            checkpoint=CHECKPOINT_SAFMN_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["safmnlam"],
        ),
        RetainedVariantSpec(
            variant_id="ganlam_dist",
            family_id="ganlam",
            family_label=FAMILY_LABELS["ganlam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["ganlam"],
            checkpoint=CHECKPOINT_GAN,
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["ganlam"],
        ),
        RetainedVariantSpec(
            variant_id="ganlam_e2e_auxdis",
            family_id="ganlam",
            family_label=FAMILY_LABELS["ganlam"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["ganlam"],
            checkpoint=CHECKPOINT_GAN_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["ganlam"],
        ),
        RetainedVariantSpec(
            variant_id="ganlam_e2e_upfroz",
            family_id="ganlam",
            family_label=FAMILY_LABELS["ganlam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["ganlam"],
            checkpoint=CHECKPOINT_GAN_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["ganlam"],
        ),
        RetainedVariantSpec(
            variant_id="ganlam_e2e_auxen",
            family_id="ganlam",
            family_label=FAMILY_LABELS["ganlam"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["ganlam"],
            checkpoint=CHECKPOINT_GAN_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["ganlam"],
        ),
        RetainedVariantSpec(
            variant_id="ainnlam_dist",
            family_id="ainnlam",
            family_label=FAMILY_LABELS["ainnlam"],
            variant_kind=VARIANT_KIND_DIST,
            infer_model_name=FAMILY_MODEL_NAMES["ainnlam"],
            checkpoint=CHECKPOINT_AINN,
            lam_checkpoint=CHECKPOINT_LAM,
            colour=FAMILY_COLOURS["ainnlam"],
        ),
        RetainedVariantSpec(
            variant_id="ainnlam_e2e_auxdis",
            family_id="ainnlam",
            family_label=FAMILY_LABELS["ainnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXDIS,
            infer_model_name=FAMILY_MODEL_NAMES["ainnlam"],
            checkpoint=CHECKPOINT_AINN_E2E_AUXDIS,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["ainnlam"],
        ),
        RetainedVariantSpec(
            variant_id="ainnlam_e2e_upfroz",
            family_id="ainnlam",
            family_label=FAMILY_LABELS["ainnlam"],
            variant_kind=VARIANT_KIND_E2E_UPFROZ,
            infer_model_name=FAMILY_MODEL_NAMES["ainnlam"],
            checkpoint=CHECKPOINT_AINN_E2E_UPFROZ,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["ainnlam"],
        ),
        RetainedVariantSpec(
            variant_id="ainnlam_e2e_auxen",
            family_id="ainnlam",
            family_label=FAMILY_LABELS["ainnlam"],
            variant_kind=VARIANT_KIND_E2E_AUXEN,
            infer_model_name=FAMILY_MODEL_NAMES["ainnlam"],
            checkpoint=CHECKPOINT_AINN_E2E_AUXEN,
            lam_checkpoint=None,
            colour=FAMILY_COLOURS["ainnlam"],
        ),
    ]
    return tuple(entries)


# Build the retained variant registry and related lookup structures.
RETAINED_VARIANTS = _build_variant_specs()
VARIANT_BY_ID: dict[str, RetainedVariantSpec] = {
    variant.variant_id: variant for variant in RETAINED_VARIANTS
}
FAMILY_SELECTOR_IDS = tuple(FAMILY_LABELS.keys())
FAMILY_TO_VARIANT_IDS: dict[str, tuple[str, ...]] = {
    "lam": ("lam",),
    "uplam": (
        "uplam_dist",
        "uplam_e2e_auxdis",
        "uplam_e2e_upfroz",
    ),
    "bicubiclam": (
        "bicubiclam_dist",
        "bicubiclam_e2e_upfroz",
    ),
    "srcnnlam": (
        "srcnnlam_dist",
        "srcnnlam_e2e_auxdis",
        "srcnnlam_e2e_upfroz",
    ),
    "srcnnlam_variable": (
        "srcnnlam_variable_dist",
        "srcnnlam_variable_e2e_auxdis",
        "srcnnlam_variable_e2e_upfroz",
    ),
    "imdnlam": (
        "imdnlam_dist",
        "imdnlam_e2e_auxdis",
        "imdnlam_e2e_upfroz",
    ),
    "safmnlam": (
        "safmnlam_dist",
        "safmnlam_e2e_auxdis",
        "safmnlam_e2e_upfroz",
    ),
    "ganlam": (
        "ganlam_dist",
        "ganlam_e2e_auxdis",
        "ganlam_e2e_upfroz",
    ),
    "ainnlam": (
        "ainnlam_dist",
        "ainnlam_e2e_auxdis",
        "ainnlam_e2e_upfroz",
    ),
}
MODEL_NAME_TO_FAMILY_ID: dict[str, str] = {
    value: key for key, value in FAMILY_MODEL_NAMES.items() if key != "uplam"
}
MODEL_NAME_TO_FAMILY_ID["UpLAM"] = "uplam"


def supported_variant_ids() -> tuple[str, ...]:
    """Return all supported exact variant identifiers."""
    return tuple(variant.variant_id for variant in RETAINED_VARIANTS)


def supported_selector_ids() -> tuple[str, ...]:
    """Return all supported family and exact selector identifiers."""
    ordered = dict.fromkeys((*FAMILY_SELECTOR_IDS, *supported_variant_ids()))
    return tuple(ordered)


def resolve_exact_variant(variant_id: str) -> RetainedVariantSpec:
    """
    Resolve one exact retained variant identifier.

    Parameters
    ----------
    variant_id : str
        The exact variant identifier to resolve, which must be one of the supported exact variant
        IDs defined in the retained variant registry.

    Returns
    -------
    RetainedVariantSpec
        The resolved retained variant specification corresponding to the provided exact variant ID.
    """
    try:
        return VARIANT_BY_ID[variant_id]
    except KeyError as exc:
        supported = ", ".join(supported_variant_ids())
        raise ValueError(
            f"Unsupported model_variant '{variant_id}'. Supported values: {supported}"
        ) from exc


def is_family_selector(selector: str) -> bool:
    """
    Return whether a selector refers to a family expansion token.

    Parameters
    ----------
    selector : str
        The selector to check, which may be an exact variant ID or a family selector token.

    Returns
    -------
    bool
        True if the selector is a family selector token that can be expanded to multiple variants,
        False if it is not a family selector (and should be treated as an exact variant ID).
    """
    return selector in FAMILY_TO_VARIANT_IDS


def expand_target_selectors(
    selectors: list[str],
    repo_root: Path,
    logger: logging.Logger | None = None,
) -> list[RetainedVariantSpec]:
    """
    Expand mixed family and exact selectors into runnable variants.

    Parameters
    ----------
    selectors : list[str]
        List of user-provided selectors, which can be exact variant IDs or family selector tokens.
    repo_root : Path
        Repository root path to resolve variant checkpoint paths against.
    logger : logging.Logger | None, optional
        Optional logger for warnings about missing checkpoints. If None, the module logger is used.

    Returns
    -------
    list[RetainedVariantSpec]
        List of resolved retained variant specifications that are runnable with available
        checkpoints.
    """
    resolved: list[RetainedVariantSpec] = []
    seen_variant_ids: set[str] = set()
    active_logger = logger or logging.getLogger(__name__)

    for raw_selector in selectors:
        selector = str(raw_selector).strip().lower()
        if not selector:
            continue

        if is_family_selector(selector):
            for variant_id in FAMILY_TO_VARIANT_IDS[selector]:
                variant = resolve_exact_variant(variant_id)
                if variant.variant_id in seen_variant_ids:
                    continue
                missing_paths = variant.missing_paths(repo_root)
                if missing_paths:
                    active_logger.warning(
                        "Skipping unavailable family variant '%s' because checkpoint(s) are "
                        "missing: %s",
                        variant.variant_id,
                        ", ".join(str(path) for path in missing_paths),
                    )
                    continue
                resolved.append(variant)
                seen_variant_ids.add(variant.variant_id)
            continue

        variant = resolve_exact_variant(selector)
        if variant.variant_id in seen_variant_ids:
            continue
        missing_paths = variant.missing_paths(repo_root)
        if missing_paths:
            raise FileNotFoundError(
                f"Selected variant '{variant.variant_id}' is missing checkpoint(s): "
                + ", ".join(str(path) for path in missing_paths)
            )
        resolved.append(variant)
        seen_variant_ids.add(variant.variant_id)

    if not resolved:
        raise ValueError("No runnable benchmark targets were resolved from the provided selectors.")
    return resolved


def family_id_from_model_name(model_name: str) -> str:
    """
    Map a wrapper class name to its canonical family identifier.

    Parameters
    ----------
    model_name : str
        The model name to map, which should correspond to the canonical model names defined in
        FAMILY_MODEL_NAMES.

    Returns
    -------
    str
        The family identifier corresponding to the provided model name.
    """
    try:
        return MODEL_NAME_TO_FAMILY_ID[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_NAME_TO_FAMILY_ID))
        raise ValueError(
            f"Unsupported model name '{model_name}' for retained checkpoint naming. "
            f"Supported values: {supported}"
        ) from exc


def resolve_e2e_variant_kind(
    model_name: str,
    *,
    aux_enabled: bool,
    freeze_upsampler: bool = False,
) -> str:
    """
    Resolve the canonical end-to-end retained variant kind for a training run.

    Parameters
    ----------
    model_name : str
        Wrapper model name.
    aux_enabled : bool
        Whether auxiliary reconstruction loss is enabled.
    freeze_upsampler : bool, optional
        Whether the upsampler branch is frozen and only the LAM branch is trained.

    Returns
    -------
    str
        Canonical retained variant kind for the requested training mode.
    """
    family_id = family_id_from_model_name(model_name)
    if freeze_upsampler:
        return VARIANT_KIND_E2E_UPFROZ
    if family_id == "bicubiclam":
        return VARIANT_KIND_E2E_UPFROZ
    return VARIANT_KIND_E2E_AUXEN if aux_enabled else VARIANT_KIND_E2E_AUXDIS


def canonical_e2e_checkpoint_prefix(
    model_name: str,
    aux_enabled: bool,
    *,
    freeze_upsampler: bool = False,
) -> str:
    """
    Derive the canonical end-to-end checkpoint prefix from model name and aux setting.

    Parameters
    ----------
    model_name : str
        The model name to derive the family identifier from, which should correspond to the
        canonical model names defined in FAMILY_MODEL_NAMES.
    aux_enabled : bool
        Whether the auxiliary loss was enabled during end-to-end training.
    freeze_upsampler : bool, optional
        Whether the upsampler branch is frozen and only the LAM branch is trained.

    Returns
    -------
    str
        The canonical checkpoint prefix for the end-to-end trained variant of the given model and
        aux setting, in the format "{family_id}_{variant_kind}".
    """
    family_id = family_id_from_model_name(model_name)
    suffix = resolve_e2e_variant_kind(
        model_name,
        aux_enabled=aux_enabled,
        freeze_upsampler=freeze_upsampler,
    )
    return f"{family_id}_{suffix}"
