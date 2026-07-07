# ruff: noqa: D100,D101,D102,D103,PLR0402,PLR0913,PLR0915,PLR2004,B905,E741,S101,E501
# Change: keep the supervisor-provided script shape and exact-style names where possible.
import argparse
import collections.abc as abc
import math
import os
from pathlib import Path  # Change: needed to resolve this repository's YAML paths.

import astropy.coordinates as coord
import astropy.units as u
import librosa
import matplotlib
import yaml  # Change: use only the repository YAML config, not JSON.

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as tri

try:
    import mpl_toolkits.basemap as basemap
except ModuleNotFoundError:
    import basemap_compat as basemap

import numpy as np
import scipy.constants as constants
import scipy.linalg as linalg
import skimage.util as skutil
import torch
from scipy.signal import windows
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader

from infer import (  # Change: reuse the project's YAML dataset and retained-variant resolution.
    _resolve_inference_variant,
    _resolve_locata_tasks,
    _select_dataset_wavs,
    _select_locata_file_ids,
)
from lam_min.util.utils import (  # Change: reuse the project's retained checkpoint loaders.
    load_ainn_lam_state,
    load_bicubic_lam_state,
    load_gan_lam_state,
    load_imdn_lam_state,
    load_safmn_lam_state,
    load_srcnn_lam_state,
    resolve_safmn_inference_architecture,
)
from utils.utils import (  # Change: match the preprocessing used by src/infer.py.
    prepare_audio_for_inference,
    resolve_requested_device,
    seed_everything,
    visibility_t_sti_seconds,
)

# =============================================================================
# CONSTANTS
# =============================================================================

_EIGENMIKE_ = {
    "1": [69, 0, 0.042],
    "2": [90, 32, 0.042],
    "3": [111, 0, 0.042],
    "4": [90, 328, 0.042],
    "5": [32, 0, 0.042],
    "6": [55, 45, 0.042],
    "7": [90, 69, 0.042],
    "8": [125, 45, 0.042],
    "9": [148, 0, 0.042],
    "10": [125, 315, 0.042],
    "11": [90, 291, 0.042],
    "12": [55, 315, 0.042],
    "13": [21, 91, 0.042],
    "14": [58, 90, 0.042],
    "15": [121, 90, 0.042],
    "16": [159, 89, 0.042],
    "17": [69, 180, 0.042],
    "18": [90, 212, 0.042],
    "19": [111, 180, 0.042],
    "20": [90, 148, 0.042],
    "21": [32, 180, 0.042],
    "22": [55, 225, 0.042],
    "23": [90, 249, 0.042],
    "24": [125, 225, 0.042],
    "25": [148, 180, 0.042],
    "26": [125, 135, 0.042],
    "27": [90, 111, 0.042],
    "28": [55, 135, 0.042],
    "29": [21, 269, 0.042],
    "30": [58, 270, 0.042],
    "31": [122, 270, 0.042],
    "32": [159, 271, 0.042],
}

# =============================================================================
# UTILITIES & MATH
# =============================================================================


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(checkpoint_path)
    assert ext in (".pth", ".tar"), "Only .pth and .tar checkpoints are supported."
    ckpt = torch.load(checkpoint_path, map_location=device)
    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return ckpt
    print(f"Loading {checkpoint_path}, epoch = {ckpt['epoch']}.")
    return ckpt["model"]


def load_yaml_config(config_path):
    """
    Load and resolve the repository inference YAML.

    Parameters
    ----------
    config_path : str
        Path to the repository YAML inference config.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Resolved inference and dataset configuration dictionaries.
    """
    # Change: the provided script loaded a standalone config; this repository uses YAML only.
    repo_root = Path(__file__).resolve().parents[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    inference_config = dict(config["inference"])
    dataset_config = dict(config["dataset"])
    _resolve_inference_variant(inference_config, repo_root)
    return inference_config, dataset_config


def is_scalar(x):
    return not isinstance(x, abc.Container)


def _polar2cart(coords_dict, units=None):
    if units not in ("degrees", "radians"):
        raise ValueError("units must be 'degrees' or 'radians'")
    coords = {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        if units == "degrees"
        else list(c)
        for m, c in coords_dict.items()
    }
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords.items()
    }


def get_xyz():
    mic_coords = _polar2cart(_EIGENMIKE_, units="degrees")
    return [list(c) for c in mic_coords.values()]


# --- Coordinate conversions ---


def eq2cart(r, lat, lon):
    r = np.atleast_1d(np.array(r, copy=False) if not is_scalar(r) else np.array([r]))
    if np.any(r < 0):
        raise ValueError("r must be non-negative.")
    return (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )


def pol2cart(r, colat, lon):
    return eq2cart(r, (np.pi / 2) - colat, lon)


def cart2pol(x, y, z):
    sph = coord.SphericalRepresentation.from_cartesian(coord.CartesianRepresentation(x, y, z))
    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)
    return r, colat, lon


def cart2eq(x, y, z):
    r, colat, lon = cart2pol(x, y, z)
    return r, (np.pi / 2) - colat, lon


def wrapped_rad2deg(lat_r, lon_r):
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


# --- Spatial sampling & steering ---


def fibonacci(N, direction=None, FoV=None, shift_lon=0, shift_colat=0):
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)
        if FoV is None or not (0 < np.rad2deg(FoV) < 360):
            raise ValueError("FoV must be in (0, 360) degrees when direction is given.")

    if N < 0:
        raise ValueError("N must be non-negative.")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)
    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5)) + shift_lon
    XYZ = np.stack(pol2cart(1, colat, lon), axis=0)

    if direction is not None:
        XYZ = XYZ[:, (direction @ XYZ) >= np.cos(FoV / 2)]

    return XYZ


def get_field(shift_lon=0, shift_colat=0):
    R = fibonacci(10, shift_lon=shift_lon, shift_colat=shift_colat)
    return R[:, np.abs(R[2, :]) < np.sin(np.deg2rad(90))]


def steering_operator(XYZ=None, R=None):
    if XYZ is None:
        XYZ = np.array(get_xyz()).T
    if R is None:
        R = get_field()
    wl = constants.speed_of_sound / (
        skutil.view_as_windows(np.linspace(1500, 4500, 10), (2,), 1).mean(axis=-1).max()
        + 500
    )
    return np.exp((-1j * 2 * np.pi / wl * XYZ.T) @ R)


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================


def extract_visibilities(data, rate, T, fc, bw, alpha):
    N_stft = int(rate * T)
    if N_stft == 0:
        raise ValueError("Not enough samples per time frame.")
    N_ch = data.shape[1]
    N_sample = (data.shape[0] // N_stft) * N_stft
    stf_data = skutil.view_as_blocks(data[:N_sample], (N_stft, N_ch)).squeeze(axis=1)

    window = windows.tukey(M=N_stft, alpha=alpha, sym=True).reshape(1, -1, 1)
    stft = np.fft.fft(stf_data * window, axis=1)
    idx_start = int((fc - 0.5 * bw) * N_stft / rate)
    idx_end = int((fc + 0.5 * bw) * N_stft / rate)
    spec = stft[:, idx_start : idx_end + 1, :].sum(axis=1)

    return spec.reshape(-1, N_ch, 1).conj() * spec.reshape(-1, 1, N_ch)


def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    S_sti = extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0)
    N_ch = data.shape[1]
    N_blk = int(T_stationarity / T_sti)
    return (
        skutil.view_as_windows(S_sti, (N_blk, N_ch, N_ch), (N_blk, N_ch, N_ch))
        .squeeze(axis=(1, 2))
        .sum(axis=1)
    )


def get_visibility_matrix(audio_in, fs, T_sti=10e-3, scale="linear", nbands=9):
    if scale == "linear":
        freq = skutil.view_as_windows(np.linspace(1500, 4500, nbands), (2,), 1).mean(axis=-1)
    elif scale == "log":
        freq = librosa.mel_frequencies(n_mels=nbands, fmin=50, fmax=4500)
    else:
        raise ValueError("scale must be 'linear' or 'log'")
    bw = 50.0

    N_px = steering_operator(np.array(get_xyz()).T, get_field()).shape[1]
    visibilities = []

    for i in range(nbands - 1):
        S = form_visibility(audio_in, fs, freq[i], bw, T_sti, 10 * T_sti)
        frames = []
        for s in S:
            S_D, S_V = linalg.eigh(s)
            S_D = np.clip(S_D / S_D.max(), 0, None) if S_D.max() > 0 else np.zeros_like(S_D)
            frames.append((S_V * S_D) @ S_V.conj().T)
        visibilities.append(frames)

    n_frames = len(visibilities[0]) if visibilities else 0
    return np.array(visibilities), np.zeros((nbands - 1, n_frames, N_px))


# =============================================================================
# DATASET
# =============================================================================


def build_dataset(inference_config, dataset_config):
    """
    Build the dataset selected by the repository YAML config.

    Parameters
    ----------
    inference_config : dict[str, Any]
        Resolved inference configuration.
    dataset_config : dict[str, Any]
        Dataset path configuration.

    Returns
    -------
    torch.utils.data.Dataset
        STARSS23 or LOCATA dataset instance selected by the YAML config.
    """
    # Change: replace the standalone directory dataset with this repository's YAML datasets.
    selection_seed = int(inference_config.get("file_selection_seed", 0) or 0)
    generator = seed_everything(selection_seed)

    if inference_config["data_set"] == "starss23":
        from data.starss_loader import StarssAudioDataset

        dataset = StarssAudioDataset(
            audio_path=Path(dataset_config["data_audio_path"]),
            ground_truth_path=Path(dataset_config["data_ground_truth_path"]),
            load_ground_truth=False,
            frame_width_ms=inference_config["frame_width_ms"],
        )
        dataset.wavs = _select_dataset_wavs(dataset.wavs, inference_config, generator)
        return dataset

    if inference_config["data_set"] == "locata":
        from data.locata_loader import LocataAudioDataset

        dataset = LocataAudioDataset(
            path=Path(dataset_config["data_audio_path"]),
            load_ground_truth=False,
            frame_width_ms=inference_config["frame_width_ms"],
            tasks=_resolve_locata_tasks(inference_config),
        )
        selected_file_ids = _select_locata_file_ids(
            [entry.file_id for entry in dataset.entries],
            inference_config,
            generator,
        )
        entry_by_id = {entry.file_id: entry for entry in dataset.entries}
        dataset.entries = [entry_by_id[file_id] for file_id in selected_file_ids]
        dataset.relevant_dir = [entry.eigenmike_dir for entry in dataset.entries]
        dataset.wavs = [entry.wav_path for entry in dataset.entries]
        return dataset

    raise ValueError(f"Unsupported dataset: {inference_config['data_set']}")


def prepare_batch_audio(batch, inference_config):
    """
    Prepare one YAML dataset batch for visualisation.

    Parameters
    ----------
    batch : dict[str, Any]
        Batch emitted by the repository dataset.
    inference_config : dict[str, Any]
        Resolved inference configuration.

    Returns
    -------
    tuple[str, np.ndarray, int]
        File identifier, prepared time-channel audio, and sample rate.
    """
    # Change: repository datasets return dictionaries, not ``(audio, name)`` tuples.
    name = batch["file_id"][0]
    audio = batch["audio"].cpu().numpy()[0].astype(np.float32)
    sample_rate = int(batch["sample_rate"][0])
    audio, _, audio_prep = prepare_audio_for_inference(audio, sample_rate, inference_config)
    return name, audio, int(audio_prep["target_sample_rate"])


# =============================================================================
# MODEL
# =============================================================================


def _init_scaled_kaiming(layer, scale=1e-6):
    nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data.fill_(1e-6)


class LAM(nn.Module):
    def __init__(self, num_bands=16, Nch=32, tau=None, D=None):
        super().__init__()
        self.num_bands = num_bands
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]

        if tau is None or D is None:
            self.tau = nn.Parameter(torch.empty((num_bands, Npx), dtype=torch.float64))
            self.D = nn.Parameter(torch.empty((num_bands, Nch, Npx), dtype=torch.complex128))
            self.tau.data.normal_(0, 1e-7)
            self.D.data.normal_(0, 1e-5)
        else:
            self.tau = nn.Parameter(tau)
            self.D = nn.Parameter(D)

        self.retanh = nn.ReLU()
        conv_kwargs = dict(dtype=torch.float64, padding="same")
        self.denoise1 = nn.Conv1d(num_bands, num_bands, kernel_size=3, **conv_kwargs)
        self.denoise2 = nn.Conv1d(num_bands, num_bands, kernel_size=5, **conv_kwargs)
        self.denoise3 = nn.Conv1d(num_bands, num_bands, kernel_size=7, **conv_kwargs)
        self.denoise4 = nn.Conv1d(num_bands, num_bands, kernel_size=9, **conv_kwargs)
        for layer in (self.denoise1, self.denoise2, self.denoise3, self.denoise4):
            _init_scaled_kaiming(layer)

    def forward(self, S):
        self.A = self.A.to(S.device)
        batch_size, freq_bands = S.shape[:2]

        latent_list = []
        for i in range(freq_bands):
            Ds, Vs = torch.linalg.eigh(S[:, i])
            Vs = Vs * torch.sqrt(torch.where(Ds > 0, Ds, torch.zeros_like(Ds))).unsqueeze(1)
            x = torch.linalg.norm(torch.matmul(self.D[i].conj().T, Vs), dim=2) ** 2 - self.tau[i]
            latent_list.append(x)

        latent_x = torch.stack(latent_list, dim=1)
        skip = latent_x.clone()

        for denoise in (self.denoise1, self.denoise2, self.denoise3, self.denoise4):
            latent_x = self.retanh(denoise(latent_x) + skip)

        A = self.A.unsqueeze(0)
        out = torch.stack(
            [
                torch.einsum(
                    "nij,bjk,nkl->bil",
                    A,
                    torch.diag_embed(latent_x[:, i].cdouble()),
                    A.transpose(1, 2).conj(),
                )
                for i in range(latent_x.shape[1])
            ],
            dim=1,
        )

        return out, latent_x


def initialize_model(inference_config, device):
    """
    Initialise the model selected by the repository YAML config.

    Parameters
    ----------
    inference_config : dict[str, Any]
        Resolved inference configuration.
    device : torch.device
        Device used for checkpoint loading and inference.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """
    # Change: YAML selects retained variants rather than the standalone ``model`` block.
    model_name = inference_config["model_name"]
    ckpt_path = inference_config["model_checkpoint"]

    if model_name == "LAM":
        model = LAM(num_bands=9)
        model.load_state_dict(load_checkpoint(ckpt_path, device))
    elif model_name == "UpLAM":
        from lam_min.model.UpLAM import UpLAM

        model = UpLAM(num_bands=9)
        model.load_state_dict(load_checkpoint(ckpt_path, device))
    elif model_name == "BicubicLAM":
        from lam_min.model.BicubicLAM import BicubicLAM

        model = BicubicLAM(num_bands=9, in_channels=4, out_channels=32)
        load_bicubic_lam_state(model, ckpt_path, device, inference_config.get("lam_checkpoint"))
    elif model_name == "SRCNNLAM":
        from lam_min.model.SRCNNLAM import SRCNNLAM

        model = SRCNNLAM(num_bands=9, in_channels=4, out_channels=32)
        load_srcnn_lam_state(model, ckpt_path, device, inference_config.get("lam_checkpoint"))
    elif model_name == "IMDNLAM":
        from lam_min.model.IMDNLAM import IMDNLAM

        model = IMDNLAM(num_bands=9, in_channels=4, out_channels=32)
        load_imdn_lam_state(model, ckpt_path, device, inference_config.get("lam_checkpoint"))
    elif model_name == "SAFMNLAM":
        from lam_min.model.SAFMNLAM import SAFMNLAM

        safmn_arch, _ = resolve_safmn_inference_architecture(inference_config, ckpt_path, device)
        model = SAFMNLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            feature_channels=int(safmn_arch["feature_channels"]),
            n_blocks=int(safmn_arch["n_blocks"]),
            ffn_scale=float(safmn_arch["ffn_scale"]),
            n_levels=int(safmn_arch["n_levels"]),
        )
        load_safmn_lam_state(model, ckpt_path, device, inference_config.get("lam_checkpoint"))
    elif model_name == "GANLAM":
        from lam_min.model.GANLAM import GANLAM

        model = GANLAM(num_bands=9, in_channels=4, out_channels=32, feature_channels=128)
        load_gan_lam_state(model, ckpt_path, device, inference_config.get("lam_checkpoint"))
    elif model_name == "AINNLAM":
        from lam_min.model.AINNLAM import AINNLAM

        model = AINNLAM(
            num_bands=9,
            in_channels=4,
            out_channels=32,
            low_channel_indices=tuple(int(i) for i in inference_config["locata_low_channel_indices"]),
        )
        load_ainn_lam_state(model, ckpt_path, device, inference_config.get("lam_checkpoint"))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device).eval()


# =============================================================================
# VISUALIZATION
# =============================================================================


def cmap_from_list(name, colors, N=256, gamma=1.0):
    if not isinstance(colors, abc.Iterable):
        raise ValueError("colors must be iterable")

    if (
        isinstance(colors[0], abc.Sized)
        and len(colors[0]) == 2
        and not isinstance(colors[0], str)
    ):
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = {k: [] for k in ("red", "green", "blue", "alpha")}
    for val, color in zip(vals, colors):
        r, g, b, a = mcolors.to_rgba(color)
        for ch, v in zip(("red", "green", "blue", "alpha"), (r, g, b, a)):
            cdict[ch].append((val, v, v))

    return mcolors.LinearSegmentedColormap(name, cdict, N, gamma)


def draw_map(
    I,
    R,
    lon_ticks,
    catalog=None,
    show_labels=False,
    show_axis=False,
    fig=None,
    ax=None,
    kmeans=False,
    gaussian_mixture=False,
):
    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([R_el.min(), R_el.max()])
    R_az_min, R_az_max = np.around([R_az.min(), R_az.max()])

    if ax is None:
        fig, ax = plt.subplots()

    bm = basemap.Basemap(
        projection="mill",
        llcrnrlat=R_el_min,
        urcrnrlat=R_el_max,
        llcrnrlon=R_az_min,
        urcrnrlon=R_az_max,
        resolution="c",
        ax=ax,
    )
    bm_labels = [1, 0, 0, 1] if show_axis else [0, 0, 0, 0]
    bm.drawparallels(
        np.linspace(R_el_min, R_el_max, 5),
        color="w",
        dashes=[1, 0],
        labels=bm_labels,
        labelstyle="+/-",
        textcolor="#565656",
        zorder=0,
        linewidth=2,
    )
    bm.drawmeridians(
        lon_ticks,
        color="w",
        dashes=[1, 0],
        labels=bm_labels,
        labelstyle="+/-",
        textcolor="#565656",
        zorder=0,
        linewidth=2,
    )

    if show_labels:
        ax.set_xlabel("Azimuth (degrees)", labelpad=20)
        ax.set_ylabel("Elevation (degrees)", labelpad=40)

    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)
    N_px = I.shape[1]
    mycmap = cmap_from_list("mycmap", I.T, N=N_px)
    ax.tripcolor(
        triangulation,
        np.arange(N_px),
        cmap=mycmap,
        shading="gouraud",
        alpha=0.9,
        edgecolors="w",
        linewidth=0.1,
    )

    cluster_center = None
    if kmeans:
        Npts = 18
        max_idx = np.square(I).sum(axis=0).argsort()[-Npts:][::-1]
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        clusters = KMeans(n_clusters=3).fit(x_y).cluster_centers_
        ax.scatter(R_x[max_idx], R_y[max_idx], c="b", s=5)
        ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)
        cluster_center = bm(clusters[0, 0], clusters[0, 1], inverse=True)

    return fig, ax, cluster_center


# =============================================================================
# INFERENCE
# =============================================================================


def main():
    parser = argparse.ArgumentParser("LAM: acoustic map visualisation")
    parser.add_argument("-C", "--config", type=str, default="config/inference_config.yaml")
    parser.add_argument("-D", "--device", default="cpu", type=str)
    parser.add_argument(
        "-A",
        "--alpha-ema",
        type=float,
        default=None,
        help="Smoothing factor for the exponential running average (0 to 1)",
    )
    args = parser.parse_args()

    inference_config, dataset_config = load_yaml_config(args.config)  # Change: YAML only.
    output_dir = Path(inference_config["output_path"]) / "visualisations" / inference_config["model_variant"]
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = inference_config["model_checkpoint"]
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}.")

    device = resolve_requested_device(args.device)  # Change: support cpu/mps/cuda names from the repo.

    # Resolve alpha_ema: argparse > config file > default (0.1)
    alpha_ema = (
        args.alpha_ema if args.alpha_ema is not None else inference_config.get("alpha_ema", 0.1)
    )

    dataset = build_dataset(inference_config, dataset_config)  # Change: YAML dataset selection.
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)

    model = initialize_model(inference_config, device)  # Change: YAML retained-variant model.

    R_field = get_field()
    lon_ticks = np.linspace(-180, 180, 5)
    T_sti_ms = int(inference_config["frame_width_ms"])
    T_sti = visibility_t_sti_seconds(float(T_sti_ms))  # Change: keep CSM frames aligned to YAML.

    with torch.no_grad():
        for batch in dataloader:
            name, audio, fs = prepare_batch_audio(batch, inference_config)  # Change: repo batch format.

            S_in, _ = get_visibility_matrix(audio, fs=fs, T_sti=T_sti, nbands=10)
            print(f"{name}: S_in shape = {S_in.shape} (bands, frames, N_ch, N_ch)")
            S_in = torch.from_numpy(S_in).to(device).permute(1, 0, 2, 3)

            if inference_config["model_name"] == "LAM":  # Change: local pasted LAM has no collect_metrics.
                _, I_pred = model(S_in)
            else:
                result = model(S_in, collect_metrics=False)  # Change: repo models accept collect_metrics.
                _, I_pred = result[:2] if isinstance(result, tuple) else (None, result)
            I_pred_np = I_pred.cpu().numpy()
            n_frames, n_bands, N_px = I_pred_np.shape
            print(f"{name}: {n_frames} frames × {n_bands} bands × {N_px} pixels")

            clip_dir = os.path.join(output_dir, name)
            os.makedirs(clip_dir, exist_ok=True)

            # Initialize tracking parameters for the exponential running average
            running_max = np.zeros(n_bands)

            for i, frame_bands in enumerate(I_pred_np):
                t_ms = i * T_sti_ms

                for b, band in enumerate(frame_bands):
                    current_max = band.max()

                    # Update the EMA for the current band
                    if i == 0:
                        running_max[b] = current_max
                    else:
                        running_max[b] = alpha_ema * current_max + (1 - alpha_ema) * running_max[b]

                    # Normalize against the running average
                    norm_val = running_max[b] if running_max[b] > 1e-10 else 1.0

                    # Clip to [0, 1] to keep formatting safe for plotting/colormaps
                    # in case a sudden peak drastically exceeds the running max
                    band_normalized = np.clip(band / norm_val, 0, 1)

                    band_rgb = np.tile(band_normalized[np.newaxis], (3, 1))

                    band_dir = os.path.join(clip_dir, "bands", f"band{b:02d}")
                    os.makedirs(band_dir, exist_ok=True)

                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    draw_map(
                        band_rgb,
                        R_field,
                        lon_ticks,
                        show_labels=True,
                        show_axis=True,
                        fig=fig,
                        ax=ax,
                    )
                    ax.set_title(f"{name}  —  t = {t_ms} ms  |  band {b}")
                    fig.savefig(
                        os.path.join(band_dir, f"frame_{i:04d}_{t_ms:06d}ms_band{b:02d}.png"),
                        bbox_inches="tight",
                        dpi=100,
                    )
                    plt.close(fig)

            print(f"  → saved {n_frames} frames [per-band ({n_bands} bands/frame)] to {clip_dir}")


if __name__ == "__main__":
    main()
