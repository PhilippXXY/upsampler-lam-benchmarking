"""
Latent Acoustic Map (LAM) model.

This module implements the Latent Acoustic Map network, which performs
Direction of Arrival (DoA) estimation by projecting covariance matrices
into a learned latent space, denoising via cascaded convolutions, and
reconstructing spatial acoustic maps using a steering operator.

The LAM architecture consists of:
    - Back-projection encoder mapping covariance matrices to latent intensity maps
    - Multi-scale 1D convolutional denoising layers with residual connections
    - Forward projection decoder using microphone array steering vectors

References
----------
.. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
"""

import time
import tracemalloc

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from lam_min.trainer.utils import steering_operator
from lam_min.util.flops import build_custom_flop_mapping


def initialize_scaled_kaiming(layer, scale=1e-6):
    """
    Initialise layer weights with scaled Kaiming uniform initialisation.

    Applies He/Kaiming uniform initialisation to layer weights and scales
    them by a small factor to improve numerical stability when working with
    small covariance matrix values. Biases are initialised to a small constant.

    Parameters
    ----------
    layer : torch.nn.Module
        Neural network layer with weight and optional bias parameters
    scale : float, optional
        Scaling factor applied to initialised weights (default: 1e-6)
    """
    torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data.fill_(1e-6)  # Small bias to avoid dead neurons


class LAM(torch.nn.Module):
    """
    Latent Acoustic Map neural network.

    A deep learning model that estimates Direction of Arrival (DoA) from
    multi-frequency covariance matrices using learned dictionary back-projection,
    multi-scale convolutional denoising, and steering vector reconstruction.

    The model processes each frequency band independently through an encoder-decoder
    architecture with shared denoising layers across all bands.

    Parameters
    ----------
    num_bands : int, optional
        Number of frequency bands to process (default: 16)
    Nch : int, optional
        Number of microphone array channels (default: 32)
    tau : torch.Tensor, optional
        Bias parameters of shape (num_bands, Npx) for thresholding latent space.
        If None, parameters are randomly initialised.
    D : torch.Tensor, optional
        Dictionary matrix of shape (num_bands, Nch, Npx) for back-projection.
        If None, parameters are randomly initialised.

    Attributes
    ----------
    A : torch.Tensor
        Steering operator matrix relating spatial pixels to microphone measurements
    tau : torch.nn.Parameter
        Learnable threshold bias for each frequency band and spatial pixel
    D : torch.nn.Parameter
        Learnable dictionary for projecting microphone signals to latent space
    retanh : torch.nn.ReLU
        ReLU activation function (misnomer from original implementation)
    denoise1, denoise2, denoise3, denoise4 : torch.nn.Conv1d
        Multi-scale 1D convolutional layers for latent space denoising

    References
    ----------
    .. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
    """

    def __init__(self, num_bands=16, Nch=32, tau=None, D=None):
        """
        Initialise the LAM model.

        Parameters
        ----------
        num_bands : int, optional
            Number of frequency bands (default: 16)
        Nch : int, optional
            Number of microphone channels (default: 32)
        tau : torch.Tensor, optional
            Pre-initialised bias parameters (default: None)
        D : torch.Tensor, optional
            Pre-initialised dictionary matrix (default: None)
        """
        super(LAM, self).__init__()
        self.num_bands = num_bands
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((self.num_bands, Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(
                torch.empty((self.num_bands, Nch, Npx), dtype=torch.complex128)
            )
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.retanh = nn.ReLU()

        # Convolution layers modified to work across all frequency bands
        self.denoise1 = torch.nn.Conv1d(
            num_bands, num_bands, kernel_size=3, padding=1, dtype=torch.float64
        )
        self.denoise2 = torch.nn.Conv1d(
            num_bands, num_bands, kernel_size=5, padding=2, dtype=torch.float64
        )
        self.denoise3 = torch.nn.Conv1d(
            num_bands, num_bands, kernel_size=7, padding=3, dtype=torch.float64
        )
        self.denoise4 = torch.nn.Conv1d(
            num_bands, num_bands, kernel_size=9, padding=4, dtype=torch.float64
        )

        initialize_scaled_kaiming(self.denoise1)
        initialize_scaled_kaiming(self.denoise2)
        initialize_scaled_kaiming(self.denoise3)
        initialize_scaled_kaiming(self.denoise4)
        self._clear_last_forward_artifacts()

    def reset_parameters(self):
        """
        Reset learnable parameters to small random values.

        Initialises tau and D parameters with small Gaussian noise to provide
        a stable starting point for training whilst maintaining numerical precision.
        """
        std = 1e-4
        self.tau.data.normal_(0, 1e-7)
        self.D.data.normal_(0, std)

    def _clear_last_forward_artifacts(self) -> None:
        """Reset cached forward artefacts used by inference-side CMD logging."""
        self._last_forward_artifacts: dict[str, torch.Tensor] = {}

    def _store_last_forward_artifacts(
        self,
        final_csm: torch.Tensor,
        denoise_stage_csms: tuple[torch.Tensor, ...],
    ) -> None:
        """
        Cache detached decoded CSMs from the most recent forward pass.
        
        Parameters
        ----------
        final_csm : torch.Tensor
            The final decoded covariance matrix output from the LAM forward pass.
        denoise_stage_csms : tuple[torch.Tensor, ...]
            A tuple containing the decoded covariance matrices from each denoising stage,
            in order of processing (stage 1 to stage 4).
        """
        self._last_forward_artifacts = {
            "lam_final": final_csm.detach(),
            **{
                f"lam_denoise{stage_index}": stage_csm.detach()
                for stage_index, stage_csm in enumerate(denoise_stage_csms, start=1)
            },
        }

    def _encode_to_latent(self, S: torch.Tensor) -> torch.Tensor:
        """
        Encode input CSMs into the pre-denoising LAM latent space.

        Parameters
        ----------
        S : torch.Tensor
            Input multi-band complex CSM of shape (batch, num_bands, Nch, Nch).

        Returns
        -------
        torch.Tensor
            Latent intensity tensor of shape (batch, num_bands, Npx) representing the
            pre-denoising spatial intensity for each frequency band.
        """
        device = S.device
        self.A = self.A.to(device)
        freq_bands = S.shape[1]

        latent_x_list = []
        for i in range(freq_bands):
            S_i = S[:, i, :, :]
            S_i = 0.5 * (S_i + S_i.transpose(-1, -2).conj())
            Ds, Vs = torch.linalg.eigh(S_i)
            idx = Ds > 0
            Ds = torch.where(idx, Ds, torch.zeros_like(Ds))
            Vs = Vs * torch.sqrt(Ds).unsqueeze(1)

            latent_x = torch.matmul(self.D[i].conj().T, Vs)
            latent_x = torch.linalg.norm(latent_x, dim=2) ** 2
            latent_x -= self.tau[i]
            latent_x_list.append(latent_x)

        return torch.stack(latent_x_list, dim=1)

    def _denoise_latent_steps(
        self, latent_x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Apply the four denoising stages and retain every intermediate latent tensor.

        Parameters
        ----------
        latent_x : torch.Tensor
            Input latent tensor of shape (batch, num_bands, Npx) representing the
            pre-denoising spatial intensity for each frequency band.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
            The final denoised latent tensor and a tuple containing the intermediate
            latent tensors from each denoising stage.
        """
        latent_x_skip = latent_x.clone()

        latent_stage1 = self.denoise1(latent_x) + latent_x_skip
        latent_stage1 = self.retanh(latent_stage1)

        latent_stage2 = self.denoise2(latent_stage1) + latent_x_skip
        latent_stage2 = self.retanh(latent_stage2)

        latent_stage3 = self.denoise3(latent_stage2) + latent_x_skip
        latent_stage3 = self.retanh(latent_stage3)

        latent_stage4 = self.denoise4(latent_stage3) + latent_x_skip
        latent_stage4 = self.retanh(latent_stage4)

        return latent_stage4, (latent_stage1, latent_stage2, latent_stage3, latent_stage4)

    def _decode_latent(self, latent_x: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent tensor back into a multi-band complex CSM.
        
        Parameters
        ----------
        latent_x : torch.Tensor
            Latent intensity tensor of shape (batch, num_bands, Npx) representing the
            denoised spatial intensity for each frequency band.
        
        Returns
        -------
        torch.Tensor
            Reconstructed multi-band complex CSM of shape (batch, num_bands, Nch, Nch).
        """
        out_list = []
        for i in range(latent_x.shape[1]):
            latent_i = latent_x[:, i, :]
            out = torch.einsum(
                "nij,bjk,nkl->bil",
                self.A.unsqueeze(0),
                torch.diag_embed(latent_i.cdouble()),
                self.A.unsqueeze(0).transpose(1, 2).conj(),
            )
            out_list.append(out)
        return torch.stack(out_list, dim=1)

    def forward(  # noqa: C901, PLR0912, PLR0915
        self, S, collect_metrics=False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        """
        Forward pass: map covariance matrices to spatial intensity maps.

        Processes multi-band covariance matrices through encoding, denoising,
        and decoding stages to produce spatial acoustic maps.

        Parameters
        ----------
        S : torch.Tensor
            Input covariance matrices of shape (batch, num_bands, Nch, Nch)
            Complex-valued covariance matrices for each frequency band
        collect_metrics : bool, optional
            If True, collect and return performance metrics (default: False)

        Returns
        -------
        out : torch.Tensor
            Reconstructed spatial covariance matrices of shape
            (batch, num_bands, Nch, Nch)
        latent_x : torch.Tensor
            Denoised latent intensity maps of shape (batch, num_bands, Npx)
            Real-valued intensity for each spatial pixel
        metrics : dict, optional
            Performance metrics (only returned if collect_metrics=True)
            Contains: `encoding_time_ms`, `latent_time_ms`, `decoding_time_ms`,
            `total_time_ms`, `encoding_flops`, `latent_flops`, `decoding_flops`,
            `total_flops`, `memory_mb`, `num_bands`, `batch_size`

        Notes
        -----
        The forward pass consists of three stages:
        1. Encoding: Back-project covariance matrices to latent intensity maps
           via learned dictionary D and eigendecomposition
        2. Denoising: Apply cascaded 1D convolutions with residual connections
           across all frequency bands
        3. Decoding: Reconstruct covariance matrices using steering operator A
           and denoised latent intensities

        The eigendecomposition ensures only positive eigenvalues contribute,
        improving robustness to noise and numerical errors.
        """
        metrics = {}
        device = S.device
        use_cuda = device.type == "cuda"
        self.A = self.A.to(device)
        batch_size, freq_bands, N_ch = S.shape[:3]
        self._clear_last_forward_artifacts()

        # Initialise timing variables for metrics collection
        encoding_start = 0.0
        latent_start = 0.0
        decoding_start = 0.0

        # Encoding stage: back-projection to latent space
        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            encoding_start = time.perf_counter()

        latent_x = self._encode_to_latent(S)

        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            encoding_end = time.perf_counter()
            metrics["encoding_time_ms"] = (encoding_end - encoding_start) * 1000.0

        # Latent space denoising stage
        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            latent_start = time.perf_counter()

        latent_x, latent_stages = self._denoise_latent_steps(latent_x)

        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            latent_end = time.perf_counter()
            metrics["latent_time_ms"] = (latent_end - latent_start) * 1000.0

        # Decoding stage: reconstruct covariance matrices from latent space
        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            decoding_start = time.perf_counter()

        out = self._decode_latent(latent_x)

        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            decoding_end = time.perf_counter()
            metrics["decoding_time_ms"] = (decoding_end - decoding_start) * 1000.0
            metrics["total_time_ms"] = (decoding_end - encoding_start) * 1000.0
            metrics["num_bands"] = freq_bands
            metrics["batch_size"] = batch_size
            metrics["num_frames"] = batch_size

            # FLOPS measurement
            flop_counter = FlopCounterMode(
                display=False,
                custom_mapping=build_custom_flop_mapping(),
            )
            with torch.no_grad():
                with flop_counter:
                    self._forward_no_metrics(S)
            metrics["flops"] = flop_counter.get_total_flops()
            metrics["flops_per_frame"] = metrics["flops"] / batch_size if batch_size > 0 else 0

            # Memory measurement
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    self._forward_no_metrics(S)
                metrics["memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            else:
                tracemalloc.start()
                try:
                    with torch.no_grad():
                        self._forward_no_metrics(S)
                    _, peak = tracemalloc.get_traced_memory()
                    metrics["memory_mb"] = peak / (1024 * 1024)
                finally:
                    tracemalloc.stop()

            denoise_stage_outputs = tuple(
                self._decode_latent(latent_stage) for latent_stage in latent_stages[:-1]
            ) + (out,)
            self._store_last_forward_artifacts(out, denoise_stage_outputs)
            return out, latent_x, metrics

        denoise_stage_outputs = tuple(
            self._decode_latent(latent_stage) for latent_stage in latent_stages[:-1]
        ) + (out,)
        self._store_last_forward_artifacts(out, denoise_stage_outputs)
        return out, latent_x, None

    def _forward_no_metrics(self, S):
        """
        Forward pass without computing metrics for the LAM model.

        This method processes a multi-band signal through the latent space model
        without calculating performance metrics.

        Parameters
        ----------
        S: torch.Tensor
            Input signal tensor of shape (batch, freq_bands, height, width)
            containing the covariance matrices for each frequency band.

        Returns
        -------
        : tuple
            A tuple containing:
                - torch.Tensor: Output tensor of shape (batch, freq_bands, height, width)
                    reconstructed from the latent representation.
                - torch.Tensor: Latent representation tensor of shape (batch, freq_bands,
                num_features)
                    after denoising operations.

        Process:
            1. Eigendecomposes each frequency band's covariance matrix
            2. Projects eigenvectors onto the learned subspace (self.D)
            3. Computes latent features and subtracts thresholds (self.tau)
            4. Applies cascaded denoising layers with residual connections and tanh activations
            5. Reconstructs output signals using the learned transformation matrix (self.A)
        """
        latent_x = self._encode_to_latent(S)
        latent_x, _ = self._denoise_latent_steps(latent_x)
        return self._decode_latent(latent_x), latent_x
