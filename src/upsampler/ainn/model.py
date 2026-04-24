"""
AINN-based upsampler with Helmholtz PDE regularisation.

This module adapts the original acoustics-informed neural network (AINN)
to the repository's trainable upsampler interface.
Predictions are made through a coordinate-query MLP, but unlike the original per-scene fitting
workflow, this version conditions every query on the observed low-resolution
complex CSM.

References
----------
.. [1] S. Zhao, F. Ma,
       "A circular microphone array with virtual microphones based on acoustics-informed neural
       networks",
       The Journal of the Acoustical Society of America,
       https://doi.org/10.1121/10.0027915
.. [2] Official implementation:
       https://github.com/sipeizhao/AINN
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from lam_min.trainer.utils import get_xyz
from upsampler.base import TrainableUpsampler

DEFAULT_LOW_CHANNEL_INDICES = (5, 9, 21, 25)


class AINNUpsampler(TrainableUpsampler):  # type: ignore[no-any-unimported]
    """AINN-baed upsampler with Helmholtz PDE regularisation as described in [1] and [2]."""

    EXPECTED_INPUT_NDIM = 4
    COMPLEX_CHANNELS = 2
    MIC_COORD_DIMS = 3
    PAIR_COORD_DIMS = 2 * MIC_COORD_DIMS
    PAPER_HIDDEN_LAYERS = 2
    MAX_EIGENMIKE_CHANNELS = 32

    def __init__(  # noqa: C901, PLR0913, PLR0915
        self,
        in_channels: int = 4,
        out_channels: int = 32,
        hidden_channels: int = 64,
        low_channel_indices: Sequence[int] = DEFAULT_LOW_CHANNEL_INDICES,
        loss_name: str = "mse",
        pde_loss_weight: float = 0.01,
        pde_freq_min_hz: float = 100.0,
        pde_freq_max_hz: float = 4000.0,
        sound_speed: float = 340.0,
    ) -> None:
        """
        Initialise AINN MLP and loss settings.

        Parameters
        ----------
        in_channels : int, optional
            Number of low-resolution channels (default: 4).
        out_channels : int, optional
            Number of high-resolution channels (default: 32).
        hidden_channels : int, optional
            Width of hidden AINN MLP layers (default: 64).
        low_channel_indices : Sequence[int], optional
            Zero-based Eigenmike channel indices defining the low-resolution branch.
        loss_name : str, optional
            Reconstruction criterion for the data loss (default: "mse").
        pde_loss_weight : float, optional
            Weight of Helmholtz residual regularisation in AINN training (default: 0.01).
        pde_freq_min_hz : float, optional
            Minimum frequency for the PDE loss (default: 100.0).
        pde_freq_max_hz : float, optional
            Maximum frequency for the PDE loss (default: 4000.0).
        sound_speed : float, optional
            Speed of sound in the medium (default: 340.0).
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.n_layers = self.PAPER_HIDDEN_LAYERS
        self.loss_name = loss_name.strip().lower()
        self.pde_loss_weight = float(pde_loss_weight)
        self.pde_freq_min_hz = float(pde_freq_min_hz)
        self.pde_freq_max_hz = float(pde_freq_max_hz)
        self.sound_speed = float(sound_speed)
        self.low_channel_indices = tuple(int(index) for index in low_channel_indices)

        positive_int_fields = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_channels": self.hidden_channels,
        }
        for field_name, field_value in positive_int_fields.items():
            if field_value <= 0:
                raise ValueError(f"{field_name} must be > 0.")

        if len(self.low_channel_indices) != self.in_channels:
            raise ValueError(
                "low_channel_indices must have length equal to in_channels, "
                f"got {self.low_channel_indices}"
            )
        if len(set(self.low_channel_indices)) != len(self.low_channel_indices):
            raise ValueError("low_channel_indices must not contain duplicates.")
        if min(self.low_channel_indices) < 0:
            raise ValueError("low_channel_indices must be non-negative.")
        if max(self.low_channel_indices) >= self.MAX_EIGENMIKE_CHANNELS:
            raise ValueError(
                "low_channel_indices must reference valid Eigenmike channels in [0, 31]."
            )
        if self.out_channels > self.MAX_EIGENMIKE_CHANNELS:
            raise ValueError(
                f"out_channels must be <= {self.MAX_EIGENMIKE_CHANNELS}, got {self.out_channels}."
            )
        if self.pde_loss_weight < 0.0:
            raise ValueError("pde_loss_weight must be >= 0.")
        if self.pde_freq_min_hz <= 0.0:
            raise ValueError("pde_freq_min_hz must be > 0.")
        if self.pde_freq_max_hz < self.pde_freq_min_hz:
            raise ValueError("pde_freq_max_hz must be >= pde_freq_min_hz.")
        if self.sound_speed <= 0.0:
            raise ValueError("sound_speed must be > 0.")

        all_mic_positions = torch.tensor(get_xyz(), dtype=torch.float32)
        low_mic_positions = all_mic_positions[list(self.low_channel_indices)]
        high_mic_positions = all_mic_positions[: self.out_channels]

        low_pair_coords = self._build_pair_coordinates(low_mic_positions)
        high_pair_coords = self._build_pair_coordinates(high_mic_positions)

        self.register_buffer("low_mic_positions", low_mic_positions, persistent=False)
        self.register_buffer("high_mic_positions", high_mic_positions, persistent=False)
        self.register_buffer("low_pair_coords", low_pair_coords, persistent=False)
        self.register_buffer("high_pair_coords", high_pair_coords, persistent=False)
        self.register_buffer(
            "low_output_indices",
            torch.tensor(self.low_channel_indices, dtype=torch.long),
            persistent=False,
        )

        self.conditioning_dims = self.COMPLEX_CHANNELS * self.in_channels * self.in_channels
        input_dims = self.PAIR_COORD_DIMS + self.conditioning_dims
        self.model_layers = nn.ModuleList()
        self.model_layers.append(nn.Linear(input_dims, self.hidden_channels, dtype=torch.float32))
        for _ in range(self.n_layers - 1):
            self.model_layers.append(
                nn.Linear(self.hidden_channels, self.hidden_channels, dtype=torch.float32)
            )
        self.output_layer = nn.Linear(
            self.hidden_channels,
            self.COMPLEX_CHANNELS,
            dtype=torch.float32,
        )

        self.loss_fn = self._build_loss(self.loss_name)
        self._init_weights()

    @staticmethod
    def _build_pair_coordinates(mic_positions: torch.Tensor) -> torch.Tensor:
        """
        Build ordered microphone-pair coordinates.

        Each CSM element is represented as
        ``[x_i, y_i, z_i, x_j, y_j, z_j]`` for an ordered pair ``(i, j)``.
        This ordering is used for both the low-resolution and high-resolution microphone sets, so
        the AINN can learn to generalise across different pairings and upsample from any
        low-resolution subset to the full high-resolution array.

        Parameters
        ----------
        mic_positions : torch.Tensor
            Tensor of shape (num_mics, 3) containing 3D microphone coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_mics * num_mics, 6) containing pair coordinates for all ordered
            microphone pairs.
        """
        num_mics = int(mic_positions.shape[0])
        row_positions = mic_positions.unsqueeze(1).expand(num_mics, num_mics, -1)
        col_positions = mic_positions.unsqueeze(0).expand(num_mics, num_mics, -1)
        return torch.cat((row_positions, col_positions), dim=-1).reshape(
            num_mics * num_mics,
            2 * mic_positions.shape[-1],
        )

    def _build_loss(self, loss_name: str) -> nn.Module:
        """
        Loss function to use for the supervised data loss component.

        The PDE loss is computed separately.

        Parameters
        ----------
        loss_name : str
            Name of the loss function to use for the data loss.
            Supported values are:
            - "l1": Mean Absolute Error (L1 Loss)
            - "mse": Mean Squared Error (MSE Loss)

        Returns
        -------
        nn.Module
            The loss function module corresponding to the specified loss_name.
        """
        if loss_name == "l1":
            return nn.L1Loss()
        if loss_name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss_name '{loss_name}'. Use one of: l1, mse.")

    def _init_weights(self) -> None:
        """
        Glorot/Xavier initialisation for all linear layers.

        This is the same initialisation used in the original AINN implementation.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _network_forward(self, network_input: torch.Tensor) -> torch.Tensor:
        """
        Forward through the MLP.

        In the original AINN implementation tanh activations are used after each hidden layer.

        Parameters
        ----------
        network_input : torch.Tensor
            Input tensor to the MLP, with shape (n_samples * n_points, n_features).

        Returns
        -------
        torch.Tensor
        Output tensor from the MLP, with shape (n_samples * n_points, 2), where the last dimension
        corresponds to the real and imaginary parts of the predicted complex value.
        """
        x = network_input
        for layer in self.model_layers:
            x = torch.tanh(layer(x))
        return cast(torch.Tensor, self.output_layer(x))

    def _validate_input(self, S_low: torch.Tensor) -> None:
        """
        Validate the low-resolution CSM input contract.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).

        Raises
        ------
        ValueError
            If S_low does not meet the expected shape, dtype, or channel requirements.
        """
        if S_low.ndim != self.EXPECTED_INPUT_NDIM:
            raise ValueError(f"S_low must be 4D [B, F, C, C], got shape={tuple(S_low.shape)}")
        if not S_low.is_complex():
            raise ValueError("S_low must be a complex tensor.")
        if S_low.shape[-2:] != (self.in_channels, self.in_channels):
            raise ValueError(
                "S_low must have shape [B, F, "
                f"{self.in_channels}, {self.in_channels}] on the last two dimensions, "
                f"got {tuple(S_low.shape)}"
            )

    def _build_conditioning_vector(self, S_low: torch.Tensor) -> torch.Tensor:
        """
        Flatten low-resolution real and imaginary CSM values into conditioning features.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).

        Returns
        -------
        torch.Tensor
            Real-valued tensor of shape (batch * num_bands, in_channels * in_channels * 2)
            containing the flattened real and imaginary parts of S_low for conditioning.
        """
        batch_size, num_bands, _, _ = S_low.shape
        n_samples = batch_size * num_bands
        real = S_low.real.reshape(n_samples, -1)
        imag = S_low.imag.reshape(n_samples, -1)
        return torch.cat((real, imag), dim=-1).to(torch.float32)

    def _build_network_inputs(
        self,
        S_low: torch.Tensor,
        pair_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """
        Expand pair coordinates and conditioning features into per-query network inputs.

        This method constructs the input for each pair-coordinate query by concatenating the query's
        coordinates with the flattened low-resolution CSM values for conditioning. The resulting
        tensor has shape (batch * num_bands, num_pairs, PAIR_COORD_DIMS + conditioning_dims), where
        num_pairs is the number of microphone pairs defined by the provided pair_coordinates.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).
        pair_coordinates : torch.Tensor
            Tensor of shape (num_pairs, PAIR_COORD_DIMS) containing the coordinates for each
            microphone pair query.

        Returns
        -------
        tuple[torch.Tensor, int, int]
            - network_inputs: Tensor of shape (batch * num_bands, num_pairs, PAIR_COORD_DIMS
                + conditioning_dims) containing the input for each pair-coordinate query.
            - batch_size: The batch size extracted from S_low.
            - num_bands: The number of frequency bands extracted from S_low.
        """
        batch_size, num_bands, _, _ = S_low.shape
        n_samples = batch_size * num_bands
        coords = pair_coordinates.to(device=S_low.device, dtype=torch.float32)
        coords = coords.unsqueeze(0).expand(n_samples, -1, -1)
        conditioning = self._build_conditioning_vector(S_low)
        conditioning = conditioning.unsqueeze(1).expand(-1, coords.shape[1], -1)
        return torch.cat((coords, conditioning), dim=-1), batch_size, num_bands

    def _predict_points(self, network_inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict real/imag values at all conditioned pair-coordinate queries.

        Parameters
        ----------
        network_inputs : torch.Tensor
            Tensor of shape (batch * num_bands, num_pairs, PAIR_COORD_DIMS + conditioning_dims)
            containing the input for each pair-coordinate query.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch * num_bands, num_pairs, COMPLEX_CHANNELS) containing the
            predicted real and imaginary parts.

        """
        n_samples, n_points, _ = network_inputs.shape
        flat_inputs = network_inputs.reshape(n_samples * n_points, -1)
        pred_points = self._network_forward(flat_inputs)
        return pred_points.reshape(n_samples, n_points, self.COMPLEX_CHANNELS)

    def _reshape_predictions(
        self,
        pred_points: torch.Tensor,
        batch_size: int,
        num_bands: int,
        output_channels: int,
    ) -> torch.Tensor:
        """
        Reshape query predictions back to a complex CSM tensor.

        The predicted real and imaginary values are first reshaped to the appropriate dimensions
        and then combined into a complex tensor. Finally, the output is symmetrised to ensure a
        Hermitian CSM.

        Parameters
        ----------
        pred_points : torch.Tensor
            Tensor of shape (batch * num_bands, num_pairs, COMPLEX_CHANNELS) containing the
            predicted real and imaginary parts for each pair-coordinate query.
        batch_size : int
            The batch size corresponding to the original input tensor.
        """
        pred_real = pred_points[..., 0].reshape(
            batch_size, num_bands, output_channels, output_channels
        )
        pred_imag = pred_points[..., 1].reshape(
            batch_size, num_bands, output_channels, output_channels
        )
        S_up = torch.complex(pred_real, pred_imag)
        return 0.5 * (S_up + S_up.transpose(-1, -2).conj())

    def _forward_no_metrics(self, S_low: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the conditional AINN to predict a high-resolution CSM.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            ``(batch, num_bands, in_channels, in_channels)``.

        Returns
        -------
        torch.Tensor
            Complex high-resolution CSM tensor with shape
            ``(batch, num_bands, out_channels, out_channels)``.
        """
        network_inputs, batch_size, num_bands = self._build_network_inputs(
            S_low,
            self.high_pair_coords,
        )
        pred_points = self._predict_points(network_inputs)
        S_up = self._reshape_predictions(pred_points, batch_size, num_bands, self.out_channels)
        return self._restore_observed_block(S_up=S_up, S_low=S_low)

    def _restore_observed_block(self, S_up: torch.Tensor, S_low: torch.Tensor) -> torch.Tensor:
        """
        Reinsert the measured low-resolution block into the predicted high-resolution CSM.

        The upsampler should infer the missing high-resolution entries, but it should not
        overwrite the entries that were directly observed in the input low-resolution CSM.

        Parameters
        ----------
        S_up : torch.Tensor
            Predicted high-resolution complex CSM tensor with shape
            ``(batch, num_bands, out_channels, out_channels)``.
        S_low : torch.Tensor
            Observed low-resolution complex CSM tensor with shape
            ``(batch, num_bands, in_channels, in_channels)``.

        Returns
        -------
        torch.Tensor
            High-resolution complex CSM with the observed low-resolution block restored.
        """
        if int(self.low_output_indices.max().item()) >= self.out_channels:
            return S_up

        low_output_indices = self.low_output_indices.to(device=S_up.device)
        restored = S_up.clone()
        restored[
            ...,
            low_output_indices.unsqueeze(1),
            low_output_indices.unsqueeze(0),
        ] = S_low.to(dtype=restored.dtype)
        return 0.5 * (restored + restored.transpose(-1, -2).conj())

    def _build_wavenumbers(
        self,
        num_bands: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build per-band wave numbers for Helmholtz residual.

        Parameters
        ----------
        num_bands : int
            Number of frequency bands, used to determine the wave number for each point.
        device : torch.device
            Device on which to create the wave number tensor.
        dtype : torch.dtype
            Data type for the wave number tensor.

        Returns
        -------
        torch.Tensor
            1D tensor of wave numbers.
        """
        freqs = torch.linspace(
            self.pde_freq_min_hz,
            self.pde_freq_max_hz,
            steps=num_bands,
            device=device,
            dtype=dtype,
        )
        return 2.0 * torch.pi * freqs / self.sound_speed

    @staticmethod
    def _laplacian(
        gradient: torch.Tensor,
        geometry_inputs: torch.Tensor,
        dims: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Compute the Laplacian over a selected 3D coordinate subspace.

        The Laplacian is computed as the sum of second derivatives along the specified dimensions.

        Parameters
        ----------
        gradient : torch.Tensor
            Tensor containing the first derivatives of the predicted field with respect to the input
            coordinates, with shape (n_samples * n_points, PAIR_COORD_DIMS).
        geometry_inputs : torch.Tensor
            Tensor containing the input coordinates for the pair-coordinate queries, with shape
            (n_samples * n_points, PAIR_COORD_DIMS).
        dims : tuple[int, int, int]
            A tuple of three integers specifying the dimensions.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_samples * n_points,) containing the computed Laplacian values for
            the specified coordinate subspace.
        """
        second_derivatives: list[torch.Tensor] = []
        for dim in dims:
            second_derivative = torch.autograd.grad(
                gradient[:, dim],
                geometry_inputs,
                grad_outputs=torch.ones_like(gradient[:, dim]),
                create_graph=True,
                retain_graph=True,
            )[0][:, dim]
            second_derivatives.append(second_derivative)
        return second_derivatives[0] + second_derivatives[1] + second_derivatives[2]

    def calculate_pde_loss(
        self,
        network_inputs: torch.Tensor,
        num_bands: int,
    ) -> torch.Tensor:
        """
        Calculate the Helmholtz residual loss on conditioned pair-coordinate queries.

        The residual is evaluated separately over the row-microphone and
        column-microphone 3D subspaces and the resulting losses are averaged.

        Parameters
        ----------
        network_inputs : torch.Tensor
            Tensor of shape (batch * num_bands, num_pairs, PAIR_COORD_DIMS + conditioning_dims)
            containing the input for each pair-coordinate query.
        num_bands : int
            Number of frequency bands, used to determine the wave number for each point.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the computed PDE loss value.
        """
        n_samples, n_points, _ = network_inputs.shape
        geometry_inputs = (
            network_inputs[..., : self.PAIR_COORD_DIMS]
            .reshape(n_samples * n_points, self.PAIR_COORD_DIMS)
            .clone()
        )
        geometry_inputs = geometry_inputs.to(torch.float32)
        geometry_inputs.requires_grad_(True)

        conditioning_inputs = (
            network_inputs[..., self.PAIR_COORD_DIMS :]
            .reshape(n_samples * n_points, self.conditioning_dims)
            .to(torch.float32)
        )
        flat_inputs = torch.cat((geometry_inputs, conditioning_inputs.detach()), dim=-1)

        pred = self._network_forward(flat_inputs)
        pred_real = pred[:, 0]
        pred_imag = pred[:, 1]

        grad_real = torch.autograd.grad(
            pred_real,
            geometry_inputs,
            grad_outputs=torch.ones_like(pred_real),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_imag = torch.autograd.grad(
            pred_imag,
            geometry_inputs,
            grad_outputs=torch.ones_like(pred_imag),
            create_graph=True,
            retain_graph=True,
        )[0]

        lap_row_real = self._laplacian(grad_real, geometry_inputs, dims=(0, 1, 2))
        lap_col_real = self._laplacian(grad_real, geometry_inputs, dims=(3, 4, 5))
        lap_row_imag = self._laplacian(grad_imag, geometry_inputs, dims=(0, 1, 2))
        lap_col_imag = self._laplacian(grad_imag, geometry_inputs, dims=(3, 4, 5))

        if num_bands <= 0 or n_samples % num_bands != 0:
            raise ValueError(
                "num_bands must be positive and divide the number of conditioned samples, "
                f"got num_bands={num_bands}, n_samples={n_samples}."
            )
        batch_size = n_samples // num_bands

        # Build wave numbers for each point based on its frequency band and compute PDE residuals
        k_per_band = self._build_wavenumbers(
            num_bands=num_bands,
            device=network_inputs.device,
            dtype=torch.float32,
        )
        k_per_sample = k_per_band.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        k_per_point = k_per_sample.repeat_interleave(n_points)
        k2 = torch.clamp(k_per_point * k_per_point, min=1.0e-12)

        residual_row_real = lap_row_real / k2 + pred_real
        residual_col_real = lap_col_real / k2 + pred_real
        residual_row_imag = lap_row_imag / k2 + pred_imag
        residual_col_imag = lap_col_imag / k2 + pred_imag

        loss_row_real = torch.mean(residual_row_real.square())
        loss_col_real = torch.mean(residual_col_real.square())
        loss_row_imag = torch.mean(residual_row_imag.square())
        loss_col_imag = torch.mean(residual_col_imag.square())
        # Return the average of the row and column residual losses for both real and imaginary parts
        return 0.25 * (loss_row_real + loss_col_real + loss_row_imag + loss_col_imag)

    def calculate_data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised data loss on real and imaginary parts.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted complex values at the points, with shape (n_samples * n_points, 2).
        target : torch.Tensor
            Ground truth complex values at the points, with shape (n_samples * n_points, 2).

        Returns
        -------
        torch.Tensor
            The computed data loss as a scalar tensor.
            It is the average of the losses for the real and imaginary parts.
        """
        loss_real = cast(
            torch.Tensor,
            self.loss_fn(pred.real.to(torch.float32), target.real.to(torch.float32)),
        )
        loss_imag = cast(
            torch.Tensor,
            self.loss_fn(pred.imag.to(torch.float32), target.imag.to(torch.float32)),
        )
        return 0.5 * (loss_real + loss_imag)

    def forward(
        self, S_low: torch.Tensor, collect_metrics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Upsample low-resolution complex CSM and optionally collect runtime metrics.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex tensor with shape [batch, num_bands, in_channels, in_channels].
        collect_metrics : bool, optional
            Whether to return runtime metrics.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, dict[str, float]]
            Upsampled complex tensor, or tensor and metrics dictionary.
        """
        self._validate_input(S_low)

        use_cuda = S_low.device.type == "cuda"
        start = 0.0
        if collect_metrics:
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(S_low.device)
                torch.cuda.synchronize(S_low.device)
            else:
                tracemalloc.start()
            start = time.perf_counter()

        batch_size, _, _, _ = S_low.shape

        S_up = self._forward_no_metrics(S_low)
        if not collect_metrics:
            return S_up

        if use_cuda:
            torch.cuda.synchronize(S_low.device)
        end = time.perf_counter()

        if use_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(S_low.device) / (1024.0 * 1024.0)
        else:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory / (1024.0 * 1024.0)

        flop_counter = FlopCounterMode(display=False)
        with torch.no_grad():
            with flop_counter:
                self._forward_no_metrics(S_low)
        total_flops = float(flop_counter.get_total_flops())
        flops_per_frame = total_flops / batch_size if batch_size > 0 else 0.0

        metrics: dict[str, float] = {
            "upsampler_time_ms": (end - start) * 1000.0,
            "upsampler_flops": total_flops,
            "upsampler_flops_per_frame": flops_per_frame,
            "upsampler_memory_mb": float(peak_memory_mb),
            "num_frames": float(batch_size),
        }
        return S_up, metrics

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        network_inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute AINN total loss = data loss + weighted PDE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted complex values at the points, with shape (n_samples * n_points, 2).
        target : torch.Tensor
            Ground truth complex values at the points, with shape (n_samples * n_points, 2).
        network_inputs : torch.Tensor | None
            The input coordinates for the PDE loss calculation, with shape
            (n_samples, n_points, n_features).
            If None, the PDE loss will not be computed and will be set to zero.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            A tuple containing the total loss as a scalar tensor and a dictionary of
            loss components.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

        loss_data = self.calculate_data_loss(pred=pred, target=target)

        if self.pde_loss_weight > 0.0 and network_inputs is not None:
            loss_pde = self.calculate_pde_loss(
                network_inputs=network_inputs,
                num_bands=int(target.shape[1]),
            )
        else:
            loss_pde = loss_data.new_zeros(())

        total = loss_data + self.pde_loss_weight * loss_pde

        stats: dict[str, float] = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_data": float(loss_data.detach().cpu().item()),
            "loss_pde": float(loss_pde.detach().cpu().item()),
            "loss_pde_weight": self.pde_loss_weight,
        }
        if self.loss_name == "l1":
            stats["loss_l1"] = float(loss_data.detach().cpu().item())
        elif self.loss_name == "mse":
            stats["loss_mse"] = float(loss_data.detach().cpu().item())

        return total, cast(dict[str, float], self.normalise_step_stats(stats))

    def training_step(
        self,
        S_low: torch.Tensor,
        S_high: torch.Tensor,
        optimiser: torch.optim.Optimizer,
        grad_clip_norm: float = 0.0,
    ) -> dict[str, float]:
        """
        Run one optimisation step.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).
        S_high : torch.Tensor
            Complex high-resolution CSM tensor with shape
            (batch, num_bands, out_channels, out_channels).
        optimiser : torch.optim.Optimizer
            The optimiser to use for the update step.
        grad_clip_norm : float, optional
            Maximum norm for gradient clipping. If <= 0, no clipping is applied (default: 0.0).

        Returns
        -------
        dict[str, float]
            A dictionary containing the loss components for the current step.
        """
        self._validate_input(S_low)
        optimiser.zero_grad(set_to_none=True)

        network_inputs, batch_size, num_bands = self._build_network_inputs(
            S_low,
            self.high_pair_coords,
        )
        pred_points = self._predict_points(network_inputs)
        pred = self._reshape_predictions(pred_points, batch_size, num_bands, self.out_channels)
        pred = self._restore_observed_block(S_up=pred, S_low=S_low)

        loss, stats = self.compute_loss(pred=pred, target=S_high, network_inputs=network_inputs)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_norm)
        optimiser.step()
        return stats

    @torch.no_grad()
    def validation_step(self, S_low: torch.Tensor, S_high: torch.Tensor) -> dict[str, float]:
        """
        Run one validation step without PDE regularisation.

        Parameters
        ----------
        S_low : torch.Tensor
            Complex low-resolution CSM tensor with shape
            (batch, num_bands, in_channels, in_channels).
        S_high : torch.Tensor
            Complex high-resolution CSM tensor with shape
            (batch, num_bands, out_channels, out_channels).

        Returns
        -------
        dict[str, float]
            A dictionary containing the loss components for the current step.
        """
        pred = cast(torch.Tensor, self.forward(S_low, collect_metrics=False))
        _, stats = self.compute_loss(pred=pred, target=S_high, network_inputs=None)
        return cast(dict[str, float], self.normalise_step_stats(stats))
