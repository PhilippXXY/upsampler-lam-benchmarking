"""
Upsampled Latent Acoustic Map (UpLAM) model.

This module implements the UpLAM network that combines complex-valued Deep
Back-Projection Network (CDBPN) for spatial resolution enhancement with the
Latent Acoustic Map (LAM) for Direction of Arrival estimation.

The UpLAM pipeline:
    1. Upsample low-resolution covariance matrices using CDBPN
    2. Apply LAM to estimate DoA from upsampled high-resolution matrices

This two-stage approach enables super-resolution acoustic mapping, improving
spatial localisation accuracy compared to direct LAM application.

References
----------
.. [1] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"
"""

import torch
from torch import nn

from lam_min.model.cdbpn import Net as CDBPN
from lam_min.model.LAM import LAM


class UpLAM(nn.Module):
    """
    Upsampled Latent Acoustic Map network.

    Combines spatial upsampling and DoA estimation in a unified architecture.
    First applies CDBPN to increase the spatial resolution of covariance matrices,
    then uses LAM to extract Direction of Arrival estimates from the enhanced data.

    Parameters
    ----------
    num_bands : int, optional
        Number of frequency bands (default: 16)
    base_filter : int, optional
        Number of base filters in CDBPN layers (default: 32)
    feat : int, optional
        Feature dimension for CDBPN initial extraction (default: 128)
    num_stages : int, optional
        Number of back-projection stages in CDBPN (default: 10)
    scale_factor : int, optional
        Spatial upsampling factor for CDBPN (default: 8)

    Attributes
    ----------
    cdbpn : CDBPN
        Complex Deep Back-Projection Network for spatial upsampling
    lam : LAM
        Latent Acoustic Map for DoA estimation

    Notes
    -----
    - The model processes complex-valued covariance matrices
    - CDBPN operates on real and imaginary components separately
    - Output intensity maps from LAM can be used for k-means clustering

    See Also
    --------
    LAM : Latent Acoustic Map for DoA estimation
    CDBPN : Complex Deep Back-Projection Network for upsampling
    """

    def __init__(
        self,
        num_bands: int = 16,
        base_filter: int = 32,
        feat: int = 128,
        num_stages: int = 10,
        scale_factor: int = 8,
        freeze_lam: bool = True,
    ) -> None:
        """
        Initialise the UpLAM model.

        Parameters
        ----------
        num_bands : int, optional
            Number of frequency bands to process (default: 16)
        base_filter : int, optional
            Base filter count for CDBPN (default: 32)
        feat : int, optional
            Feature extraction dimension (default: 128)
        num_stages : int, optional
            Number of up-down projection stages (default: 10)
        scale_factor : int, optional
            Upsampling scale factor (default: 8)
        """
        super(UpLAM, self).__init__()
        self.cdbpn = CDBPN(num_bands, base_filter, feat, num_stages, scale_factor=scale_factor)
        self.lam = LAM(num_bands)
        self._last_upsampler_output: torch.Tensor | None = None
        if freeze_lam:
            for param in self.lam.parameters():
                param.requires_grad = False

    @property
    def upsampler(self) -> nn.Module:
        """
        Return the trainable upsampler branch.

        This keeps UpLAM compatible with the repository's end-to-end trainer
        interface without altering the retained checkpoint key layout.
        """
        return self.cdbpn

    def _prepare_cdbpn_input(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cast complex input to the precision expected by the upstream CDBPN path.

        Parameters
        ----------
        S : torch.Tensor
            Complex low-resolution CSM tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Real and imaginary components in float64 precision.
        """
        S_complex = S.to(dtype=self.lam.D.dtype)
        return S_complex.real, S_complex.imag

    def forward_components(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a forward pass and expose intermediate tensors for training.

        Parameters
        ----------
        S : torch.Tensor
            Complex low-resolution CSM tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Upsampled CSM, final LAM reconstruction, and latent map.
        """
        x_rel, x_imag = self._prepare_cdbpn_input(S)
        S_pred = self.cdbpn(x_rel, x_imag, collect_metrics=False).to(dtype=self.lam.D.dtype)
        self._last_upsampler_output = S_pred.detach()
        out, x, _ = self.lam(S_pred, collect_metrics=False)
        return S_pred, out, x

    def forward(self, S, collect_metrics=False):
        """
        Forward pass: upsample and estimate DoA from covariance matrices.

        Applies two-stage processing: spatial upsampling followed by DoA estimation.

        Parameters
        ----------
        S : torch.Tensor
            Input low-resolution covariance matrices of shape
            (batch, num_bands, Nch_low, Nch_low)
            Complex-valued covariance matrices
        collect_metrics : bool, optional
            If True, collect and return performance metrics (default: False)

        Returns
        -------
        out : torch.Tensor
            Reconstructed high-resolution spatial covariance matrices of shape
            (batch, num_bands, Nch_high, Nch_high)
        x : torch.Tensor
            Latent intensity maps of shape (batch, num_bands, Npx)
            Real-valued DoA intensity estimates for each spatial pixel
        metrics : dict, optional
            Performance metrics (only returned if collect_metrics=True)
            Contains timing, FLOPs, and memory metrics for each stage

        Notes
        -----
        The forward pass consists of:
        1. CDBPN upsampling: Increases spatial resolution by scale_factor
        2. LAM processing: Extracts DoA information from upsampled data

        The real and imaginary components are processed separately by CDBPN
        before being recombined into complex matrices for LAM.
        """
        self._last_upsampler_output = None
        if collect_metrics:
            metrics = {}

            # CDBPN with metrics
            x_rel, x_imag = self._prepare_cdbpn_input(S)
            cdbpn_result = self.cdbpn(x_rel, x_imag, collect_metrics=True)
            S_pred, cdbpn_metrics = cdbpn_result
            S_pred = S_pred.to(dtype=self.lam.D.dtype)
            self._last_upsampler_output = S_pred.detach()
            metrics.update(cdbpn_metrics)

            # LAM processing with metrics
            lam_result = self.lam(S_pred, collect_metrics=True)
            out, x, lam_metrics = lam_result

            # Prefix LAM metrics
            for key, value in lam_metrics.items():
                metrics[f'lam_{key}'] = value

            # Compute combined totals
            metrics['total_time_ms'] = (
                metrics.get('upsampler_time_ms', 0) + 
                metrics.get('lam_total_time_ms', 0)
            )
            metrics['total_flops'] = (
                metrics.get('upsampler_flops', 0) + 
                metrics.get('lam_flops', 0)
            )
            metrics['total_memory_mb'] = (
                metrics.get('upsampler_memory_mb', 0) + 
                metrics.get('lam_memory_mb', 0)
            )

            # Per-frame metrics
            num_frames = S.shape[0]
            metrics['num_frames'] = num_frames
            if num_frames > 0:
                metrics['latency_per_frame_ms'] = metrics['total_time_ms'] / num_frames
                metrics['flops_per_frame'] = metrics['total_flops'] / num_frames

            return out, x, metrics

        # Standard forward pass without metrics
        _, out, x = self.forward_components(S)

        return out, x
