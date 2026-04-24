"""
Single-band Latent Acoustic Map layer.

This module implements a single-frequency-band version of the LAM architecture,
suitable for processing one frequency band at a time. It performs the same
back-projection, denoising, and reconstruction operations as the full LAM model
but for a single frequency channel.

The LAMLayer is primarily used as a building block or for testing individual
frequency band processing independently.

See Also
--------
LAM : Multi-band LAM model that processes all frequencies jointly
"""

import torch
from torch import nn

from lam_min.trainer.utils import steering_operator


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
    torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data.fill_(1e-6)  # Small bias to avoid dead neurons


class LAMLayer(torch.nn.Module):
    """
    Single-band Latent Acoustic Map layer.

    A single-frequency variant of the LAM architecture that processes one
    covariance matrix at a time. Uses learned dictionary back-projection,
    cascaded convolutional denoising, and steering vector reconstruction.

    Parameters
    ----------
    Nch : int, optional
        Number of microphone array channels (default: 32)
    tau : torch.Tensor, optional
        Bias parameters of shape (Npx,) for thresholding latent space.
        If None, parameters are randomly initialised.
    D : torch.Tensor, optional
        Dictionary matrix of shape (Nch, Npx) for back-projection.
        If None, parameters are randomly initialised.

    Attributes
    ----------
    A : torch.Tensor
        Steering operator matrix relating spatial pixels to microphone measurements
    tau : torch.nn.Parameter
        Learnable threshold bias for each spatial pixel
    D : torch.nn.Parameter
        Learnable dictionary for projecting microphone signals to latent space
    relu : torch.nn.ReLU
        ReLU activation function
    denoise1, denoise2, denoise3, denoise4 : torch.nn.Conv1d
        Multi-scale 1D convolutional layers for latent space denoising
    """

    def __init__(self, Nch=32, tau=None, D=None):
        """
        Initialise the LAMLayer.

        Parameters
        ----------
        Nch : int, optional
            Number of microphone channels (default: 32)
        tau : torch.Tensor, optional
            Pre-initialised bias parameters (default: None)
        D : torch.Tensor, optional
            Pre-initialised dictionary matrix (default: None)
        """
        super().__init__()
        self.A = torch.from_numpy(steering_operator())
        self.A.requires_grad = False
        Npx = self.A.shape[-1]
        if tau is None or D is None:
            self.tau = torch.nn.Parameter(torch.empty((Npx), dtype=torch.float64))
            self.D = torch.nn.Parameter(torch.empty((Nch, Npx), dtype=torch.complex128))
            self.reset_parameters()
        else:
            self.tau = torch.nn.Parameter(tau)
            self.D = torch.nn.Parameter(D)
        self.relu = nn.ReLU()

        # layers used for cascaded conv with small bias
        self.denoise1 = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1, dtype=torch.float64)
        self.denoise2 = torch.nn.Conv1d(1, 1, kernel_size=5, padding=2, dtype=torch.float64)
        self.denoise3 = torch.nn.Conv1d(1, 1, kernel_size=7, padding=3, dtype=torch.float64)
        self.denoise4 = torch.nn.Conv1d(1, 1, kernel_size=9, padding=4, dtype=torch.float64)
        # He initializtion with small variance to handle small numbers better
        initialize_scaled_kaiming(self.denoise1)
        initialize_scaled_kaiming(self.denoise2)
        initialize_scaled_kaiming(self.denoise3)
        initialize_scaled_kaiming(self.denoise4)

    def reset_parameters(self):
        """
        Reset learnable parameters to small random values.

        Initialises tau and D parameters with small Gaussian noise to provide
        a stable starting point for training whilst maintaining numerical precision.
        """
        std = 1e-4
        self.tau.data.normal_(0, 1e-7)
        self.D.data.normal_(0, std)

    def forward(self, S):
        """
        Forward pass: map single-band covariance to spatial intensity map.

        Processes a single-frequency covariance matrix through encoding,
        denoising, and decoding stages.

        Parameters
        ----------
        S : torch.Tensor
            Input covariance matrix of shape (batch, 1, Nch, Nch)
            Complex-valued covariance matrix (single frequency band)

        Returns
        -------
        out : torch.Tensor
            Reconstructed spatial covariance matrix of shape (batch, 1, Nch, Nch)
        latent_x : torch.Tensor
            Denoised latent intensity map of shape (batch, Npx)
            Real-valued intensity for each spatial pixel

        Notes
        -----
        The forward pass consists of:
        1. Encoding: Back-project covariance matrix to latent intensity via
           learned dictionary D and eigendecomposition
        2. Denoising: Apply cascaded 1D convolutions with residual connections
        3. Decoding: Reconstruct covariance matrix using steering operator A
           and denoised latent intensity

        The eigendecomposition ensures only positive eigenvalues contribute.
        """
        device = S.device
        self.A = self.A.to(device)
        S = S.squeeze(1)
        batch_size, N_ch = S.shape[:2]
        Ds, Vs = torch.linalg.eigh(S)
        idx = Ds > 0
        Ds = torch.where(idx, Ds, torch.zeros_like(Ds))
        Vs = Vs * torch.sqrt(Ds).unsqueeze(1)
        latent_x = torch.matmul(self.D.conj().T, Vs)
        latent_x = torch.linalg.norm(latent_x, dim=2) ** 2
        latent_x -= self.tau
        latent_x = latent_x.unsqueeze(1)
        latent_x_skip = latent_x
        latent_s1 = self.denoise1(latent_x)
        latent_x  = latent_x_skip.add(latent_s1)
        latent_x  = self.relu(latent_x)
        latent_s2 = self.denoise2(latent_x)
        latent_x  = latent_x_skip.add(latent_s2)
        latent_x  = self.relu(latent_x)
        latent_s3 = self.denoise3(latent_x)
        latent_x  = latent_x_skip.add(latent_s3)
        latent_x  = self.relu(latent_x)
        latent_s4 = self.denoise4(latent_x)
        latent_x  = latent_x_skip.add(latent_s4)
        latent_x  = self.relu(latent_x)
        latent_x  = latent_x.squeeze(1)
        out = torch.einsum('nij,bjk,nkl->bil', self.A.unsqueeze(0),
                         torch.diag_embed(latent_x.cdouble()),
                         self.A.unsqueeze(0).transpose(1, 2).conj())
        return out, latent_x
