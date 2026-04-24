"""Training-only LAM losses."""

from __future__ import annotations

import torch
from scipy.spatial import KDTree
from torch import nn

from lam_min.trainer.utils import get_field


class ComplexMSELoss(nn.Module):
    """Mean Squared Error loss for complex tensors, computed separately on real and imag parts."""

    def __init__(self) -> None:
        """Initialise the Complex MSE Loss module."""
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute complex MSE on the real and imaginary parts separately.

        Parameters
        ----------
        target : torch.Tensor
            Target complex tensor.
        pred : torch.Tensor
            Predicted complex tensor.

        Returns
        -------
        torch.Tensor
            Scalar complex MSE value.
        """
        if not target.is_complex() or not pred.is_complex():
            raise ValueError("Both tensors must be complex")

        target_real = target.real
        target_imag = target.imag
        pred_real = pred.real
        pred_imag = pred.imag

        mse_loss_real = self.mse_loss(target_real, pred_real)
        mse_loss_imag = self.mse_loss(target_imag, pred_imag)

        total_loss = mse_loss_real + mse_loss_imag
        return total_loss


class MSETVLoss(nn.Module):
    """MSE + TV regularisation."""

    def __init__(
        self,
        l1_weight: float = 0.0001,
        device: str = "cuda:0",
        reg: str = "l1",
    ) -> None:
        """
        Initialise the MSE + TV loss module.

        Parameters
        ----------
        l1_weight : float, optional
            Weight for the TV regularisation term, by default 0.0001.
        device : str, optional
            Device to run the loss computation on, by default "cuda:0".
        reg : str, optional
            Type of regularisation to apply ("l1", "l2", or "tv_only"), by default "l1".
        """
        super().__init__()
        self.mse_loss = ComplexMSELoss()
        self.l1_weight = float(l1_weight)
        self.R = get_field()
        kdtree = KDTree(self.R.T)
        self.num_neighbors = 6
        _, indices = kdtree.query(self.R.T, k=self.num_neighbors + 1)
        self.indices = indices
        self.reg = reg
        self.device = device

    def total_variation_fibonacci_lossi(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute the Total Variation (TV) loss for a Fibonacci grid considering multiple neighbors.

        Parameters
        ----------
        latent : torch.Tensor
            Latent tensor of shape ``(batch, n_points)``.

        Returns
        -------
        torch.Tensor
            The TV loss for the latent space.
        """
        tv_loss = 0
        for neighbour_index in range(1, self.num_neighbors + 1):
            diff = latent[:, self.indices[:, 0]] - latent[:, self.indices[:, neighbour_index]]
            tv_loss += torch.sum(torch.abs(diff))
        return tv_loss

    def forward(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the loss function.

        Computes the combined loss as the sum of MSE and TV regularisation,
        where the TV term is weighted by `l1_weight`.

        Parameters
        ----------
        target : torch.Tensor
            Ground truth matrix.
        pred : torch.Tensor
            Predicted complex tensor.
        latent : torch.Tensor
            Latent tensor used for TV regularisation.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Combined loss, reconstruction loss, and TV term.
        """
        mse_loss = self.mse_loss(target, pred)

        if self.reg == "l1":
            tv_loss = self.l1_weight * (
                self.total_variation_fibonacci_lossi(latent) + torch.norm(latent, p=1)
            )
        elif self.reg == "l2":
            tv_loss = self.l1_weight * (
                self.total_variation_fibonacci_lossi(latent) + torch.norm(latent, p=2) ** 2
            )
        else:
            tv_loss = self.l1_weight * self.total_variation_fibonacci_lossi(latent)

        combined_loss = mse_loss + tv_loss

        return combined_loss, mse_loss, tv_loss
