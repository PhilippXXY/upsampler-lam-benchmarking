"""Core SAFMN building blocks adapted for CSM super-resolution."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_f
from torch import nn


class LayerNorm(nn.Module):
    """
    Layer normalisation with channel-first support.

    Parameters
    ----------
    normalized_shape : int
        Number of channels to normalise.
    eps : float, optional
        Numerical stability epsilon (default: 1e-6).
    data_format : str, optional
        Input layout, either "channels_first" or "channels_last"
        (default: "channels_first").
    """

    def __init__(
        self,
        normalised_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_first",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalised_shape))
        self.bias = nn.Parameter(torch.zeros(normalised_shape))
        self.eps = float(eps)
        self.data_format = data_format
        if self.data_format not in ["channels_first", "channels_last"]:
            raise ValueError(
                "data_format must be one of ['channels_first', 'channels_last'], "
                f"got '{self.data_format}'."
            )
        self.normalized_shape = (int(normalised_shape),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor with same shape as input.
        """
        if self.data_format == "channels_last":
            return torch_f.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


class CCM(nn.Module):
    """
    Convolutional Channel Mixer block from SAFMN.

    Parameters
    ----------
    dim : int
        Number of input and output channels.
    growth_rate : float, optional
        Expansion factor for the hidden channel width (default: 2.0).
    """

    def __init__(self, dim: int, growth_rate: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        if hidden_dim <= 0:
            raise ValueError(
                f"CCM hidden dimension must be > 0. Got dim={dim}, growth_rate={growth_rate}."
            )

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, dtype=torch.float32),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel mixing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, channels, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor with same shape as input.
        """
        return self.ccm(x)  # type: ignore[no-any-return]


class SAFM(nn.Module):
    """
    Spatially-Adaptive Feature Modulation block.

    Parameters
    ----------
    dim : int
        Number of input/output channels.
    n_levels : int, optional
        Number of multi-scale channel chunks (default: 4).
    """

    def __init__(self, dim: int, n_levels: int = 4) -> None:
        super().__init__()

        self.n_levels = n_levels
        if self.n_levels <= 0:
            raise ValueError("n_levels must be > 0.")
        if dim <= 0:
            raise ValueError("dim must be > 0.")
        chunk_dim = dim // self.n_levels

        # Spatial weighting
        self.mfr = nn.ModuleList(
            [
                nn.Conv2d(
                    chunk_dim,
                    chunk_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=chunk_dim,
                    dtype=torch.float32,
                )
                for _ in range(self.n_levels)
            ]
        )

        # Feature aggregation
        self.aggr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dtype=torch.float32)

        # Activation
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale spatial modulation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, channels, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor with same shape as input.
        """
        height, width = x.shape[-2:]
        chunks = x.chunk(self.n_levels, dim=1)

        out: list[torch.Tensor] = []
        for level in range(self.n_levels):
            if level == 0:
                scaled = self.mfr[level](chunks[level])
            else:
                pool_h = max(height // (2**level), 1)
                pool_w = max(width // (2**level), 1)
                scaled = torch_f.adaptive_max_pool2d(chunks[level], output_size=(pool_h, pool_w))
                scaled = self.mfr[level](scaled)
                scaled = torch_f.interpolate(scaled, size=(height, width), mode="nearest")
            out.append(scaled)

        mod = self.aggr(torch.cat(out, dim=1))
        return self.act(mod) * x  # type: ignore[no-any-return]


class AttBlock(nn.Module):
    """
    SAFMN feature mixing block (SAFM + CCM with residual links).

    Parameters
    ----------
    dim : int
        Feature channel width.
    ffn_scale : float, optional
        Expansion ratio used by CCM (default: 2.0).
    n_levels : int, optional
        Number of SAFM multi-scale levels (default: 4).
    """

    def __init__(self, dim: int, ffn_scale: float = 2.0, n_levels: int = 4) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        # Multiscale block
        self.safm = SAFM(dim=dim, n_levels=n_levels)
        # Feedforward layer
        self.ccm = CCM(dim=dim, growth_rate=ffn_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one feature mixing block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, channels, height, width].

        Returns
        -------
        torch.Tensor
            Output tensor with same shape as input.
        """
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x
