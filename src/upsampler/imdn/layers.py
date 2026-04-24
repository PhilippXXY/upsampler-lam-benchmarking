"""Building blocks for the IMDN upsampler."""

from __future__ import annotations

import torch
from torch import nn


def conv_block(  # noqa: C901, PLR0912, PLR0913
    input_channel: int,
    output_channel: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    pad_type: str = "zero",
    norm_type: str | None = None,
    act_type: str | None = "relu",
) -> nn.Sequential:
    """
    Create a convolution block with optional padding, norm, and activation.

    Parameters
    ----------
    input_channel : int
        Number of input channels.
    output_channel : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default: 1).
    dilation : int, optional
        Dilation factor for the convolution (default: 1).
    groups : int, optional
        Group count for grouped convolution (default: 1).
    bias : bool, optional
        Whether the convolution includes a bias term (default: True).
    pad_type : str, optional
        Padding mode: "zero", "reflect", "replicate" (default: "zero").
    norm_type : str | None, optional
        Normalisation layer: "batch", "instance", or None (default: None).
    act_type : str | None, optional
        Activation: "relu", "lrelu", "prelu", or None (default: "relu").

    Returns
    -------
    nn.Sequential
        Convolution block in execution order.
    """
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

    pad = pad_type.lower()
    if padding == 0:
        pad_layer: nn.Module | None = None
    elif pad == "reflect":
        pad_layer = nn.ReflectionPad2d(padding)
        padding = 0
    elif pad == "replicate":
        pad_layer = nn.ReplicationPad2d(padding)
        padding = 0
    elif pad == "zero":
        pad_layer = None
    else:
        raise ValueError(f"Unsupported pad_type '{pad_type}'.")

    conv = nn.Conv2d(
        in_channels=input_channel,
        out_channels=output_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )

    norm_layer: nn.Module | None = None
    if norm_type is not None:
        norm = norm_type.lower()
        if norm == "batch":
            norm_layer = nn.BatchNorm2d(output_channel, affine=True)
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d(output_channel, affine=False)
        else:
            raise ValueError(f"Unsupported norm_type '{norm_type}'.")

    activation: nn.Module | None = None
    if act_type is not None:
        act = act_type.lower()
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "lrelu":
            activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        elif act == "prelu":
            activation = nn.PReLU(num_parameters=1, init=0.05)
        else:
            raise ValueError(f"Unsupported act_type '{act_type}'.")

    modules: list[nn.Module] = []
    for module in (pad_layer, conv, norm_layer, activation):
        if module is not None:
            modules.append(module)
    return nn.Sequential(*modules)


class CCALayer(nn.Module):
    """
    Contrast-aware Channel Attention (CCA) module from Hui et al. (2019).

    Attributes
    ----------
    channel : int
        Number of input channels.
    reduction : int, optional
        Reduction factor for the channel attention (default: 16).
    """

    def __init__(self, channel: int, reduction: int = 16) -> None:
        """
        Contrast-aware Channel Attention (CCA) module from Hui et al. (2019).

        Parameters
        ----------
        channel : int
            Number of input channels.
        reduction : int, optional
            Reduction factor for the channel attention (default: 16).
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(
                in_channels=channel,
                out_channels=channel // reduction,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channel // reduction,
                out_channels=channel,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CCA module.

        Computes the contrast-aware channel attention and applies it to the input feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (batch_size, channel, height, width).

        Returns
        -------
        torch.Tensor
            Output feature map after applying contrast-aware channel attention,
            with the same shape as the input.
        """
        input_tensor_dim_expected = 4
        if x.dim() != input_tensor_dim_expected:
            raise ValueError(
                "Input tensor must have 4 dimensions (batch_size, channel, height, width)"
            )

        mean_channels = x.sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))

        variance = ((x - mean_channels) ** 2).sum(3, keepdim=True).sum(2, keepdim=True) / (
            x.size(2) * x.size(3)
        )

        std = variance.pow(0.5)

        y = std + self.avg_pool(x)

        y = self.conv_du(y)
        return x * y  # type: ignore[no-any-return]


class IMDModule(nn.Module):
    """
    Information Multi-Distillation Module (IMDModule) from Hui et al. (2019).

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    distillation_rate : float, optional
        Fraction of channels to distill at each step (default: 0.25).
    """

    def __init__(self, in_channels: int, distillation_rate: float = 0.25) -> None:
        """
        Information Multi-Distillation Module (IMDModule) from Hui et al. (2019).

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        distillation_rate : float, optional
            Fraction of channels to distill at each step (default: 0.25).
        """
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dilation=1,
            groups=1,
        )
        self.c2 = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dilation=1,
            groups=1,
        )
        self.c3 = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dilation=1,
            groups=1,
        )
        self.c4 = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.distilled_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            dilation=1,
            groups=1,
        )
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        )
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the IMDModule.

        It performs multi-distillation of the input feature map and applies contrast-aware channel
        attention.

        Parameters
        ----------
        input : torch.Tensor
            Input feature map of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (batch_size, in_channels, height, width) after applying
            the IMDModule.
        """
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(
            tensor=out_c1,
            split_size_or_sections=[self.distilled_channels, self.remaining_channels],
            dim=1,
        )
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(
            tensor=out_c2,
            split_size_or_sections=[self.distilled_channels, self.remaining_channels],
            dim=1,
        )
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(
            tensor=out_c3,
            split_size_or_sections=[self.distilled_channels, self.remaining_channels],
            dim=1,
        )
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused  # type: ignore[no-any-return]
