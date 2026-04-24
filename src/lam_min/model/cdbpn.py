"""
Complex Deep Back-Projection Network (CDBPN) for covariance matrix upsampling.

Implementation adapted from the original DBPN repository for complex-valued
microphone array correlation matrix super-resolution. The network uses iterative
up- and down-sampling with dense skip connections to progressively refine
spatial resolution.

The CDBPN processes real and imaginary components separately through parallel
networks before recombining into complex output matrices.

Architecture Components
-----------------------
- DenseBlock : Fully-connected layers with batch normalisation
- ConvBlock : 2D convolution with normalisation and activation
- DeconvBlock : Transposed convolution for upsampling
- ResnetBlock : Residual block for feature refinement
- UpBlock/DownBlock : Iterative projection blocks with error feedback
- Upsampler : Progressive upsampling via pixel shuffle

References
----------
.. [1] DBPN: https://github.com/alterzero/DBPN-Pytorch
.. [2] Roman et al., "UpLAM: Upsampling Latent Acoustic Map"

Notes
-----
This is an inference-only implementation. Training functionality from the
original repository has been removed.
"""

import math
import time
import tracemalloc

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision.transforms import *

from lam_min.util.flops import build_custom_flop_mapping


class DenseBlock(torch.nn.Module):
    """
    Fully-connected dense block with optional normalisation and activation.

    Parameters
    ----------
    input_size : int
        Input feature dimension
    output_size : int
        Output feature dimension
    bias : bool, optional
        Whether to use bias in linear layer (default: True)
    activation : str, optional
        Activation function type (default: 'relu')
        Options: 'relu', 'prelu', 'lrelu', 'tanh', 'sigmoid', or None
    norm : str, optional
        Normalisation type (default: 'batch')
        Options: 'batch', 'instance', or None
    """

    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Apply dense block transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input features

        Returns
        -------
        torch.Tensor
            Transformed features after linear layer, optional normalisation,
            and optional activation
        """
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    """
    2D convolutional block with optional normalisation and activation.

    Parameters
    ----------
    input_size : int
        Number of input channels
    output_size : int
        Number of output channels
    kernel_size : int, optional
        Convolution kernel size (default: 3)
    stride : int, optional
        Convolution stride (default: 1)
    padding : int, optional
        Zero padding (default: 1)
    bias : bool, optional
        Whether to use bias (default: True)
    activation : str, optional
        Activation function type (default: 'prelu')
        Options: 'relu', 'prelu', 'lrelu', 'tanh', 'sigmoid', or None
    norm : str, optional
        Normalisation type (default: None)
        Options: 'batch', 'instance', or None
    """

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation='prelu',
        norm=None
        ):
        """
        Initialize a ConvBlock module with configurable convolution, normalization, and activation.

        This block combines a 2D convolution layer with optional batch/instance normalization
        and various activation functions.
        """
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            input_size, output_size,
            kernel_size, stride, padding,
            bias=bias,
            dtype=torch.double
            )

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(dtype=torch.double)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Apply 2D convolution with optional normalisation and activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_size, height, width)

        Returns
        -------
        torch.Tensor
            Transformed tensor after convolution, optional normalisation,
            and optional activation
        """
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    """
    2D transposed convolutional block for upsampling.

    Parameters
    ----------
    input_size : int
        Number of input channels
    output_size : int
        Number of output channels
    kernel_size : int, optional
        Convolution kernel size (default: 4)
    stride : int, optional
        Convolution stride for upsampling (default: 2)
    padding : int, optional
        Zero padding (default: 1)
    bias : bool, optional
        Whether to use bias (default: True)
    activation : str, optional
        Activation function type (default: 'prelu')
    norm : str, optional
        Normalisation type (default: None)
    """

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=True,
        activation='prelu',
        norm=None
        ):
        """
        Initialise a DeconvBlock with deconvolution, normalization, and activation layers.

        Parameters
        ----------
        input_size: int
            Number of input channels.
        output_size: int
            Number of output channels.
        kernel_size: int, optional
            Size of the convolutional kernel. Defaults to 4.
        stride: int, optional
            Stride of the convolution. Defaults to 2.
        padding: int, optional
            Padding added to the input. Defaults to 1.
        bias: bool, optional
            Whether to use bias in the deconvolution layer. Defaults to True.
        activation: str, optional
            Activation function to apply. Options are 'prelu', 'relu', 'lrelu', 'tanh', 'sigmoid'.
            Defaults to 'prelu'.
        norm: str, optional
            Normalisation layer to apply. Options are 'batch', 'instance', or None for no normalisation.
            Defaults to None.
        """
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(
            input_size, output_size, kernel_size, stride, padding,
            bias=bias, dtype=torch.double)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(dtype=torch.double)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the deconvolution block.

        Applies a deconvolution operation to the input tensor, optionally followed by
        batch normalization and an activation function.

        Parameters
        ----------
        x : torch.Tensor
        Input tensor to be processed through the deconvolution layer.

        Returns
        -------
        torch.Tensor
            Output tensor after deconvolution, batch normalization (if enabled),
            and activation function (if enabled).
        """
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    """
    A residual block module with two convolutional layers, optional normalization, and activation.

    Parameters
    ----------
    num_filter : int
        Number of filters for the convolutional layers.
    kernel_size : int, optional
        Size of the convolutional kernels (default: 3).
    stride : int, optional
        Stride for the convolutional layers (default: 1).
    padding : int, optional
        Padding for the convolutional layers (default: 1).
    bias : bool, optional
        Whether to use bias in the convolutional layers (default: True).
    activation : str, optional
        Activation function to use: 'relu', 'prelu', 'lrelu', 'tanh', or 'sigmoid' (default: 'prelu').
    norm : str or None, optional
        Normalization type: 'batch', 'instance', or None (default: 'batch').
    """

    def __init__(
        self,
        num_filter,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation='prelu',
        norm='batch'
        ):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter,
                                     num_filter,
                                     kernel_size,
                                     stride,
                                     padding,
                                     bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter,
                                     num_filter,
                                     kernel_size,
                                     stride,
                                     padding,
                                     bias=bias)
        """
        Initialises a ResnetBlock module.

        Parameters
        ----------
        num_filter: int
            Number of filters for the convolutional layers.
        kernel_size: int, optional
            Size of the convolutional kernels. Default is 3.
        stride: int, optional
            Stride for the convolutional layers. Default is 1.
        padding: int, optional
            Padding for the convolutional layers. Default is 1.
        bias: bool, optional
            Whether to use bias in the convolutional layers. Default is True.
        activation: str, optional
            Type of activation function to use.
            Options are 'relu', 'prelu', 'lrelu', 'tanh', 'sigmoid'.
            Default is 'prelu'.
        norm: str, optional
            Type of normalization to use. Options are 'batch', 'instance'. Default is 'batch'.
        """
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(dtype=torch.double)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        """
        Apply a residual block with two convolutions and a skip connection.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map with residual added.
        """
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out

class UpBlock(torch.nn.Module):
    """
    A neural network module representing an upsampling block using a combination of deconvolution and convolution layers.

    Parameters
    ----------
    num_filter: int
        Number of filters for each convolutional and deconvolutional layer.
    kernel_size: int, optional
        Size of the convolutional kernels. Default is 8.
    stride: int, optional
        Stride for the convolutional and deconvolutional layers. Default is 4.
    padding: int, optional
        Padding for the convolutional and deconvolutional layers. Default is 2.
    bias: bool, optional
        If True, adds a learnable bias to the layers. Default is True.
    activation: str, optional
        Activation function to use. Default is 'prelu'.
    norm: str or None, optional
        Normalization layer to use. Default is None.

    Attributes
    ----------
    up_conv1: DeconvBlock
        First deconvolutional block.
    up_conv2: ConvBlock
        Convolutional block.
    up_conv3: DeconvBlock
        Second deconvolutional block.

    Forward Input:
        x (torch.Tensor): Input tensor.

    Forward Output:
        torch.Tensor: Output tensor after upsampling and residual connections.
    """
    
    def __init__(self,
                 num_filter,
                 kernel_size=8,
                 stride=4,
                 padding=2,
                 bias=True,
                 activation='prelu',
                 norm=None):
        """
        Initialise an UpBlock module for upsampling operations.

        Parameters
        ----------
        num_filter: int
            Number of filters/channels for the convolutional layers.
        kernel_size: int, optional
            Size of the convolutional kernel. Defaults to 8.
        stride: int, optional
            Stride for the convolutional operations. Defaults to 4.
        padding: int, optional
            Padding for the convolutional operations. Defaults to 2.
        bias: bool, optional
            Whether to use bias in the convolutional layers. Defaults to True.
        activation: str, optional
            Type of activation function to use. Defaults to 'prelu'.
        norm: optional
            Normalization layer to apply. Defaults to None.
        """
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding,
            bias=bias, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding,
            bias=bias, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(
            num_filter, num_filter, kernel_size, stride, padding,
            bias=bias, activation=activation, norm=None)

    def forward(self, x):
        """
        Forward pass of the residual upsampling block.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Output tensor after applying upsampling operations with residual connections.
        """
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class UpBlockPix(torch.nn.Module):
    """
    Upsampling block with pixel shuffle and residual connection.

    This module performs progressive upsampling with a residual learning pathway.
    It combines upsampling, convolution, and residual connections to enhance
    image reconstruction quality.

    Attributes
    ----------
        up_conv1 (Upsampler): First upsampling layer.
        up_conv2 (ConvBlock): Convolutional block for feature refinement.
        up_conv3 (Upsampler): Second upsampling layer for residual pathway.

    Parameters
    ----------
        num_filter (int): Number of convolutional filters.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 8.
        stride (int, optional): Stride for convolution. Defaults to 4.
        padding (int, optional): Padding for convolution. Defaults to 2.
        scale (int, optional): Upsampling scale factor. Defaults to 4.
        bias (bool, optional): Whether to use bias in convolution. Defaults to True.
        activation (str, optional): Activation function type. Defaults to 'prelu'.
        norm (optional): Normalization type. Defaults to None.

    Returns
    -------
        torch.Tensor: Upsampled feature tensor with residual enhancement.
    """

    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        scale=4,
        bias=True,
        activation='prelu',
        norm=None):
        """
        Initialise UpBlockPix layer.
        
        Parameters
        ----------
            num_filter (int): Number of filters/channels for convolutional operations.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 8.
            stride (int, optional): Stride for the convolutional operation. Defaults to 4.
            padding (int, optional): Padding for the convolutional operation. Defaults to 2.
            scale (int, optional): Upsampling scale factor. Defaults to 4.
            bias (bool, optional): Whether to use bias in convolutional layers. Defaults to True.
            activation (str, optional): Type of activation function to use. Defaults to 'prelu'.
            norm (str, optional): Type of normalization to apply. Defaults to None.
        """
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(
            num_filter, num_filter, kernel_size, stride, padding,
            activation, norm=None) # type: ignore
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
        """
        Forward pass of the upsampling block.
        
        Performs a multi-scale upsampling operation using cascaded convolutions.
        The method applies an initial upsampling, followed by a downsampling path,
        and combines the results with residual connections.
        
        Parameters
        ----------
            x: Input tensor to be upsampled.
        
        Returns
        -------
            Upsampled tensor combining the high-frequency details from the first
            convolution with the refined high-frequency information from subsequent
            convolutions.
        """
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_UpBlock(torch.nn.Module):
    """
    Dense Upsampling Block for super-resolution.

    A residual upsampling module that applies a series of deconvolution and convolution
    operations to progressively upsample feature maps. It combines dense connections with
    multi-scale processing to enhance feature representation during upsampling.

    Parameters
    ----------
        num_filter (int): Number of output filters/channels.
        kernel_size (int, optional): Kernel size for deconvolution/convolution operations.
            Defaults to 8.
        stride (int, optional): Stride for deconvolution/convolution operations. Defaults to 4.
        padding (int, optional): Padding for deconvolution/convolution operations. Defaults to 2.
        num_stages (int, optional): Number of stages for input channel calculation. Defaults to 1.
        bias (bool, optional): Whether to use bias in convolution layers. Defaults to True.
        activation (str, optional): Activation function type ('prelu', 'relu',
            etc.). Defaults to 'prelu'.
        norm (str, optional): Normalization layer type. Defaults to None.

    Returns
    -------
        torch.Tensor:
        Upsampled feature tensor combining residual connections and multi-scale features.
    """

    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        bias=True,
        activation='prelu',
        norm=None):
        """
        Initialise the D_UpBlock module.

        Parameters
        ----------
            num_filter (int): Number of filters/channels for the convolutional layers.
            kernel_size (int, optional): Size of the deconvolution kernel. Defaults to 8.
            stride (int, optional): Stride for the deconvolution operation. Defaults to 4.
            padding (int, optional): Padding for the deconvolution operation. Defaults to 2.
            num_stages (int, optional): Number of stages, used to scale input channels. 
                Defaults to 1.
            bias (bool, optional): Whether to use bias in convolutional layers. Defaults to True.
            activation (str, optional): Type of activation function to use. Defaults to 'prelu'.
            norm (str, optional): Type of normalization to apply. Defaults to None.
        """
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None) # type: ignore
        self.up_conv1 = DeconvBlock(num_filter,
                                    num_filter,
                                    kernel_size,
                                    stride,
                                    padding,
                                    activation, # type: ignore
                                    norm=None)
        self.up_conv2 = ConvBlock(num_filter,
                                  num_filter,
                                  kernel_size,
                                  stride,
                                  padding,
                                  activation, # type: ignore
                                  norm=None)
        self.up_conv3 = DeconvBlock(num_filter,
                                    num_filter,
                                    kernel_size,
                                    stride,
                                    padding,
                                    activation, # type: ignore
                                    norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_UpBlockPix(torch.nn.Module):
    """
    Dense pixel-based upsampling block.

    Performs a dense up-projection using pixel-based upsampling (pixel-shuffle)
    with an internal 1x1 projection followed by an upsampling -> conv ->
    upsampling sequence and residual connection.

    Parameters
    ----------
    num_filter : int
        Number of output filters for internal convolutions.
    kernel_size : int, optional
        Kernel size for intermediate convolutions (default: 8).
    stride : int, optional
        Stride for convolutions (default: 4).
    padding : int, optional
        Padding for convolutions (default: 2).
    num_stages : int, optional
        Number of input stages used to scale channels for the 1x1 conv
        (default: 1).
    scale : int, optional
        Upsampling scale factor passed to `Upsampler` (default: 4).
    bias : bool, optional
        Whether convolution layers include a bias term (default: True).
    activation : str, optional
        Activation type used in sub-blocks (default: 'prelu').
    norm : str or None, optional
        Normalisation type for sub-blocks (default: None).

    Returns
    -------
    torch.Tensor
        Upsampled feature tensor with residual addition.
    """

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None) # type: ignore
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
        """
        Compute the forward pass of the pixel-based dense up-block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output feature tensor after upsampling and residual addition.
        """
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class DownBlock(torch.nn.Module):
    """
    Downsampling projection block.

    Implements a down-projection sequence (conv -> deconv -> conv) with a
    residual connection to produce a learned downsampling operation.

    Parameters
    ----------
    num_filter : int
        Number of filters for the internal convolutional layers.
    kernel_size : int, optional
        Kernel size for convolutions (default: 8).
    stride : int, optional
        Stride for convolutions (default: 4).
    padding : int, optional
        Padding for convolutions (default: 2).
    bias : bool, optional
        Whether layers include a bias term (default: True).
    activation : str, optional
        Activation type for sub-blocks (default: 'prelu').
    norm : str or None, optional
        Normalisation type for sub-blocks (default: None).
    """

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore

    def forward(self, x):
        """
        Forward pass for the downsampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be downsampled.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with residual connection applied.
        """
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class DownBlockPix(torch.nn.Module):
    """
    Pixel-based downsampling block.

    Variant of `DownBlock` that uses a pixel-based `Upsampler` in the
    projection path to allow alternative projection strategies while keeping
    residual connections.

    Parameters
    ----------
    num_filter : int
        Number of filters for convolutional layers.
    kernel_size : int, optional
        Kernel size for convolutions (default: 8).
    stride : int, optional
        Stride for convolutions (default: 4).
    padding : int, optional
        Padding for convolutions (default: 2).
    scale : int, optional
        Upsampling scale factor passed to `Upsampler` (default: 4).
    bias : bool, optional
        Whether layers include a bias term (default: True).
    activation : str, optional
        Activation type for sub-blocks (default: 'prelu').
    norm : str or None, optional
        Normalisation type for sub-blocks (default: None).
    """

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='prelu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore

    def forward(self, x):
        """
        Execute the forward pass of the pixel-based down block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after down-projection and residual addition.
        """
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class D_DownBlock(torch.nn.Module):
    """
    Dense downsampling block.

    Performs a dense down-projection using a learned 1x1 projection followed
    by a conv -> deconv -> conv sequence with a residual connection.

    Parameters
    ----------
    num_filter : int
        Number of filters for convolutional layers.
    kernel_size : int, optional
        Kernel size for convolutions (default: 8).
    stride : int, optional
        Stride for convolutions (default: 4).
    padding : int, optional
        Padding for convolutions (default: 2).
    num_stages : int, optional
        Number of stages used to compute the 1x1 projection input channels
        (default: 1).
    bias : bool, optional
        Whether layers include a bias term (default: True).
    activation : str, optional
        Activation used in sub-blocks (default: 'prelu').
    norm : str or None, optional
        Normalisation type for sub-blocks (default: None).
    """

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None) # type: ignore
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None) # type: ignore

    def forward(self, x):
        """
        Forward pass for the dense down-projection block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after dense down-projection and residual addition.
        """
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class D_DownBlockPix(torch.nn.Module):
    """
    Downsampling block with pixelwise operations for super-resolution.

    This module performs a series of downsampling and upsampling convolutions
    with residual connections. It combines convolutional blocks with upsampling
    to create a deep back-projection network component.
    """

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
                 num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None) # type: ignore
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                    padding, activation, norm=None) # type: ignore
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride,
                                    padding, activation, norm=None) # type: ignore

    def forward(self, x):
        """
        Forward pass of the module.

        Parameters
        ----------
            x: Input tensor.

        Returns
        -------
            Tensor: Output tensor computed by downsampling convolutions and residual connections.
        """
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class PSBlock(torch.nn.Module):
    """
    Sub-pixel convolution block that upsamples via PixelShuffle.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    output_size : int
        Number of output channels.
    scale_factor : int
        Upscaling factor for PixelShuffle.
    kernel_size, stride, padding, bias, activation, norm : optional
        Block configuration options.

    Returns
    -------
    torch.Tensor
        Upsampled output.
    """

    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1,
                 padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2,
                                    kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the module.

        Applies convolution, pixel shuffle, optional batch normalization, and optional activation.

        Parameters
        ----------
            x: Input tensor.

        Returns
        -------
        torch.Tensor:
            Output tensor after convolution, pixel shuffle, optional normalization, and optional activation.
        """
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    """
    Upsampling module for increasing spatial dimensions of feature maps.

    This module performs upsampling by a given scale factor using a series of
    convolution blocks followed by pixel shuffle operations. An optional activation
    function can be applied after upsampling.
    """

    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None)) # type: ignore
            modules.append(torch.nn.PixelShuffle(2))
            if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
            #modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the upsampling module.

        Parameters
        ----------
            x: Input tensor to be upsampled.

        Returns
        -------
        torch.Tensor:
            Upsampled tensor, optionally passed through an activation function.
        """
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    """
    Upsample by 2 using deconv, pixel-shuffle, or resize+conv methods.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    output_size : int
        Number of output channels.
    bias, upsample, activation, norm : optional
        Configuration options for the upsampling method.

    Returns
    -------
    torch.Tensor
        Upsampled tensor.
    """

    def __init__(self, input_size, output_size, bias=True,
                 upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        """
        Forward pass of the upsampling module.

        Parameters
        ----------
        x:
            Input tensor to be upsampled.

        Returns
        -------
        torch.Tensor
            Upsampled output tensor.
        """
        out = self.upsample(x)
        return out


# Complex DBPN (CDBPN) network
class Net(nn.Module):
    """
    Complex Deep Back-Projection Network for covariance matrix upsampling.

    Implements a complex-valued super-resolution network using iterative
    up- and down-sampling projections with dense connections. Processes real
    and imaginary components in parallel through symmetric network branches.

    Parameters
    ----------
    num_channels : int
        Number of input/output channels (frequency bands)
    base_filter : int
        Number of base filters in projection blocks
    feat : int
        Feature dimension for initial extraction layers
    num_stages : int
        Number of back-projection stages (controls output concatenation)
    scale_factor : int
        Spatial upsampling factor (2, 4, or 8)

    Attributes
    ----------
    feat0_rel, feat0_imag : ConvBlock
        Initial feature extraction layers for real and imaginary components
    feat1_rel, feat1_imag : ConvBlock
        Secondary feature extraction reducing to base_filter dimension
    up1_rel, up1_imag : UpBlock
        First upsampling projection blocks
    down1_rel, down1_imag : DownBlock
        First downsampling projection blocks
    up2_rel, up2_imag : UpBlock
        Second upsampling projection blocks
    output_conv_rel, output_conv_imag : ConvBlock
        Final reconstruction convolutions

    References
    ----------
    .. [1] Haris et al., "Deep Back-Projection Networks for Super-Resolution"
    """

    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor) -> None:
        """
        Initialise the CDBPN (Convolutional Dense Back-Projection Network) model.

        Parameters
        ----------
        num_channels: int
            Number of input/output channels.
        base_filter: int
            Number of base filters for convolutional layers.
        feat: int
            Number of feature channels for initial feature extraction.
        num_stages: int
            Number of back-projection stages in the network.
        scale_factor: int
            Upsampling scale factor (2, 4, or 8).
        """
        super(Net, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        else:
            kernel = 8
            stride = 4
            padding = 2

        #Initial Feature Extraction
        self.feat0_rel = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat0_imag = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1_rel = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.feat1_imag = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1_rel = UpBlock(base_filter, kernel, stride, padding)
        self.up1_imag = UpBlock(base_filter, kernel, stride, padding)
        self.down1_rel = DownBlock(base_filter, kernel, stride, padding)
        self.down1_imag = DownBlock(base_filter, kernel, stride, padding)
        self.up2_rel = UpBlock(base_filter, kernel, stride, padding)
        self.up2_imag = UpBlock(base_filter, kernel, stride, padding)
        #Reconstruction
        self.output_conv_rel = ConvBlock(num_stages*base_filter // 5, num_channels, 3, 1, 1, activation=None, norm=None) # type: ignore
        self.output_conv_imag = ConvBlock(num_stages*base_filter // 5, num_channels, 3, 1, 1, activation=None, norm=None) # type: ignore

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight).double() # type: ignore
                if m.bias is not None:
                    m.bias.data.zero_() # type: ignore
                    m.bias.data = m.bias.data.double() # type: ignore
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight).double() # type: ignore
                if m.bias is not None:
                    m.bias.data = m.bias.data.double() # type: ignore

    def forward(self, x_rel, x_imag, collect_metrics=False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass: upsample complex covariance matrices.

        Processes real and imaginary components through parallel networks
        with iterative up-down projection stages.

        Parameters
        ----------
        x_rel : torch.Tensor
            Real component of input covariance matrices
            Shape: (batch, num_channels, height, width)
        x_imag : torch.Tensor
            Imaginary component of input covariance matrices
            Shape: (batch, num_channels, height, width)
        collect_metrics : bool, optional
            If True, collect and return performance metrics (default: False)

        Returns
        -------
        torch.Tensor
            Upsampled complex covariance matrices
            Shape: (batch, num_channels, height*scale_factor, width*scale_factor)
            Complex-valued tensor combining real and imaginary outputs
        metrics : dict, optional
            Performance metrics (only returned if collect_metrics=True)
            Contains: upsampler_time_ms, upsampler_feature_extraction_time_ms,
            upsampler_back_projection_time_ms, upsampler_reconstruction_time_ms

        Notes
        -----
        Processing pipeline:
        1. Initial feature extraction (feat0, feat1) for both components
        2. First upsampling (up1) projection
        3. Downsampling (down1) followed by second upsampling (up2)
        4. Concatenate intermediate features (h1, h2)
        5. Final reconstruction convolution
        6. Combine real and imaginary components into complex tensor
        """
        metrics = {}
        device = x_rel.device
        use_cuda = device.type == "cuda"
        
        # Store original inputs for FLOPs/memory measurement
        if collect_metrics:
            x_rel_orig = x_rel.clone()
            x_imag_orig = x_imag.clone()

        # Feature Extraction Stage
        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            feat_start = time.perf_counter()

        #real
        x_rel = self.feat0_rel(x_rel)
        x_rel = self.feat1_rel(x_rel)
        #imag
        x_imag = self.feat0_imag(x_imag)
        x_imag = self.feat1_imag(x_imag)

        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            feat_end = time.perf_counter()
            metrics['upsampler_feature_extraction_time_ms'] = (feat_end - feat_start) * 1000.0 # type: ignore

        # ===== BACK-PROJECTION STAGE =====
        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            bp_start = time.perf_counter()

        #real
        h1_rel = self.up1_rel(x_rel)
        h2_rel = self.up2_rel(self.down1_rel(h1_rel))
        #imag
        h1_imag = self.up1_imag(x_imag)
        h2_imag = self.up2_imag(self.down1_imag(h1_imag))

        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            bp_end = time.perf_counter()
            metrics['upsampler_back_projection_time_ms'] = (bp_end - bp_start) * 1000.0 # type: ignore

        # ===== RECONSTRUCTION STAGE =====
        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            recon_start = time.perf_counter()

        #real
        x_rel = self.output_conv_rel(torch.cat((h2_rel, h1_rel),1))
        #imag
        x_imag = self.output_conv_imag(torch.cat((h2_imag, h1_imag),1))
        result = torch.complex(x_rel, x_imag)

        if collect_metrics:
            if use_cuda:
                torch.cuda.synchronize(device)
            recon_end = time.perf_counter()
            metrics['upsampler_reconstruction_time_ms'] = (recon_end - recon_start) * 1000.0 # type: ignore
            metrics['upsampler_time_ms'] = (recon_end - feat_start) * 1000.0 # type: ignore
            
            num_frames = result.shape[0]
            metrics['num_frames'] = num_frames
            
            # FLOPs measurement
            flop_counter = FlopCounterMode(
                display=False,
                custom_mapping=build_custom_flop_mapping(),
            )
            with torch.no_grad():
                with flop_counter:
                    self._forward_no_metrics(x_rel_orig, x_imag_orig) # type: ignore
            metrics['upsampler_flops'] = flop_counter.get_total_flops()
            metrics['upsampler_flops_per_frame'] = metrics['upsampler_flops'] / num_frames if num_frames > 0 else 0
            
            # Memory measurement
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    self._forward_no_metrics(x_rel_orig, x_imag_orig) # type: ignore
                metrics['upsampler_memory_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            else:
                tracemalloc.start()
                try:
                    with torch.no_grad():
                        self._forward_no_metrics(x_rel_orig, x_imag_orig) # type: ignore
                    _, peak = tracemalloc.get_traced_memory()
                    metrics['upsampler_memory_mb'] = peak / (1024 * 1024)
                finally:
                    tracemalloc.stop()
            
            return result, metrics

        return result

    def _forward_no_metrics(self, x_rel, x_imag):
        x_rel = self.feat0_rel(x_rel)
        x_rel = self.feat1_rel(x_rel)
        x_imag = self.feat0_imag(x_imag)
        x_imag = self.feat1_imag(x_imag)
        h1_rel = self.up1_rel(x_rel)
        h2_rel = self.up2_rel(self.down1_rel(h1_rel))
        h1_imag = self.up1_imag(x_imag)
        h2_imag = self.up2_imag(self.down1_imag(h1_imag))
        x_rel = self.output_conv_rel(torch.cat((h2_rel, h1_rel), 1))
        x_imag = self.output_conv_imag(torch.cat((h2_imag, h1_imag), 1))
        return torch.complex(x_rel, x_imag)
