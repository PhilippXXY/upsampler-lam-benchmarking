"""Upsampler models for LAM benchmarking."""

from upsampler.ainn import AINNUpsampler
from upsampler.base import TrainableUpsampler
from upsampler.bicubic import BicubicUpsampler
from upsampler.gan import GANUpsampler
from upsampler.imdn import IMDNUpsampler
from upsampler.safmn import SAFMNUpsampler
from upsampler.srcnn import SRCNNUpsampler

__all__ = [
    "AINNUpsampler",
    "TrainableUpsampler",
    "BicubicUpsampler",
    "SRCNNUpsampler",
    "IMDNUpsampler",
    "SAFMNUpsampler",
    "GANUpsampler",
]
