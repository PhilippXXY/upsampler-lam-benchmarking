"""Super-Resolution Convolutional Neural Network (SRCNN) upsampler."""

from upsampler.srcnn.model import SRCNNUpsampler
from upsampler.srcnn.variable import VariableSRCNNUpsampler

__all__ = ["SRCNNUpsampler", "VariableSRCNNUpsampler"]
