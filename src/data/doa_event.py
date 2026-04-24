"""
Module for Direction of Arrival (DoA) event data structures.

This module provides data classes to represent DoA events with information
about sound source direction, timing, and distance.
"""

from dataclasses import dataclass

import torch


@dataclass
class DoaEvent:
    """
    Data class representing Direction of Arrival (DoA) events.

    Attributes
    ----------
    frame : torch.Tensor
        Frame identifier or index for the DoA event.
        Store as `int64`.
    t_sec : torch.Tensor
        Timestamp of the event in seconds.
        Store as `float32`.
    active_class_index : torch.Tensor
        Index of the active sound class/category for this event.
        Store as `int64`.
    source_number_index : torch.Tensor
        Index or identifier of the sound source.
        Store as `int64`.
    azimuth : torch.Tensor
        Azimuth angle in degrees, indicating horizontal direction of arrival.
        Store as `int64`.
    elevation : torch.Tensor
        Elevation angle in degrees, indicating vertical direction of arrival.
        Store as `int64`.
    distance_cm : torch.Tensor
        Distance to the sound source in centimeters.
        Store as `int64`.

    """

    frame: torch.Tensor
    t_sec: torch.Tensor
    active_class_index: torch.Tensor
    source_number_index: torch.Tensor
    azimuth: torch.Tensor
    elevation: torch.Tensor
    distance_cm: torch.Tensor
