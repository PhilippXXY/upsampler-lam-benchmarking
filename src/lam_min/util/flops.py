"""
Custom FLOP formulas for inference accounting in LAM-based models.

Purpose
-------
PyTorch's default ``FlopCounterMode`` only includes formulas for a limited
set of operators. In this project, that can under-report FLOPs for:
- complex-valued matrix products
- linear algebra routines such as Hermitian eigendecomposition
- lightweight pointwise operations that appear frequently in the forward pass

This module provides a small custom mapping to improve coverage whilst keeping
the implementation simple and inexpensive.

Scope and Limitations
---------------------
- These formulas are intended for benchmarking consistency, not exact hardware
  instruction counts.
- Some operations are modelled coarsely by design (for example ``linalg_eigh``).
- Complex-valued operations are scaled with pragmatic factors to avoid severe
  under-counting relative to real-valued paths.

References
----------
- PyTorch FLOP registry used by ``FlopCounterMode``:
  https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L580-L602
- Dispatch path where FLOPs are counted only for registered operator packets:
  https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L778-L785
- Raw-callback behaviour (``_get_raw`` handling in custom mapping):
  https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L677-L680
- Built-in GEMM formula conventions (``mm``, ``bmm``, ``addmm``):
  https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L58-L88
"""

from __future__ import annotations

from math import prod
from typing import Any

import torch


def _mark_raw(fn):
    """
    Mark a custom FLOP function as a raw callback.

    Parameters
    ----------
    fn : callable
        FLOP function to register with ``FlopCounterMode``.

    Returns
    -------
    callable
        The same function, annotated with ``_get_raw = True`` so the PyTorch
        FLOP counter passes tensors instead of shape tuples.

    Notes
    -----
    Raw callbacks are useful when FLOP estimation depends on dtype details,
    such as distinguishing real from complex tensors.

    Source
    ------
    PyTorch applies shape wrapping unless ``_get_raw`` is set:
    https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/flop_counter.py#L677-L680
    """
    fn._get_raw = True
    return fn


def _tensor_numel(value: Any) -> int:
    """
    Count the number of elements in a nested tensor-like structure.

    Parameters
    ----------
    value : Any
        A tensor, or a tuple/list containing tensors, or any other object.

    Returns
    -------
    int
        Total number of elements across all tensors found in ``value``.
        Non-tensor objects contribute ``0``.
    """
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_numel(item) for item in value)
    return 0


def _is_complex_tensor(value: Any) -> bool:
    """
    Check whether a value is a complex-valued PyTorch tensor.

    Parameters
    ----------
    value : Any
        Object to inspect.

    Returns
    -------
    bool
        ``True`` if ``value`` is a tensor with complex dtype, else ``False``.
    """
    return isinstance(value, torch.Tensor) and value.is_complex()


def _complex_factor(*values: Any) -> int:
    """
    Return a scaling factor for complex arithmetic.

    Parameters
    ----------
    *values : Any
        Values inspected for complex tensor dtypes.

    Returns
    -------
    int
        ``4`` if any input is complex-valued, otherwise ``1``.

    Notes
    -----
    This is a practical approximation. A complex multiply-accumulate generally
    costs multiple real floating-point operations, so this factor reduces
    systematic under-counting when benchmarking complex paths.
    """
    return 4 if any(_is_complex_tensor(value) for value in values) else 1


@_mark_raw
def mm_flops_raw(a: torch.Tensor, b: torch.Tensor, *args: Any, **kwargs: Any) -> int:
    """
    Estimate FLOPs for matrix multiplication (``aten.mm``).

    Parameters
    ----------
    a : torch.Tensor
        Left matrix with shape ``(m, k)``.
    b : torch.Tensor
        Right matrix with shape ``(k, n)``.
    *args : Any
        Unused extra positional arguments required by callback signature.
    **kwargs : Any
        Unused extra keyword arguments required by callback signature.

    Returns
    -------
    int
        Estimated FLOP count. Returns ``0`` if tensor ranks or inner
        dimensions are incompatible.

    Notes
    -----
    Uses the standard dense GEMM estimate ``m * n * 2 * k``, then applies
    a complex scaling factor where relevant.
    """
    a_dim_expected = b_dim_expected = 2
    if a.ndim != a_dim_expected or b.ndim != b_dim_expected:
        return 0
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        return 0
    return int(m * n * 2 * k * _complex_factor(a, b))


@_mark_raw
def bmm_flops_raw(a: torch.Tensor, b: torch.Tensor, *args: Any, **kwargs: Any) -> int:
    """
    Estimate FLOPs for batched matrix multiplication (``aten.bmm``).

    Parameters
    ----------
    a : torch.Tensor
        Left batch of matrices with shape ``(batch, m, k)``.
    b : torch.Tensor
        Right batch of matrices with shape ``(batch, k, n)``.
    *args : Any
        Unused extra positional arguments required by callback signature.
    **kwargs : Any
        Unused extra keyword arguments required by callback signature.

    Returns
    -------
    int
        Estimated FLOP count. Returns ``0`` if tensor ranks or dimensions are
        incompatible.
    """
    a_dim_expected = b_dim_expected = 3
    if a.ndim != a_dim_expected or b.ndim != b_dim_expected:
        return 0
    batch, m, k = a.shape
    batch2, k2, n = b.shape
    if batch != batch2 or k != k2:
        return 0
    return int(batch * m * n * 2 * k * _complex_factor(a, b))


@_mark_raw
def addmm_flops_raw(
    self: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *args: Any, **kwargs: Any
) -> int:
    """
    Estimate FLOPs for addmm by reusing matrix-multiply cost.

    Parameters
    ----------
    self : torch.Tensor
        Bias/addition tensor from ``aten.addmm``. Not used in this estimate.
    a : torch.Tensor
        Left matrix with shape ``(m, k)``.
    b : torch.Tensor
        Right matrix with shape ``(k, n)``.
    *args : Any
        Unused extra positional arguments required by callback signature.
    **kwargs : Any
        Unused extra keyword arguments required by callback signature.

    Returns
    -------
    int
        Estimated FLOP count for the matrix multiplication component.

    Notes
    -----
    The addition term is intentionally omitted to match the same convention as
    PyTorch's built-in ``addmm`` formula.
    """
    return mm_flops_raw(a, b)


@_mark_raw
def linalg_eigh_flops_raw(a: torch.Tensor, *args: Any, **kwargs: Any) -> int:
    """
    Approximate FLOPs for Hermitian eigendecomposition (``aten.linalg_eigh``).

    Parameters
    ----------
    a : torch.Tensor
        Input matrix or batched matrices with trailing shape ``(..., n, n)``.
    *args : Any
        Unused extra positional arguments required by callback signature.
    **kwargs : Any
        Unused extra keyword arguments required by callback signature.

    Returns
    -------
    int
        Coarse FLOP estimate across the batch.

    Notes
    -----
    This approximation is intentionally simple and mainly prevents severe
    under-counting in LAM, where eigendecomposition is a dominant operation.
    Real and complex inputs use different coarse coefficients.
    """
    a_dim_expected = 2
    if a.ndim < a_dim_expected:
        return 0
    n = int(a.shape[-1])
    if n <= 0:
        return 0
    batch = int(prod(a.shape[:-2])) if a.ndim > a_dim_expected else 1
    # Coarse scaling constants (complex > real).
    coeff = 16 if a.is_complex() else 8
    return int(batch * coeff * (n**3))


@_mark_raw
def linalg_vector_norm_flops_raw(a: torch.Tensor, *args: Any, **kwargs: Any) -> int:
    """
    Approximate FLOPs for vector norm operations.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor.
    *args : Any
        Unused extra positional arguments required by callback signature.
    **kwargs : Any
        Unused extra keyword arguments required by callback signature.

    Returns
    -------
    int
        Coarse estimate for reduction and elementwise work.
    """
    coeff = 4 if a.is_complex() else 2
    return int(a.numel() * coeff)


@_mark_raw
def pointwise_out_flops_raw(*args: Any, out_val: Any = None, **kwargs: Any) -> int:
    """
    Estimate FLOPs for simple pointwise operators.

    Parameters
    ----------
    *args : Any
        Unused extra positional arguments required by callback signature.
    out_val : Any, optional
        Output tensor (or nested tensors) from the operator.
    **kwargs : Any
        Unused extra keyword arguments required by callback signature.

    Returns
    -------
    int
        One FLOP per output element, summed over all output tensors.

    Notes
    -----
    This deliberately under-models complex pointwise behaviour but gives a
    stable and cheap baseline contribution for operations otherwise not counted.
    """
    return _tensor_numel(out_val)


def build_custom_flop_mapping() -> dict[Any, Any]:
    """
    Build the custom operator-to-formula mapping for ``FlopCounterMode``.

    Returns
    -------
    dict[Any, Any]
        Mapping from ``torch.ops.aten`` operator packets to FLOP callback
        functions. Only operators available in the local PyTorch build are
        included.

    Notes
    -----
    The map extends default counting coverage with:
    - complex-aware GEMM variants
    - selected linear algebra operations used by LAM
    - common pointwise operations used in both LAM and CDBPN
    """
    mapping: dict[Any, Any] = {}
    aten = torch.ops.aten

    if hasattr(aten, "mm"):
        mapping[aten.mm] = mm_flops_raw
    if hasattr(aten, "bmm"):
        mapping[aten.bmm] = bmm_flops_raw
    if hasattr(aten, "addmm"):
        mapping[aten.addmm] = addmm_flops_raw
    if hasattr(aten, "linalg_eigh"):
        mapping[aten.linalg_eigh] = linalg_eigh_flops_raw
    if hasattr(aten, "linalg_vector_norm"):
        mapping[aten.linalg_vector_norm] = linalg_vector_norm_flops_raw

    for op_name in ("add", "sub", "mul", "relu", "prelu", "sqrt", "where", "gt", "abs"):
        op = getattr(aten, op_name, None)
        if op is not None:
            mapping[op] = pointwise_out_flops_raw

    return mapping
