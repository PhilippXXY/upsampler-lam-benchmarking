"""Steady-state runtime measurement helpers for inference benchmarking."""

from __future__ import annotations

import statistics
import threading
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def _measurement_count(value: Any, *, default: int, minimum: int = 0) -> int:
    """
    Resolve an integer measurement count from config-like input.

    Parameters
    ----------
    value : Any
        The input value to resolve, which may be of any type.
    default : int
        The default count to use if the input value is invalid or not provided.
    minimum : int, optional
        The minimum allowed count to enforce non-negativity and avoid invalid measurements.

    Returns
    -------
    int
        The resolved measurement count, which is guaranteed to be an integer greater than or
        equal to `minimum`.
    """
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = default
    return max(minimum, resolved)


def _measurement_interval_seconds(value: Any, *, default_ms: float) -> float:
    """
    Resolve a polling interval in seconds from milliseconds input.

    Parameters
    ----------
    value : Any
        The input value to resolve, which may be of any type.
    default_ms : float
        The default interval in milliseconds to use if the input value is invalid or not provided.

    Returns
    -------
    float
        The resolved interval in seconds, which is guaranteed to be a positive float.
    """
    try:
        resolved_ms = float(value)
    except (TypeError, ValueError):
        resolved_ms = default_ms
    return max(1.0e-4, resolved_ms / 1000.0)


def _runtime_measurement_config(inference_config: dict[str, Any]) -> dict[str, int | float]:
    """
    Read optional steady-state runtime measurement settings.

    Parameters
    ----------
    inference_config : dict[str, Any]
        The inference configuration dictionary, which may contain optional keys for runtime
        measurement settings. Supported keys include:
        - "latency_warmup_runs": Number of warmup runs before latency measurement (default: 0).
        - "latency_measurement_runs": Number of runs to measure latency (default: 0).
        - "memory_warmup_runs": Number of warmup runs before memory measurement (default: 0).
        - "memory_measurement_runs": Number of runs to measure memory (default: 0).
        - "memory_poll_interval_ms": Polling interval in milliseconds for memory measurement
        (default: 1.0 ms).

    Returns
    -------
    dict[str, int | float]
        A dictionary containing the resolved runtime measurement configuration with the following
        keys:
        - "latency_warmup_runs": Resolved number of warmup runs for latency measurement.
        - "latency_measurement_runs": Resolved number of runs for latency measurement.
        - "memory_warmup_runs": Resolved number of warmup runs for memory measurement.
        - "memory_measurement_runs": Resolved number of runs for memory measurement.
        - "memory_poll_interval_s": Resolved polling interval in seconds for memory measurement.
    """
    return {
        "latency_warmup_runs": _measurement_count(
            inference_config.get("latency_warmup_runs"),
            default=0,
        ),
        "latency_measurement_runs": _measurement_count(
            inference_config.get("latency_measurement_runs"),
            default=0,
        ),
        "memory_warmup_runs": _measurement_count(
            inference_config.get("memory_warmup_runs"),
            default=0,
        ),
        "memory_measurement_runs": _measurement_count(
            inference_config.get("memory_measurement_runs"),
            default=0,
        ),
        "memory_poll_interval_s": _measurement_interval_seconds(
            inference_config.get("memory_poll_interval_ms"),
            default_ms=1.0,
        ),
    }


def runtime_measurement_enabled(inference_config: dict[str, Any]) -> bool:
    """Return whether steady-state latency or memory measurement is enabled."""
    measurement_cfg = _runtime_measurement_config(inference_config)
    return (
        int(measurement_cfg["latency_measurement_runs"]) > 0
        or int(measurement_cfg["memory_measurement_runs"]) > 0
    )


def runtime_measurement_summary(inference_config: dict[str, Any]) -> dict[str, int | float]:
    """Return the resolved runtime measurement configuration for logging/debugging."""
    return _runtime_measurement_config(inference_config)


def _synchronise_measurement_device(device: torch.device) -> None:
    """Synchronise the active accelerator before/after timing-sensitive sections."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _summarise_samples(samples: list[float]) -> tuple[float, float, float]:
    """
    Return median, mean, and population standard deviation for scalar samples.

    Parameters
    ----------
    samples : list[float]
        A list of scalar samples to summarise. If the list is empty, all returned statistics will
        be zero.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the median, mean, and population standard deviation of the input samples.
    """
    if not samples:
        return 0.0, 0.0, 0.0
    median = float(statistics.median(samples))
    mean = float(statistics.fmean(samples))
    std = float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0
    return median, mean, std


def _measure_latency_forward_ms(
    forward_fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup_runs: int,
    measurement_runs: int,
) -> tuple[float, float, float]:
    """
    Measure steady-state wall-clock latency for a forward function.

    Parameters
    ----------
    forward_fn : Callable[[], Any]
        A callable that executes the forward pass to be measured. It should not include any
        timing-sensitive setup or teardown code, as those should be handled in the warmup and
        measurement loops.
    device : torch.device
        The torch device on which the forward function will be executed. This is used to ensure
        proper synchronisation for accurate timing.
    warmup_runs : int
        The number of warmup runs to execute before starting latency measurement. Warmup runs are
        used to mitigate the effects of just-in-time compilation, caching, and other one-time setup
        overheads that can skew latency measurements. A value of 0 means no warmup runs will be
        executed.
    measurement_runs : int
        The number of measurement runs to execute. A value of 0 means no measurement runs will be
        executed.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the median, mean, and population standard deviation of the measured
        latencies in milliseconds. If `measurement_runs` is 0, all returned values will be zero.
    """
    if measurement_runs <= 0:
        return 0.0, 0.0, 0.0

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _synchronise_measurement_device(device)
            forward_fn()
            _synchronise_measurement_device(device)

        samples_ms: list[float] = []
        for _ in range(measurement_runs):
            _synchronise_measurement_device(device)
            start = time.perf_counter()
            forward_fn()
            _synchronise_measurement_device(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarise_samples(samples_ms)


def _measure_cpu_peak_delta_mb(
    forward_fn: Callable[[], Any],
    *,
    warmup_runs: int,
    measurement_runs: int,
    poll_interval_s: float,
) -> tuple[float, float, float, str]:
    """
    Measure CPU peak memory delta using RSS polling when possible.

    This function attempts to measure the peak memory usage of the `forward_fn` by polling the
    Resident Set Size (RSS) of the current process at regular intervals while the `forward_fn`
    is executing. If the `psutil` library is not available, it falls back to using `tracemalloc` to
    measure the peak memory allocated by Python objects during the execution of `forward_fn`.
    The RSS-based measurement captures the total memory usage of the process, including all Python
    and native allocations, while the `tracemalloc`-based measurement captures only the memory
    allocated by Python objects.

    Parameters
    ----------
    forward_fn
        A callable that executes the forward pass to be measured. It should not include any
        timing-sensitive setup or teardown code, as those should be handled in the warmup and
        measurement loops.
    warmup_runs : int
        The number of warmup runs to execute before starting memory measurement.
    measurement_runs : int
        The number of measurement runs to execute.
    poll_interval_s : float
        The interval at which to poll the memory usage, in seconds.

    Returns
    -------
    tuple[float, float, float, str]
        A tuple containing the median, mean, and population standard deviation of the measured
        peak memory deltas in megabytes, as well as a string indicating the measurement backend used
        ("psutil_rss" for RSS polling, "tracemalloc" for tracemalloc-based measurement, or
        "disabled" if no measurement was performed).
        If `measurement_runs` is 0, all returned values will be zero and the backend will be
        "disabled".
    """
    if measurement_runs <= 0:
        return 0.0, 0.0, 0.0, "disabled"

    with torch.inference_mode():
        for _ in range(warmup_runs):
            forward_fn()

        if psutil is not None:
            process = psutil.Process()
            samples_mb: list[float] = []
            for _ in range(measurement_runs):
                baseline_rss = process.memory_info().rss
                peak_rss = baseline_rss
                stop_event = threading.Event()

                def _poll_peak_rss(stop_event: threading.Event = stop_event) -> None:
                    nonlocal peak_rss
                    while not stop_event.is_set():
                        rss = process.memory_info().rss
                        peak_rss = max(rss, peak_rss)
                        stop_event.wait(poll_interval_s)

                polling_thread = threading.Thread(target=_poll_peak_rss, daemon=True)
                polling_thread.start()
                try:
                    forward_fn()
                finally:
                    stop_event.set()
                    polling_thread.join()
                peak_rss = max(peak_rss, process.memory_info().rss)
                samples_mb.append(max(0.0, peak_rss - baseline_rss) / (1024.0 * 1024.0))

            median, mean, std = _summarise_samples(samples_mb)
            return median, mean, std, "psutil_rss"

        samples_mb = []
        for _ in range(measurement_runs):
            tracemalloc.start()
            try:
                forward_fn()
                _, peak = tracemalloc.get_traced_memory()
                samples_mb.append(peak / (1024.0 * 1024.0))
            finally:
                tracemalloc.stop()

    median, mean, std = _summarise_samples(samples_mb)
    return median, mean, std, "tracemalloc"


def _measure_peak_memory_forward_mb(
    forward_fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup_runs: int,
    measurement_runs: int,
    poll_interval_s: float,
) -> tuple[float, float, float, str]:
    """
    Measure steady-state peak memory delta for a forward function.

    Parameters
    ----------
    forward_fn : Callable[[], Any]
        A callable that executes the forward pass to be measured. It should not include any
        timing-sensitive setup or teardown code, as those should be handled in the warmup and
        measurement loops.
    device : torch.device
        The torch device on which the forward function will be executed.
    warmup_runs : int
        The number of warmup runs to execute before starting memory measurement.
    measurement_runs : int
        The number of measurement runs to execute.
    poll_interval_s : float
        The interval at which to poll the memory usage for CPU measurements, in seconds.
    """
    if measurement_runs <= 0:
        return 0.0, 0.0, 0.0, "disabled"

    if device.type == "cuda":
        with torch.inference_mode():
            for _ in range(warmup_runs):
                _synchronise_measurement_device(device)
                forward_fn()
                _synchronise_measurement_device(device)

            samples_mb: list[float] = []
            for _ in range(measurement_runs):
                _synchronise_measurement_device(device)
                baseline_allocated = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                forward_fn()
                _synchronise_measurement_device(device)
                peak_allocated = torch.cuda.max_memory_allocated(device)
                samples_mb.append(max(0.0, peak_allocated - baseline_allocated) / (1024.0 * 1024.0))

        median, mean, std = _summarise_samples(samples_mb)
        return median, mean, std, "cuda_peak_allocated"

    return _measure_cpu_peak_delta_mb(
        forward_fn,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        poll_interval_s=poll_interval_s,
    )


def _component_forward_functions(
    model: torch.nn.Module,
    *,
    model_name: str,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[Callable[[], Any], Callable[[], Any] | None, Callable[[], Any] | None]:
    """
    Build callables for full, upsampler-only, and LAM-only steady-state measurement.

    This function constructs separate callables for measuring the latency and memory of the full
    model forward pass, the upsampler component (if present), and the LAM component (if present).

    Parameters
    ----------
    model : torch.nn.Module
        The model to be measured, which may contain separate upsampler and LAM components.
    model_name : str
        The name of the model, which may be used to apply special handling for known architectures.
    input_tensor : torch.Tensor
        A representative input tensor to be used for the forward passes during measurement.
    device : torch.device
        The torch device on which the forward functions will be executed.

    Returns
    -------
    tuple[Callable[[], Any], Callable[[], Any] | None, Callable[[], Any] | None]
        A tuple containing three callables:
        - The first callable executes the full model forward pass.
        - The second callable executes only the upsampler component, or `None` if no separate
        upsampler is present.
        - The third callable executes only the LAM component, or `None` if no separate LAM is
        present.
    """
    measurement_input = input_tensor.contiguous()

    def _full_forward() -> Any:
        return model(measurement_input, collect_metrics=False)

    if model_name == "LAM":
        return _full_forward, None, _full_forward

    if not hasattr(model, "lam"):
        return _full_forward, None, None

    if model_name == "UpLAM":

        def _upsampler_forward() -> Any:
            x_rel, x_imag = model._prepare_cdbpn_input(measurement_input)  # type: ignore[operator]
            return model.cdbpn(x_rel, x_imag, collect_metrics=False).to(dtype=model.lam.D.dtype)  # type: ignore[operator, union-attr]

    elif hasattr(model, "upsampler"):

        def _upsampler_forward() -> Any:
            return model.upsampler(measurement_input, collect_metrics=False).to(  # type: ignore[operator]
                dtype=model.lam.D.dtype  # type: ignore[union-attr]
            )

    else:
        return _full_forward, None, None

    with torch.inference_mode():
        lam_input = _upsampler_forward().contiguous()
        _synchronise_measurement_device(device)

    def _lam_forward() -> Any:
        return model.lam(lam_input, collect_metrics=False)  # type: ignore[operator]

    return _full_forward, _upsampler_forward, _lam_forward


def apply_steady_state_runtime_metrics(  # noqa: C901, PLR0912, PLR0913, PLR0915
    metrics: dict[str, Any],
    *,
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    device: torch.device,
    inference_config: dict[str, Any],
) -> None:
    """
    Override single-pass timing and memory fields with steady-state measurements.

    Parameters
    ----------
    metrics : dict[str, Any]
        The metrics dictionary to update with steady-state runtime measurements.
    model : torch.nn.Module
        The model to be measured, which may contain separate upsampler and LAM components.
    model_name : str
        The name of the model, which may be used to apply special handling for known architectures.
    input_tensor : torch.Tensor
        A representative input tensor to be used for the forward passes during measurement.
    device : torch.device
        The torch device on which the forward functions will be executed.
    inference_config : dict[str, Any]
        The inference configuration dictionary, which may contain optional keys for runtime
        measurement settings. Supported keys include:
        - "latency_warmup_runs": Number of warmup runs before latency measurement (default: 0).
        - "latency_measurement_runs": Number of runs to measure latency (default: 0).
        - "memory_warmup_runs": Number of warmup runs before memory measurement (default: 0).
        - "memory_measurement_runs": Number of runs to measure memory (default: 0).
        - "memory_poll_interval_ms": Polling interval in milliseconds for memory measurement
        (default: 1.0 ms).
    """
    measurement_cfg = _runtime_measurement_config(inference_config)
    latency_runs = int(measurement_cfg["latency_measurement_runs"])
    memory_runs = int(measurement_cfg["memory_measurement_runs"])

    if latency_runs <= 0 and memory_runs <= 0:
        return

    num_frames = int(metrics.get("num_frames", 0) or input_tensor.shape[0] or 0)
    full_forward, upsampler_forward, lam_forward = _component_forward_functions(
        model,
        model_name=model_name,
        input_tensor=input_tensor,
        device=device,
    )

    metrics["runtime_measurement_method"] = "steady_state_repeated_forward"
    metrics["latency_warmup_runs"] = int(measurement_cfg["latency_warmup_runs"])
    metrics["latency_measurement_runs"] = latency_runs
    metrics["memory_warmup_runs"] = int(measurement_cfg["memory_warmup_runs"])
    metrics["memory_measurement_runs"] = memory_runs
    metrics["memory_poll_interval_ms"] = float(measurement_cfg["memory_poll_interval_s"]) * 1000.0

    if latency_runs > 0:
        total_time_ms, total_time_mean_ms, total_time_std_ms = _measure_latency_forward_ms(
            full_forward,
            device=device,
            warmup_runs=int(measurement_cfg["latency_warmup_runs"]),
            measurement_runs=latency_runs,
        )
        metrics["total_time_ms"] = total_time_ms
        metrics["total_time_mean_ms"] = total_time_mean_ms
        metrics["total_time_std_ms"] = total_time_std_ms
        if num_frames > 0:
            metrics["latency_per_frame_ms"] = total_time_ms / float(num_frames)

        if upsampler_forward is None:
            metrics["upsampler_time_ms"] = 0.0
            metrics["upsampler_time_mean_ms"] = 0.0
            metrics["upsampler_time_std_ms"] = 0.0
        else:
            (
                upsampler_time_ms,
                upsampler_time_mean_ms,
                upsampler_time_std_ms,
            ) = _measure_latency_forward_ms(
                upsampler_forward,
                device=device,
                warmup_runs=int(measurement_cfg["latency_warmup_runs"]),
                measurement_runs=latency_runs,
            )
            metrics["upsampler_time_ms"] = upsampler_time_ms
            metrics["upsampler_time_mean_ms"] = upsampler_time_mean_ms
            metrics["upsampler_time_std_ms"] = upsampler_time_std_ms

        if lam_forward is None:
            metrics["lam_total_time_ms"] = 0.0
            metrics["lam_total_time_mean_ms"] = 0.0
            metrics["lam_total_time_std_ms"] = 0.0
        elif lam_forward is full_forward:
            metrics["lam_total_time_ms"] = total_time_ms
            metrics["lam_total_time_mean_ms"] = total_time_mean_ms
            metrics["lam_total_time_std_ms"] = total_time_std_ms
        else:
            (
                lam_total_time_ms,
                lam_total_time_mean_ms,
                lam_total_time_std_ms,
            ) = _measure_latency_forward_ms(
                lam_forward,
                device=device,
                warmup_runs=int(measurement_cfg["latency_warmup_runs"]),
                measurement_runs=latency_runs,
            )
            metrics["lam_total_time_ms"] = lam_total_time_ms
            metrics["lam_total_time_mean_ms"] = lam_total_time_mean_ms
            metrics["lam_total_time_std_ms"] = lam_total_time_std_ms

        # When both upsampler and LAM are measured independently, derive the total
        # from the component times so that total >= standalone-LAM is guaranteed.
        # The end-to-end _full_forward measurement is data-dependent (eigh converges
        # faster on low-rank upsampled CSMs), making it unsuitable for fair comparison.
        if (
            upsampler_forward is not None
            and lam_forward is not None
            and lam_forward is not full_forward
        ):
            total_time_ms = metrics["upsampler_time_ms"] + metrics["lam_total_time_ms"]
            metrics["total_time_ms"] = total_time_ms
            metrics["total_time_mean_ms"] = (
                metrics["upsampler_time_mean_ms"] + metrics["lam_total_time_mean_ms"]
            )
            if num_frames > 0:
                metrics["latency_per_frame_ms"] = total_time_ms / float(num_frames)

    if memory_runs <= 0:
        return

    total_memory_mb, total_memory_mean_mb, total_memory_std_mb, memory_backend = (
        _measure_peak_memory_forward_mb(
            full_forward,
            device=device,
            warmup_runs=int(measurement_cfg["memory_warmup_runs"]),
            measurement_runs=memory_runs,
            poll_interval_s=float(measurement_cfg["memory_poll_interval_s"]),
        )
    )
    metrics["memory_measurement_backend"] = memory_backend
    metrics["total_memory_mb"] = total_memory_mb
    metrics["total_memory_mean_mb"] = total_memory_mean_mb
    metrics["total_memory_std_mb"] = total_memory_std_mb

    if upsampler_forward is None:
        metrics["upsampler_memory_mb"] = 0.0
        metrics["upsampler_memory_mean_mb"] = 0.0
        metrics["upsampler_memory_std_mb"] = 0.0
    else:
        (
            upsampler_memory_mb,
            upsampler_memory_mean_mb,
            upsampler_memory_std_mb,
            _,
        ) = _measure_peak_memory_forward_mb(
            upsampler_forward,
            device=device,
            warmup_runs=int(measurement_cfg["memory_warmup_runs"]),
            measurement_runs=memory_runs,
            poll_interval_s=float(measurement_cfg["memory_poll_interval_s"]),
        )
        metrics["upsampler_memory_mb"] = upsampler_memory_mb
        metrics["upsampler_memory_mean_mb"] = upsampler_memory_mean_mb
        metrics["upsampler_memory_std_mb"] = upsampler_memory_std_mb

    if lam_forward is None:
        metrics["lam_memory_mb"] = 0.0
        metrics["lam_memory_mean_mb"] = 0.0
        metrics["lam_memory_std_mb"] = 0.0
    elif lam_forward is full_forward:
        metrics["lam_memory_mb"] = total_memory_mb
        metrics["lam_memory_mean_mb"] = total_memory_mean_mb
        metrics["lam_memory_std_mb"] = total_memory_std_mb
    else:
        lam_memory_mb, lam_memory_mean_mb, lam_memory_std_mb, _ = _measure_peak_memory_forward_mb(
            lam_forward,
            device=device,
            warmup_runs=int(measurement_cfg["memory_warmup_runs"]),
            measurement_runs=memory_runs,
            poll_interval_s=float(measurement_cfg["memory_poll_interval_s"]),
        )
        metrics["lam_memory_mb"] = lam_memory_mb
        metrics["lam_memory_mean_mb"] = lam_memory_mean_mb
        metrics["lam_memory_std_mb"] = lam_memory_std_mb
