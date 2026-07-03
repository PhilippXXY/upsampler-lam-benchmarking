# Correlation Matrix Distance (CMD) Methodology

The Correlation Matrix Distance (CMD) quantifies how similar two Cross-Spectral Matrices (CSMs) are.
It was originally introduced for MIMO channel evaluation [R18](../../references.md#r18) and is used here to measure how faithfully each pipeline stage preserves the spatial information encoded in the full-resolution CSM.

## Definition

For two Hermitian matrices $\mathbf{A}, \mathbf{B} \in \mathbb{C}^{C \times C}$ with non-zero Frobenius norms, CMD is defined as

$$
\text{CMD}(\mathbf{A}, \mathbf{B})
  = 1 - \frac{\operatorname{tr}(\mathbf{A}\mathbf{B})}
             {\|\mathbf{A}\|_F \, \|\mathbf{B}\|_F}
  \in [0, 1].
$$

A value of $0$ indicates identical matrices up to a positive scalar; a value of $1$ indicates orthogonal matrices (no shared spatial structure).
Pairs where either matrix is non-finite, or where the denominator is zero, are set to `nan` and excluded from aggregation.

## Hermitian PSD Pre-Processing

Raw upsampler outputs may not be exactly Hermitian or positive semi-definite (PSD).
Before CMD is computed for any upsampler-stage CSM, the matrix is projected onto the Hermitian PSD cone.

The residuals from this projection are logged as diagnostic metrics
(`upsampler_hermitian_residual`, `upsampler_psd_projection_residual`) to flag degenerate
upsampler outputs.
Reference and LAM-stage CSMs are used as-is; only the Hermitian symmetrisation step inside
`_compute_cmd_tensor` is applied to all inputs uniformly.

## Pipeline Comparisons

Three CMD comparisons are evaluated for each recording.

| Key | Pair | Interpretation |
| --- | ---- | -------------- |
| `cmd_reference_to_upsampler` | Reference → Upsampler output | Fidelity of spatial upsampling |
| `cmd_upsampler_to_lam` | Upsampler output → LAM final | Information transformation inside LAM |
| `cmd_reference_to_lam` | Reference → LAM final | End-to-end spatial fidelity |

For models with intermediate LAM denoising stages, per-stage variants
`cmd_reference_to_lam_denoise{1–4}` are also recorded.

## Aggregation

CMD is computed frame-by-frame and band-by-band, producing a tensor of shape
`(frames, bands)` per comparison per recording.
Two aggregation levels are reported.

| Level | How | Stored as |
| ----- | --- | --------- |
| Per-frame | Median across the 9 frequency bands (finite values only) | `cmd_*_per_frame` list in the metrics JSON |
| Global | Median of all finite per-frame values across the entire recording | `cmd_*_median` scalar |

The **global median** is the primary reported figure.

## Implementation

| Symbol | Code location |
| ------ | ------------- |
| Core CMD formula | `compute_correlation_matrix_distance` in `src/utils/cmd_metrics.py` |
| Frame-band batched variant | `compute_cmd_per_frame_per_band` in `src/utils/cmd_metrics.py` |
| Hermitian PSD projection | `project_to_hermitian_psd` in `src/utils/cmd_metrics.py` |
| Per-frame median reduction | `reduce_cmd_per_frame` in `src/utils/cmd_metrics.py` |
| Pipeline integration | `add_cmd_metrics` called in `src/infer.py` |
