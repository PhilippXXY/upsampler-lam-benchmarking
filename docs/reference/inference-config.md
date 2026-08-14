# Inference Config

Full key reference for `config/inference_config.yaml`.
For measurement definitions, see [Scientific Benchmarking](../workflows/scientific-benchmarking.md).

## Top-Level Structure

| Section | Purpose |
| --- | --- |
| `inference` | Runtime, model, evaluation, output, and logging settings |
| `dataset` | Dataset root paths |

## `inference`

### Model selection

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `model_variant` | string | `bicubiclam_dist` | Exact retained variant used for one local inference run, for example `lam`, `uplam_dist`, `bicubiclam_e2e_upfroz`, or `ainnlam_e2e_auxen` |
| `benchmark_targets` | list[string] | `["lam", "bicubiclam", "uplam", "srcnnlam", "imdnlam", "safmnlam", "ganlam", "ainnlam"]` | Mixed selector list used by `src/evaluate_inference.py` when `--targets` is omitted. Entries may be family selectors such as `bicubiclam` or exact variant IDs such as `ainnlam_e2e_auxen` |

### Dataset and channel selection

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `data_set` | string | `locata` | Supported values: `locata`, `starss23`, `stairs26` |
| `locata_low_channel_indices` | list[int] | `[5, 9, 21, 25]` | Zero-based Eigenmike indices used for the 4-channel LOCATA path. AINNLAM also uses them to select the low-resolution microphone geometry at inference time |
| `variable_input_channel_counts` | list[int] | `[4, 8, 16, 24, 32]` | Trained microphone counts accepted by variable-SRCNN checkpoints |
| `locata_tasks` | list[string] or omitted | `["task1", "task2", "task3", "task4"]` | Optional LOCATA task subset; when omitted, task1-task4 are used |

### Audio and visibility frontend

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `sampling_rate` | int | `24000` | Target sample rate used after LOCATA resampling and in CSM generation |
| `frame_width_ms` | int | `100` | Output frame width in milliseconds |
| `max_audio_length_sec` | int or float | `0` | `0` means full-length audio |
| `visibility_diagonal_loading` | float | `1.0e-10` | Added to the covariance matrix for numerical stability |
| `visibility_eigenvalue_floor` | float | `1.0e-12` | Floors tiny normalised eigenvalues during visibility processing |

### File selection and loader settings

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `batch_size` | int | `1` | The current inference loop effectively expects `1` |
| `num_workers` | int | `4` | DataLoader worker count |
| `max_files` | int | `0` | `0` means all files |
| `file_selection_mode` | string | `stratified` | Supported values: `sorted`, `random`, `stratified` |
| `file_selection_seed` | int | `0` | Seed used for `random` or `stratified` file selection |
| `selected_files` | list[string] | `[]` | Explicit file ID list. Overrides the sampling mode |

### Metrics and SELD evaluation

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `collect_metrics` | bool | `true` | Enables runtime and evaluation metrics collection |
| `latency_warmup_runs` | int | `0` | Optional steady-state latency warm-up passes per file. `0` keeps the inline model timing only |
| `latency_measurement_runs` | int | `0` | Optional repeated full forward timings per file. When greater than `0`, `src/infer.py` overrides latency fields with the median steady-state result |
| `memory_warmup_runs` | int | `0` | Optional warm-up passes before peak memory measurement |
| `memory_measurement_runs` | int | `0` | Optional repeated peak-memory measurements per file. When greater than `0`, `src/infer.py` overrides memory fields with steady-state measurements |
| `memory_poll_interval_ms` | int or float | `1.0` | CPU/MPS memory polling interval used by the steady-state peak sampler |
| `normalised_memory_enabled` | bool | `true` | Enables the separate normalised runtime-only memory benchmark pass in `src/evaluate_inference.py` |
| `normalised_memory_duration_sec` | int or float | `10.0` | Canonical duration used by the normalised runtime-only memory benchmark |
| `normalised_memory_pad_short_files` | bool | `true` | Zero-pads files shorter than the normalised duration during the runtime-only benchmark pass |
| `benchmark_runtime_only` | bool | `false` | Internal switch used by the benchmark runner. When `true`, `src/infer.py` skips prediction CSV writing and SELD evaluation, but still writes per-file runtime metrics |
| `class_agnostic_evaluation` | bool | `true` | If `false`, class-aware evaluation uses `num_classes` |
| `num_classes` | int | `3` | Used only when class-aware evaluation is enabled |
| `doa_threshold_deg` | float | `10.0` | Angular threshold for a correct detection |
| `seld_average` | string | `macro` | Supported values: `macro`, `micro` |

### Intensity clustering

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `n_max` | int | `50` | Number of top pixels passed into the K=1 fallback k-means call |
| `max_sources` | int | `3` | Upper bound on detected sources per frame and on the number of candidates extracted by the sequence clusterer per frame |
| `intensity_threshold` | float | `0.0` | Relative intensity threshold used by the legacy k-means path; `0` disables it |
| `adaptive_k` | bool | `true` | When `true`, routes through the sequence-level clusterer described below. When `false`, runs the legacy fixed-K k-means with `max_sources=1` |
| `peak_ratio_threshold` | float | `0.3` | Ratio threshold used by the adaptive K estimator in the legacy path, and passed as `candidate_peak_ratio` in the K=1 fallback call |

### Sequence clustering

These parameters are used only when `adaptive_k: true`.  They control the two-stage pipeline with per-frame candidate extraction followed by temporal track formation, with a K=1 k-means fallback for uncovered frames.

See [DoA Extraction Pipeline](../workflows/benchmarking/doa-extraction.md) for a full description of the algorithm and the empirical justification for these defaults.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `merge_radius_deg` | float | `25.0` | Angular radius of the suppression cone around each accepted peak. Pixels inside this cone are grouped into the candidate's support region and masked before the next search |
| `candidate_peak_ratio` | float | `0.3` | A candidate is rejected if its peak pixel is below `candidate_peak_ratio × frame_max`. Controls how far down the intensity distribution secondary sources are sought |
| `candidate_mass_ratio` | float | `0.05` | A candidate is rejected if its support region contains less than this fraction of total frame energy. On the 484-pixel Fibonacci sphere a 25° cone covers approximately 23 pixels (~4.8% of uniform energy), so `0.05` is just above the isotropic floor and admits genuine sources while discarding stray noise peaks |
| `track_max_jump_deg` | float | `20.0` | Maximum great-circle displacement (degrees) allowed when linking a candidate to an existing track. Matches the DCASE evaluation angular threshold, which means only physically plausible source motion is linked |
| `track_max_gap` | int | `3` | Number of consecutive frames without a matching candidate that a track can survive before it is finalised. Three frames corresponds to 300 ms at the default 100 ms frame width, sufficient to bridge typical low-energy transitions |
| `track_min_frames` | int | `3` | Minimum number of active frames a track must contain to be retained. Rejects spurious single-frame candidates from noise. On sequences shorter than this value the effective minimum is clamped to `max(1, n_frames)` |
| `track_min_active_ratio` | float | `0.05` | Minimum fraction of total sequence frames a track must cover. Suppresses ghost tracks that appear only during transient noise bursts on long recordings |
| `track_min_peak_rel` | float | `0.5` | Minimum mean `peak_rel` (peak intensity relative to frame max) across all active frames in the track. Discards tracks formed from consistently weak peaks that are unlikely to correspond to real sources |
| `band_peak_ratio` | float | `0.2` | A frequency band is counted as *supporting* a candidate if the maximum intensity inside its support region reaches at least `band_peak_ratio × band_max`. Used to compute the `band_support` quality metric but not currently used in track survival filtering |
| `activity_threshold` | float | `0.01` | Frames whose maximum intensity is below `activity_threshold × file_max` are treated as silent and produce no candidates. Prevents track formation during noise-only tails |

### Outputs

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `output_path` | string | `output` | Root directory under which timestamped run folders are created |

### Logging

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `logging.level` | string | `DEBUG` | Root logger level |
| `logging.format` | string | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Shared formatter string |
| `logging.handlers[].type` | string | `file`, `console` | Supported values: `file`, `console` |
| `logging.handlers[].filename` | string | `logs/inference_{timestamp}.log` | File handler only. `{timestamp}` is substituted at runtime |
| `logging.handlers[].level` | string | `DEBUG` or `INFO` | Per-handler level |
| `logging.handlers[].mode` | string | `w` | File handler only |

## `dataset`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `data_audio_path` | string | `data/locata/eval` | Audio root for the selected dataset |
| `data_ground_truth_path` | string | `data/starss23/metadata_dev/dev-test-sony` | Used by STARSS23 evaluation only |

## Behaviour Notes

| Topic | Detail |
| --- | --- |
| LOCATA channel selection | `LAM` keeps all 32 channels. The wrapper models use the configured 4-channel subset |
| Retained variant resolution | `model_variant` is resolved inside `src/infer.py` into the existing runtime loader fields `model_name`, `model_checkpoint`, and optional `lam_checkpoint` |
| `*_dist` meaning | Distinct variants such as `bicubiclam_dist` are logical retained runtime names. They usually resolve to separate retained upsampler and LAM checkpoints rather than to one combined `.pth` file |
| Steady-state runtime metrics | When `latency_measurement_runs` or `memory_measurement_runs` is greater than `0`, `src/infer.py` performs warm-up and repeated forward passes and writes median latency and peak-memory fields, plus mean/std companion fields, into `metrics_*.json` |
| Runtime-only benchmark mode | When `benchmark_runtime_only` is `true`, `src/infer.py` still preprocesses audio, builds visibility matrices, runs inference, and writes per-file metrics JSON, but it does not write prediction CSVs or run SELD evaluation |
| Normalised memory benchmark | `src/evaluate_inference.py` can run a second runtime-only pass that forces every selected file to `normalised_memory_duration_sec` before the shared frontend and model forward path |
| Total vs component memory | `total_memory_mb` reflects an end-to-end forward peak. `lam_memory_mb` and `upsampler_memory_mb` are measured on isolated component passes and are therefore informative but not additive |
| Family selectors in benchmarking | `benchmark_targets` and `--targets` accept both family selectors and exact variant IDs. Family selectors expand in the fixed order `dist`, `e2e_auxdis`, `e2e_upfroz` where that family supports them, and skip variants whose checkpoints are not present locally. |
