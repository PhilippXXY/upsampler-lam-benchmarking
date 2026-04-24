# AudibleLight

Multichannel training dataset providing low/high-resolution CSM pairs. See [R5](../references.md#r5).

## Role in This Repository

| Use case | Supported |
| --- | --- |
| Inference | No |
| Standalone training | Yes |
| End-to-end training | Yes |

## Loader Class

The training loader is `data.audiblelight_loader.AudibleLightCSMPairDataset`.

## Expected Layout

The loader expects the dataset root to contain the structures referenced by:

- `em32_dev/metadata.csv` when available
- `em32_dev/dev-train/*.wav`
- `metadata_dev/dev-train/*.csv`

If `metadata.csv` is missing, the loader falls back to scanning the waveform directory.

## Sample Contract

`AudibleLightCSMPairDataset.__getitem__()` returns:

| Key | Type | Meaning |
| --- | --- | --- |
| `file_id` | `str` | File identifier |
| `scene_id` | `str` | Scene identifier extracted from the file name |
| `dataset` | `str` | Fixed to `audiblelight` |
| `audio_path` | `str` | Source WAV path |
| `ground_truth_path` | `str` or `None` | Optional DCASE ground-truth CSV |
| `has_dcase_gt` | `bool` | Whether a ground-truth CSV is present |
| `S_low` | `torch.Tensor` | Complex low-resolution CSM tensor |
| `S_high` | `torch.Tensor` | Complex high-resolution CSM tensor |

## Loader Behaviour

| Step | What happens |
| --- | --- |
| Discover files | Uses `metadata.csv` when present, otherwise scans the directory |
| Scene parsing | Extracts `scene_id` from `scene_##_mic##` style file IDs |
| Split | Performs a deterministic train/val/test split over scene IDs rather than file IDs |
| Channel selection | Uses `low_channel_indices` for `S_low` and all available channels for `S_high` |
| CSM generation | Calls `get_visibility_matrix(...)` for both low and high branches |
| Cache | `cache_csm=True` stores computed tensors in memory for the current process. `precomputed_csm_root` stores `.pt` tensors on disk and reuses them across runs |

## Why the Scene Split Matters

The split is scene-based to reduce leakage between train and validation data. Files from the same scene are kept together instead of being scattered across splits.

## Practical Notes

- The loader expects the configured `sampling_rate` and raises if the raw file rate differs.
- `max_files` is applied after the split has been computed.
- This dataset is the first-stage default for the retained standalone and end-to-end training runs.
- The on-disk cache path is namespaced by dataset root and CSM settings, so multiple training configs can share one cache root safely.
