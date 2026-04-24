# EigenScape

Second training dataset, broadening scene coverage for fine-tuning. Supports strict channel filtering before CSM generation. See [R6](../references.md#r6).

## Role in This Repository

| Use case | Supported |
| --- | --- |
| Inference | No |
| Standalone training | Yes |
| End-to-end training | Yes |

## Loader Class

The training loader is `data.eigenscape_loader.EigenscapeCSMPairDataset`.

## Expected Layout

The loader recursively scans the configured root for WAV files. It infers the scene class from the file stem, for example `Beach-01-Raw.wav`.

## Sample Contract

`EigenscapeCSMPairDataset.__getitem__()` returns:

| Key | Type | Meaning |
| --- | --- | --- |
| `file_id` | `str` | WAV stem |
| `dataset` | `str` | Fixed to `eigenscape` |
| `class_name` | `str` | Parsed scene label such as `Beach` or `TrainStation` |
| `audio_path` | `str` | Source WAV path |
| `ground_truth_path` | `None` | No DCASE ground-truth path is used |
| `has_dcase_gt` | `bool` | Always `false` |
| `S_low` | `torch.Tensor` | Complex low-resolution CSM tensor |
| `S_high` | `torch.Tensor` | Complex high-resolution CSM tensor |

## Loader Behaviour

| Step | What happens |
| --- | --- |
| Scan | Recursively collects WAV files beneath `root_path` |
| Class parsing | Derives scene labels from the filename stem |
| Channel filtering | Applies `expected_channels`, `target_high_channels`, and `allow_channel_fallback` |
| Split | Performs class-wise deterministic splitting using either `split_ratio` or `split_counts` |
| Resample | Resamples to the configured training rate before CSM generation |
| Cache | `cache_csm=True` stores computed tensors in memory for the current process. `precomputed_csm_root` stores `.pt` tensors on disk and reuses them across runs |

## Channel Handling

| Setting | Behaviour |
| --- | --- |
| `expected_channels=32`, `allow_channel_fallback=false` | Strict 32-channel training only |
| `allow_channel_fallback=true` | Missing low channels are clamped and the high-resolution branch is padded or truncated to `target_high_channels` |

This has been implemented to allow training on EigenScape recordings that do not have the full 32-channel complement.
The fallback behaviour is not ideal but it is better than discarding otherwise usable data.
