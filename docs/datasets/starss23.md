# STARSS23

4-channel inference dataset for wrapper models and SELD-style evaluation. See [R4](../references.md#r4).

## Role in This Repository

| Use case | Supported |
| --- | --- |
| `LAM` baseline inference | No |
| Wrapper-model inference | Yes |
| Standalone training | No |
| End-to-end training | No |

## Loader Classes

| Class | Purpose |
| --- | --- |
| `data.starss_loader.StarssAudioDataset` | Full inference loader returning audio, sample rate, file ID, and optional DCASE ground truth |
| `data.starss_loader.StarssGroundTruthLoader` | DCASE CSV parser and frame-to-time conversion |

## Expected Layout

The config points to separate audio and ground-truth roots:

```yaml
dataset:
  data_audio_path: "data/starss23/mic_dev/dev-test-sony"
  data_ground_truth_path: "data/starss23/metadata_dev/dev-test-sony"
```

The audio loader expects a directory containing `*.wav` files. The ground-truth loader expects matching CSV files keyed by the WAV stem.

## Sample Contract

`StarssAudioDataset.__getitem__()` returns:

| Key | Type | Meaning |
| --- | --- | --- |
| `audio` | `torch.Tensor` | Time-major audio with shape `(T, 4)` |
| `sample_rate` | `int` | Audio sample rate |
| `file_id` | `str` | WAV stem |
| `audio_path` | `str` | Source WAV path |
| `ground_truth` | `DoaEvent`, optional | Included only when `load_ground_truth=True` |

## Loader Behaviour

| Step | What happens |
| --- | --- |
| Scan | Collects all `*.wav` files under `audio_path` |
| Validate | Requires exactly 4 channels per file |
| Load ground truth | Optional DCASE-format CSV parsing keyed by `file_id` |
| Frame alignment | `frame_width_ms` is used to convert frame numbers into seconds |

## Practical Notes

- `dataset.data_ground_truth_path` is mandatory for evaluation runs that compute SELD metrics.
- Channel order is already fixed in the STARSS23 subset used here so there is no extra channel-selection step applied.
