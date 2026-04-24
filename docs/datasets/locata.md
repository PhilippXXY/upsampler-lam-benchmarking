# LOCATA

32-channel Eigenmike evaluation dataset for LAM and reduced-channel wrapper inference. See [R3](../references.md#r3).

## Role in This Repository

| Use case | Supported |
| --- | --- |
| `LAM` baseline inference | Yes |
| Wrapper-model inference | Yes, after selecting 4 channels |
| Standalone training | No |
| End-to-end training | No |

## Loader Classes

| Class | Purpose |
| --- | --- |
| `data.locata_loader.LocataAudioDataset` | Full inference loader returning audio, sample rate, file ID, and optional ground truth |
| `data.locata_loader.LocataGroundTruthLoader` | Frame-level DoA ground-truth conversion |

## Expected Layout

The loader scans task directories beneath `dataset.data_audio_path` and expects Eigenmike recordings under the LOCATA structure used by the downloaded archive.

The current config defaults to:

```yaml
dataset:
  data_audio_path: "data/locata/eval"
```

## Sample Contract

`LocataAudioDataset.__getitem__()` returns:

| Key | Type | Meaning |
| --- | --- | --- |
| `audio` | `torch.Tensor` | Time-major audio with shape `(T, 32)` |
| `sample_rate` | `int` | Original LOCATA sample rate |
| `file_id` | `str` | Recording identifier such as `task1_recording1` |
| `audio_path` | `str` | Source WAV path |
| `ground_truth` | `DoaEvent`, optional | Included only when `load_ground_truth=True` |

## Loader Behaviour

| Step | What happens |
| --- | --- |
| Scan | `_scan_eigenmike_recordings(...)` discovers Eigenmike files per task |
| Filter | `locata_tasks` can restrict the task subset |
| Load | Audio is read as `float32` with `always_2d=True` |
| Validate | The loader requires exactly 32 raw channels |
| Prepare for inference | `prepare_locata_audio_for_inference(...)` resamples, crops, and optionally selects the 4-channel subset |

## Channel Assumptions

| Model path | Channel behaviour |
| --- | --- |
| `LAM` | Uses the full 32-channel recording |
| Wrapper models | Use `locata_low_channel_indices`, which default to the zero-based equivalent of microphones 6, 10, 22, and 26 |

## Practical Notes

- LOCATA is the only inference dataset that exercises the full 32-channel LAM path.
- Resampling is handled in the repository rather than assumed from the raw files.
- Task filtering happens before file selection, so `max_files` and `selected_files` apply to the chosen task subset.
