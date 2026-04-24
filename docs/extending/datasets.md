# Custom Datasets

Decide whether the new dataset is for inference, training, or both — each uses a different sample contract.

## Pick the Right Contract

| Workflow | Required return shape |
| --- | --- |
| Inference | raw audio sample plus metadata |
| Training | `S_low` and `S_high` complex tensors plus metadata |

## Inference Dataset Contract

The current inference path expects a file-level item with:

| Key | Meaning |
| --- | --- |
| `audio` | `torch.Tensor` with time on axis 0 |
| `sample_rate` | integer sample rate |
| `file_id` | stable identifier used for outputs |
| `audio_path` | source path for logging |
| `ground_truth`, optional | evaluation target if the dataset supports it |

To add a new inference dataset:

1. Implement a loader under `src/data/`.
2. Return the same kind of file-level record as the existing audio loaders.
3. Add the dataset branch to `src/infer.py`.
4. Update `config/inference_config.yaml` and the docs.

## Training Dataset Contract

The training entrypoints expect each item to represent one file and to contain:

| Key | Meaning |
| --- | --- |
| `S_low` | low-resolution complex CSM tensor |
| `S_high` | high-resolution complex CSM tensor |
| metadata fields | file IDs, class labels, dataset tags, and optional ground-truth paths |

To add a new training dataset:

1. Implement a loader that returns `S_low` and `S_high`.
2. Decide whether resampling or channel filtering belongs inside the loader.
3. Register the loader in `build_dataset_list(...)`.
4. Add config keys under the relevant YAML file.

## API Reference

### STARSS23 inference loader

::: data.starss_loader.StarssAudioDataset
    options:
      members:
        - __init__
        - __getitem__

### LOCATA inference loader

::: data.locata_loader.LocataAudioDataset
    options:
      members:
        - __init__
        - __getitem__

### AudibleLight training loader

::: data.audiblelight_loader.AudibleLightCSMPairDataset
    options:
      members:
        - __init__
        - __getitem__

### EigenScape training loader

::: data.eigenscape_loader.EigenscapeCSMPairDataset
    options:
      members:
        - __init__
        - __getitem__
