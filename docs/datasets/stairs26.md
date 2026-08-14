# STAIRS26

STAIRS26 provides 32-channel Eigenmike development recordings for acoustic-image localisation. Its 4-channel evaluation recordings are also accepted by the loader.

## Inference setup

Extract `32ch_audio_dev.zip`, then point inference at either the archive root or a directory beneath it:

```yaml
inference:
  data_set: "stairs26"

dataset:
  data_audio_path: "data/stairs26/32ch_audio_dev"
```

The loader searches recursively, so `data_audio_path` can point to the archive root or directly to an audio directory. It returns time-major audio, sample rate, file ID, and source path.

For variable SRCNN–LAM, provide canonical zero-based Eigenmike channels on the command line:

```bash
uv run python src/infer.py \
  --config config/inference_config.yaml \
  --device cuda \
  --input-channel-indices 5 9 21 25
```

With 32-channel development audio, inference retains the full CSM as the reconstruction reference. Runtime and CSM metrics remain available. SELD scoring is skipped because STAIRS26 uses acoustic-image JSON labels rather than the DCASE event CSV format consumed by the current evaluator.

The official data is available from the [STAIRS26 Zenodo record](https://zenodo.org/records/18171005), with task details on the [DCASE 2026 Task 3 page](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes).
