# Custom Upsamplers

Standalone upsamplers plug into the repository through the `TrainableUpsampler` base class.

## What a New Upsampler Must Provide

| Requirement | Why it exists |
| --- | --- |
| `forward(...)` | Produces the reconstructed high-resolution CSM |
| `training_step(...)` | Owns optimisation logic for one frame chunk |
| `validation_step(...)` | Returns loss statistics without optimiser updates |
| consistent loss statistics | Lets the trainer log and checkpoint models without model-specific branches |

## Registration Steps

1. Implement the class under `src/upsampler/<family>/`.
2. Make it inherit from `TrainableUpsampler`.
3. Add a branch in `build_model(...)` in `src/train_upsamplers.py`.
4. Add any model-specific config keys to `config/train_upsamplers.yaml`.
5. If the model should support inference inside a wrapper, add the matching wrapper path separately.

## Practical Expectations

| Topic | Expectation |
| --- | --- |
| Input tensor | Complex tensor shaped `(batch, num_bands, in_channels, in_channels)` |
| Output tensor | Complex tensor shaped `(batch, num_bands, out_channels, out_channels)` |
| Metrics | If `collect_metrics=True`, return runtime metrics in the same shape used by the current models |
| Hermitian structure | The current models project their output back to a Hermitian matrix; a new upsampler should do the same unless there is a strong reason not to |

## API Reference

### Shared base class

::: upsampler.base.TrainableUpsampler

### Standalone model factory

::: train_upsamplers.build_model
