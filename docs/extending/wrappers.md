# Custom Wrappers

Wrappers bridge repository-native upsamplers and the retained LAM implementation.

## Wrapper Responsibilities

A wrapper must do more than a standalone upsampler:

- expose a forward path compatible with inference
- expose `forward_components(...)` so the end-to-end trainer can access intermediate tensors
- separate the upsampler branch from the LAM branch clearly enough for checkpoint initialisation

## Minimum Contract

| Method or attribute | Why it matters |
| --- | --- |
| `upsampler` | Lets the trainer and checkpoint initialiser access the front-end module |
| `lam` | Lets the trainer and checkpoint initialiser access the back-end LAM module |
| `forward_components(...)` | Returns intermediate tensors needed by the joint loss |

`forward_components(...)` is expected to return:

1. the raw upsampler prediction
2. the final wrapper output used by the LAM side
3. the latent tensor used by the original LAM loss

## Registration Steps

1. Implement the wrapper in `src/lam_min/model/`.
2. Add a branch in `build_end_to_end_model(...)`.
3. Add the correct warm-start logic in `initialise_model(...)`.
4. Add inference support in `src/infer.py` if needed.
5. Add retained family and variant entries in `src/utils/model_variants.py` if it should be part of the comparison workflow.

## API Reference

### End-to-end wrapper factory

::: training.end_to_end.build_end_to_end_model

### Wrapper initialisation

::: training.end_to_end.initialise_model

### LAM-loss construction

::: training.end_to_end.build_lam_loss
