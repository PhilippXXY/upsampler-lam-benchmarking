# End-to-End Training Config

Full key reference for `config/train_end_to_end.yaml`.

## Top-Level Structure

| Section | Purpose |
| --- | --- |
| `training` | Global trainer settings and stage schedule |
| `initialisation` | Warm-start and resume checkpoint settings |
| `loss` | Joint LAM and auxiliary loss settings |
| `model` | Wrapper selection and wrapper-side hyperparameters |
| `data` | Dataset and CSM construction settings |

## `training`

The keys below mirror the standalone trainer unless noted otherwise.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `seed` | int | `0` | Passed to the repository seed helper |
| `device` | string | `mps` | Requested device. LAM-based training falls back to CPU on Apple Silicon as `float64` and `complex128` are not supported |
| `epochs` | int | `4` | Global stage default |
| `learning_rate` | float | `1.0e-6` | Global stage default. This matches the original LAM paper's Adam learning rate |
| `weight_decay` | float | `1.0e-4` | Global stage default |
| `frame_batch_size` | int | `32` | Number of frames processed per optimiser step |
| `gradient_clip_norm` | float | `1.0` | `0.0` disables clipping |
| `num_workers` | int | `0` | DataLoader worker count |
| `max_train_files` | int | `0` | Global stage default (0 = no limit) |
| `max_val_files` | int | `0` | Global stage default (0 = no limit) |
| `log_every_files` | int | `10` | File-level logging cadence |
| `early_stopping_patience` | int | `3` | Global stage default |
| `early_stopping_min_delta` | float | `0.0` | Global stage default |
| `resume_from_checkpoint` | string | empty | Optional combined wrapper checkpoint to resume from. This is the config-level equivalent of `--resume-checkpoint` |
| `output_root` | string | `output/training_end_to_end` | Root directory for logs and metrics JSON |
| `checkpoint_dir` | string | `src/lam_min/checkpoints/e2e` | Output directory for combined checkpoints |
| `checkpoint_prefix` | string | empty | Optional manual override for combined checkpoint names. When empty, the trainer derives the canonical retained end-to-end name from `model.name`, `training.freeze_upsampler`, and `loss.aux_enabled`. `BicubicLAM` always maps to `bicubiclam_e2e_upfroz` because the bicubic upsampler is fixed |
| `freeze_upsampler` | bool | `false` | Freezes the whole upsampler branch, keeps it in eval mode, and optimises only LAM. This produces the retained `*_e2e_upfroz` naming and requires `loss.aux_enabled: false` |

### `training.stages[]`

The stage schema matches the standalone trainer:

| Key | Type | Meaning |
| --- | --- | --- |
| `name` | string | Stage label used in logs and checkpoint names |
| `enabled` | bool | If `false`, the stage is skipped |
| `epochs` | int | Epoch count for this stage |
| `learning_rate` | float | Stage-specific optimiser learning rate |
| `weight_decay` | float | Stage-specific weight decay |
| `max_train_files` | int | Optional cap on training files |
| `max_val_files` | int | Optional cap on validation files |
| `train_sampling` | string | `proportional` or `balanced` |
| `early_stopping_patience` | int | Early stopping patience |
| `early_stopping_min_delta` | float | Minimum validation improvement |
| `datasets.audiblelight` | bool | Enable or disable AudibleLight |
| `datasets.eigenscape` | bool | Enable or disable EigenScape |

## `initialisation`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `upsampler_checkpoint` | string | empty | Standalone checkpoint used to warm-start the wrapper upsampler branch. Leave this empty for `BicubicLAM`. It is required for the trainable wrappers when not resuming |
| `lam_checkpoint` | string | `src/lam_min/checkpoints/LAM.pth` | Standalone LAM checkpoint used when not resuming a combined wrapper |
| `resume_checkpoint` | string | empty | Fallback combined wrapper checkpoint. Used when neither `--resume-checkpoint` nor `training.resume_from_checkpoint` is set |
| `resume_strict` | bool | `true` | Strictness for `load_state_dict` when resuming a combined checkpoint |

## `loss`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `lam_method` | string | `original_msetv` | Currently the only supported LAM loss mode |
| `lam_tv_weight` | float | `1.0e-5` | Total-variation weight for the original-method LAM loss |
| `aux_enabled` | bool | `true` | Enables the extra auxiliary upsampler reconstruction loss for joint-training modes. `training.freeze_upsampler: true` requires this to be `false` |
| `aux_weight` | float | `0.25` | For joint AuxEn training, this is the configured fraction of the initial LAM contribution. The trainer derives the applied multiplier as `aux_weight * (initial_lam_total / initial_aux_raw)` from one random-init probe pass |
| `use_model_specific_aux_loss` | bool | `true` | Reuses the upsampler's own reconstruction helper where available |

### Loss Composition

The end-to-end trainer reports and early-stops on `loss_total`, which is calculated per frame chunk as:

```text
non-GAN: loss_total = loss_lam_total + loss_aux
GANLAM: loss_total = loss_lam_total + loss_g_total + loss_aux
loss_lam_total = loss_lam_reconstruction + loss_lam_tv
loss_g_total = loss_g_content + loss_g_adv
loss_g_content = content_weight * loss_g_content_raw
loss_g_adv = adversarial_weight * loss_g_adv_raw
effective_aux_weight = aux_weight * (initial_lam_total / initial_aux_raw)
loss_aux = effective_aux_weight * loss_aux_raw
e2e_upfroz: loss_total = loss_lam_total
```

- `loss_lam_reconstruction` is the complex MSE between the LAM output and the high-resolution target CSM.
- `loss_lam_tv` is the original-method LAM regularisation term scaled by `lam_tv_weight`.
- For `GANLAM`, `loss_g_content_raw` is the GAN generator reconstruction term on `S_pred`, and `loss_g_adv_raw` is the discriminator-based adversarial term on `S_pred`.
- `loss_aux_raw` is the extra auxiliary reconstruction loss on the wrapper upsampler output.
- For joint AuxEn runs, the trainer computes `initial_lam_total` and `initial_aux_raw` once from a temporary random-initialised wrapper on one real training chunk before the first optimiser step. That probe ignores `initialisation.*`. Only the real training model is warm-started or resumed.
- Explicit CSM comparisons: `loss_lam_reconstruction` compares `S_out` to `S_high`, and `loss_aux_raw` compares `S_pred` to `S_high`. `S_low` is used only as model input.
- When `use_model_specific_aux_loss` is enabled, the trainer reuses each upsampler's own reconstruction-loss helper. Otherwise it uses a generic complex MSE on the predicted and target CSM.
- `GANLAM` trains its discriminator only while the upsampler remains trainable. In `*_e2e_upfroz`, the GAN wrapper uses the same LAM-only optimisation path as the non-GAN wrappers.
- Epoch summaries, per-file `avg loss`, checkpoint selection, and early stopping all use the average `loss_total` over processed chunks.
- The saved per-epoch stats expose `loss_aux_weight_config`, `loss_aux_baseline_ratio`, and `loss_aux_weight`, where `loss_aux_weight` is the actual applied multiplier.

## `model`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | `BicubicLAM` | Supported values: `UpLAM`, `BicubicLAM`, `SRCNNLAM`, `IMDNLAM`, `SAFMNLAM`, `GANLAM`, `AINNLAM` |
| `in_channels` | int | `4` | Low-resolution microphone count |
| `out_channels` | int | `32` | High-resolution microphone count |
| `feature_channels` | int | `64` | Used by SRCNN, IMDN, and SAFMN. GAN uses `128` in the inference path |
| `mapping_channels` | int | `32` | Used by SRCNN and IMDN |
| `n_blocks` | int | `8` | Used by SAFMN and as a residual-block default for GAN configs |
| `n_residual_blocks` | int | `8` | Used by GAN wrappers |
| `ffn_scale` | float | `2.0` | Used by SAFMN |
| `n_levels` | int | `4` | Used by SAFMN |
| `hidden_channels` | int | `64` | Used by AINN |
| `latent_channels` | int | `64` | Used by AINN wrappers |
| `loss_name` | string | `mse` | Used by auxiliary loss paths where relevant |
| `fft_loss_weight` | float | `0.1` | Used by SAFMN-style auxiliary loss paths |
| `pde_loss_weight` | float | `0.5` | Used by AINN |
| `pde_freq_min_hz` | float | `100.0` | Used by AINN |
| `pde_freq_max_hz` | float | `4000.0` | Used by AINN |
| `sound_speed` | float | `343.0` | Used by AINN |
| `adversarial_weight` | float | `0.01` | Used by GAN |
| `content_weight` | float | `0.01` | Used by GAN |
| `critic_iters` | int | `4` | Used by GAN. Controls how many discriminator updates happen per generator update in end-to-end mode |
| `discriminator_lr_scale` | float | `5.0` | Used by GAN. Scales the discriminator learning rate relative to the stage learning rate |
| `beta1_adam` | float | `0.9` | Used by GAN |
| `beta2_adam` | float | `0.999` | Used by GAN |

## `data`

### Shared CSM settings

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `low_channel_indices` | list[int] | `[5, 9, 21, 25]` | Zero-based Eigenmike channels used for the low-resolution branch. AINNLAM also passes them into the AINN upsampler geometry |
| `sampling_rate` | int | `24000` | Target sample rate for CSM generation |
| `nbands` | int | `9` | Number of frequency bands passed into `get_visibility_matrix` |

### `data.audiblelight`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Global dataset enable |
| `root_path` | string | `data/AudibleLight_Eigenmike32-5_DCASE-STARSS23_Dataset` | Dataset root |
| `split_ratio` | list[float] | `[0.8, 0.1, 0.1]` | Train/val/test split ratio over scene IDs |
| `seed` | int | `0` | Used for the deterministic split |
| `cache_csm` | bool | `true` | Cache computed CSM tensors in memory |
| `precomputed_csm_root` | string | empty | Optional on-disk cache root for precomputed `S_low`/`S_high` tensors. Missing entries are materialised once and reused across runs |

### `data.eigenscape`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Global dataset enable |
| `root_path` | string | `data/eigenscape` | Dataset root |
| `split_ratio` | list[float] | `[0.75, 0.125, 0.125]` | Ratio-based class-wise split |
| `seed` | int | `0` | Used for the deterministic split |
| `cache_csm` | bool | `true` | Cache computed CSM tensors in memory |
| `precomputed_csm_root` | string | empty | Optional on-disk cache root for precomputed `S_low`/`S_high` tensors. Missing entries are materialised once and reused across runs |
| `expected_channels` | int or null | `32` | Strict channel-count filter. `null` disables the check |
| `target_high_channels` | int | `32` | Target channel count for the high-resolution branch |
| `allow_channel_fallback` | bool | `false` | Pads or truncates mismatched files when enabled |

## Behaviour Notes

| Topic | Detail |
| --- | --- |
| Combined checkpoint format | Most saved wrapper state dicts use both `upsampler.*` and `lam.*` prefixes. `BicubicLAM` checkpoints contain only `lam.*` keys because the bicubic upsampler has no trainable parameters |
| Canonical checkpoint naming | Leaving `training.checkpoint_prefix` empty produces retained end-to-end names such as `srcnnlam_e2e_auxdis`, `srcnnlam_e2e_upfroz`, and `srcnnlam_e2e_auxen`. Bicubic interpolation derives `bicubiclam_e2e_upfroz` because its upsampler is always fixed |
| Resume precedence | `--resume-checkpoint` overrides `training.resume_from_checkpoint`, which overrides `initialisation.resume_checkpoint`; any of those combined checkpoints override the separate warm-start checkpoints |
| Logged training loss | `avg loss`, `train loss`, and `val loss` all refer to the chunk-averaged `loss_total` described above |
| Optimiser layout | Non-GAN wrappers use Adam for the trainable parameters in the active branch. `GANLAM` uses separate Adam optimisers for `upsampler.generator + lam` and `upsampler.discriminator` only when the upsampler remains trainable. Frozen-upsampler training uses one Adam optimiser over the trainable LAM parameters |
| Forward contract | Wrapper models must provide `forward_components(...)` for the trainer |
| CSM caching | `cache_csm` is in-memory and process-local. `precomputed_csm_root` is persistent on-disk caching |
