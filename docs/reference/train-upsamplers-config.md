# Upsampler Training Config

Full key reference for `config/train_upsamplers.yaml`.

## Top-Level Structure

| Section | Purpose |
| --- | --- |
| `training` | Global trainer settings and stage schedule |
| `model` | Standalone upsampler selection and architecture hyperparameters |
| `data` | Dataset and CSM construction settings |

## `training`

### Global trainer settings

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `seed` | int | `42` | Passed to the repository seed helper |
| `device` | string | `mps` | Requested training device |
| `epochs` | int | `8` | Global default used when a stage omits its own value |
| `learning_rate` | float | `1.0e-4` | Global stage default |
| `weight_decay` | float | `1.0e-4` | Global stage default |
| `frame_batch_size` | int | `32` | Number of frames processed per optimiser step |
| `gradient_clip_norm` | float | `1.0` | `0.0` disables clipping |
| `num_workers` | int | `0` | DataLoader worker count |
| `max_train_files` | int | `0` | `0` means all files |
| `max_val_files` | int | `0` | `0` means all files |
| `log_every_files` | int | `10` | File-level logging cadence |
| `early_stopping_patience` | int | `3` | Global stage default |
| `early_stopping_min_delta` | float | `0.0` | Global stage default |
| `resume_from_checkpoint` | string | empty | Optional checkpoint used to initialise model weights before training. This is the config-level equivalent of `--resume-checkpoint` |
| `output_root` | string | `output/training` | Root directory for logs and metrics JSON |
| `checkpoint_dir` | string | `src/upsampler/imdn/checkpoints` | Output directory for checkpoints |
| `checkpoint_prefix` | string | `imdn` | Prefix used for generated checkpoint filenames |

### `training.stages[]`

| Key | Type | Meaning |
| --- | --- | --- |
| `name` | string | Stage label used in logs and checkpoint names |
| `enabled` | bool | If `false`, the stage is skipped |
| `epochs` | int | Epoch count for this stage |
| `learning_rate` | float | Stage-specific optimiser learning rate |
| `weight_decay` | float | Stage-specific weight decay |
| `max_train_files` | int | Optional cap on training files for this stage |
| `max_val_files` | int | Optional cap on validation files for this stage |
| `train_sampling` | string | `proportional` or `balanced` |
| `early_stopping_patience` | int | Early stopping patience for this stage |
| `early_stopping_min_delta` | float | Minimum validation improvement for this stage |
| `datasets.audiblelight` | bool | Enable or disable AudibleLight in this stage |
| `datasets.eigenscape` | bool | Enable or disable EigenScape in this stage |

## `model`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | `IMDNUpsampler` | Selects the standalone model class |
| `in_channels` | int | `4` | Low-resolution microphone count |
| `out_channels` | int | `32` | High-resolution microphone count |
| `feature_channels` | int | `64` | Used by SRCNN, IMDN, and SAFMN. GAN fixes this internally to `128` |
| `mapping_channels` | int | `32` | Used by SRCNN and IMDN |
| `n_blocks` | int | `8` | Used by SAFMN and by the config as a GAN residual-block default |
| `ffn_scale` | float | `2.0` | Used by SAFMN |
| `n_levels` | int | `4` | Used by SAFMN |
| `hidden_channels` | int | `64` | Used by AINN |
| `loss_name` | string | `mse` | Reconstruction loss selector. Behaviour is model-specific |
| `fft_loss_weight` | float | `0.1` | Used by SAFMN's spectral loss term |
| `pde_loss_weight` | float | `0.5` | Used by AINN |
| `pde_freq_min_hz` | float | `100.0` | Used by AINN |
| `pde_freq_max_hz` | float | `4000.0` | Used by AINN |
| `sound_speed` | float | `343.0` | Used by AINN |
| `adversarial_weight` | float | `0.01` | Used by GAN |
| `content_weight` | float | `0.01` | Used by GAN |
| `critic_iters` | int | `4` | Generator update frequency for GAN training |
| `discriminator_lr_scale` | float | `5.0` | Discriminator learning-rate multiplier for GAN |
| `beta1_adam` | float | `0.9` | GAN Adam beta |
| `beta2_adam` | float | `0.999` | GAN Adam beta |

## `data`

### Shared CSM settings

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `low_channel_indices` | list[int] | `[5, 9, 21, 25]` | Zero-based Eigenmike channels used for the low-resolution branch. AINN also uses them to build the low-resolution microphone geometry |
| `sampling_rate` | int | `24000` | Target sample rate for CSM generation |
| `nbands` | int | `9` | Number of frequency bands passed into `get_visibility_matrix` |

### `data.audiblelight`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Global dataset enable |
| `root_path` | string | `data/AudibleLight_Eigenmike32-5_DCASE-STARSS23_Dataset` | Dataset root |
| `split_ratio` | list[float] | `[0.8, 0.1, 0.1]` | Train/val/test split ratio over scene IDs |
| `seed` | int | `42` | Used for the deterministic split |
| `cache_csm` | bool | `true` | Cache computed CSM tensors in memory |
| `precomputed_csm_root` | string | empty | Optional on-disk cache root for precomputed `S_low`/`S_high` tensors. Missing entries are materialised once and reused across runs |

### `data.eigenscape`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Global dataset enable |
| `root_path` | string | `data/eigenscape` | Dataset root |
| `split_ratio` | list[float] | `[0.75, 0.125, 0.125]` | Ratio-based class-wise split |
| `seed` | int | `42` | Used for the deterministic split |
| `cache_csm` | bool | `true` | Cache computed CSM tensors in memory |
| `precomputed_csm_root` | string | empty | Optional on-disk cache root for precomputed `S_low`/`S_high` tensors. Missing entries are materialised once and reused across runs |
| `expected_channels` | int or null | `32` | Strict channel-count filter; `null` disables the check |
| `target_high_channels` | int | `32` | Target channel count for the high-resolution branch |
| `allow_channel_fallback` | bool | `false` | Pads or truncates mismatched files when enabled |

## Behaviour Notes

| Topic | Detail |
| --- | --- |
| File-level batching | The trainer expects dataset items to represent whole files. It handles frame chunking internally |
| `balanced` sampling | Uses a weighted sampler so both enabled datasets contribute roughly equally |
| Bicubic training | `BicubicUpsampler` has no trainable parameters and therefore acts as a fixed baseline with loss reporting |
| Resume precedence | `--resume-checkpoint` overrides `training.resume_from_checkpoint` |
| CSM caching | `cache_csm` is in-memory and process-local. `precomputed_csm_root` is persistent on-disk caching |
