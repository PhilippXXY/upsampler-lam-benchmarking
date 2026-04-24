# Training Overview

Standalone (`*_dist`) checkpoints were trained locally on an Apple M4 Max (64 GB unified memory, 40-Core GPU).
End-to-end checkpoints were trained on an NVIDIA A100 Tensor Core GPU (Intel Xeon Platinum 8268 host CPU, 40 GB VRAM, 384 GB RAM).

This page documents the common schedules once rather than repeating them on each model page.
Model-specific hyperparameters (architecture, loss function) are on the individual model pages.

---

## Training Modes

| Mode | Entry point | What is trained | Checkpoint format |
| --- | --- | --- | --- |
| **Standalone** (`*_dist`) | `src/train_upsamplers.py` | Upsampler only | Upsampler state dict |
| **End-to-end AuxDis** (`*_e2e_auxdis`) | `src/train_end_to_end.py` | Upsampler + LAM jointly, **no** auxiliary loss | Combined wrapper state dict |
| **End-to-end UpFroz** (`*_e2e_upfroz`) | `src/train_end_to_end.py` | LAM only (upsampler frozen) | Combined wrapper state dict |
| **End-to-end AuxEn** (`*_e2e_auxen`) | `src/train_end_to_end.py` | Upsampler + LAM jointly, **with** auxiliary loss | Combined wrapper state dict |

> `BicubicLAM` only supports `*_e2e_upfroz` because the bicubic upsampler has no trainable parameters.

---

## Shared Data Settings

All training runs use:

| Parameter | Value |
| --- | --- |
| `low_channel_indices` | `[5, 9, 21, 25]` |
| `sampling_rate` | `24000` Hz |
| `nbands` | `9` |

| Dataset | Split ratio | Seed |
| --- | --- | --- |
| AudibleLight | `0.8 / 0.1 / 0.1` (scene-based) | `42` |
| EigenScape | `0.75 / 0.125 / 0.125` (class-wise) | `42` |

---

## Standalone Training Schedule

All standalone upsampler checkpoints (`*_dist`) were produced locally using `config/train_upsamplers.yaml`.

| Parameter | Value |
| --- | --- |
| seed | `42` |
| device | `mps` |
| `frame_batch_size` | `32` |
| `gradient_clip_norm` | `1.0` |

Two sequential stages:

| Stage | Datasets | Epochs | Learning rate | Weight decay | Sampling | Early stopping patience |
| --- | --- | --- | --- | --- | --- | --- |
| `pretrain_audiblelight_light` | AudibleLight only | `8` | `1.0e-4` | `1.0e-4` | `proportional` | `3` |
| `finetune_mixed_light` | AudibleLight + EigenScape | `4` | `5.0e-5` | `1.0e-4` | `balanced` | `2` |

---

## End-to-End Training Schedule

All end-to-end checkpoints share the same stage layout and data settings as standalone training.
What differs between the three modes is listed below.

### Common E2E Parameters

| Parameter | Value |
| --- | --- |
| seed | `42` |
| device | `cuda` |
| `frame_batch_size` | `32` |
| `gradient_clip_norm` | `1.0` |
| LAM loss | `original_msetv` |
| `lam_tv_weight` | `1.0e-5` |
| `use_model_specific_aux_loss` | `true` |

**Stage: `finetune_mixed_light`** — same as standalone (100 epochs, lr `1.0e-4`, balanced, patience `10`).

### Mode-Specific Differences

| Setting | AuxDis | UpFroz | AuxEn |
| --- | --- | --- | --- |
| `freeze_upsampler` | `false` | `true` | `false` |
| `aux_enabled` | `false` | `false` | `true` |
| `aux_weight` | — | — | `0.25` |
| Optimised parameters | upsampler + LAM | LAM only | upsampler + LAM |

### Warm-Start Initialisation

| Source | Standalone checkpoint | Value |
| --- | --- | --- |
| `upsampler_checkpoint` | Matching `*_dist` checkpoint | e.g. `src/upsampler/srcnn/checkpoints/srcnn.pth` |
| `lam_checkpoint` | `src/lam_min/checkpoints/LAM.pth` | — |

> `BicubicLAM` leaves `upsampler_checkpoint` empty; it only warm-starts LAM.

---

## Model-Specific Losses

The training schedule above is identical across models. What varies is each model's reconstruction loss:

| Model | `loss_name` | Extra loss terms |
| --- | --- | --- |
| SRCNN | `mse` | — |
| IMDN | `mse` | — |
| SAFMN | `l1` | `fft_loss_weight: 0.1` |
| GAN | `l1` | `adversarial_weight: 0.01`, `content_weight: 0.01`, `critic_iters: 4`, `discriminator_lr_scale: 5.0` |
| AINN | `mse` | `pde_loss_weight: 0.5`, freq range `100–4000 Hz`, `sound_speed: 343.0` |

See each model's page for architecture-specific hyperparameters.
