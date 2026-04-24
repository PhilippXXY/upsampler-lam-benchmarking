# Model Overview

This page lists all model families available in the repository.
For architecture details, see each model's page. For the shared training schedule, see [Training Overview](training-overview.md).

## Model Inventory

| Family | Variants | Role | Architecture | Docs | Refs |
| --- | --- | --- | --- | --- | --- |
| LAM | `lam` | Upstream baseline | Direct 32-ch localisation model | [LAM and UpLAM](lam-family.md) | [R1](../references.md#r1), [R2](../references.md#r2) |
| UpLAM | `uplam_dist`, `uplam_e2e_auxdis`, `uplam_e2e_upfroz` | Upstream wrapper baseline | Complex-valued deep back-projection upsampler + LAM | [LAM and UpLAM](lam-family.md) | [R1](../references.md#r1), [R2](../references.md#r2) |
| Bicubic | `bicubiclam_dist`, `bicubiclam_e2e_upfroz` | Repository baseline | Fixed bicubic interpolation + Hermitian projection | [Bicubic](bicubic.md) | [R7](../references.md#r7) |
| SRCNN | `srcnnlam_dist`, `srcnnlam_e2e_auxdis`, `srcnnlam_e2e_upfroz` | Repository upsampler | Shallow convolutional super-resolution | [SRCNN](srcnn.md) | [R8](../references.md#r8), [R9](../references.md#r9) |
| IMDN | `imdnlam_dist`, `imdnlam_e2e_auxdis`, `imdnlam_e2e_upfroz` | Repository upsampler | Lightweight feature-distillation model | [IMDN](imdn.md) | [R10](../references.md#r10), [R11](../references.md#r11) |
| SAFMN | `safmnlam_dist`, `safmnlam_e2e_auxdis`, `safmnlam_e2e_upfroz` | Repository upsampler | Modulation-based convolutional model | [SAFMN](safmn.md) | [R12](../references.md#r12), [R13](../references.md#r13) |
| GAN | `ganlam_dist`, `ganlam_e2e_auxdis`, `ganlam_e2e_upfroz` | Repository upsampler | Residual generator + adversarial discriminator | [GAN](gan.md) | [R14](../references.md#r14), [R15](../references.md#r15) |
| AINN | `ainnlam_dist`, `ainnlam_e2e_auxdis`, `ainnlam_e2e_upfroz` | Repository upsampler | Coordinate-based implicit network + PDE regularisation | [AINN](ainn.md) | [R16](../references.md#r16), [R17](../references.md#r17) |

## Variant Naming Convention

| Suffix | Meaning |
| --- | --- |
| `*_dist` | **Distinct** — standalone upsampler checkpoint + separate `LAM.pth` |
| `*_e2e_auxdis` | **End-to-end, auxiliary disabled** — joint upsampler + LAM training without auxiliary loss |
| `*_e2e_upfroz` | **End-to-end, upsampler frozen** — only LAM parameters are updated |
| `*_e2e_auxen` | **End-to-end, auxiliary enabled** — joint training with auxiliary upsampler loss (experimental, not included in standard benchmarks) |
