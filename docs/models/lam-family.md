# LAM and UpLAM

`LAM` and `UpLAM` are **upstream methods** used as baselines and references rather than repository-native model contributions.
See [R1](../references.md#r1) and [R2](../references.md#r2).

## Role

| Model | Role |
| --- | --- |
| `LAM` | Direct 32-channel baseline on LOCATA; also used as the back-end for all wrapper models |
| `UpLAM` | Standalone 4→32 wrapper baseline |

> This site explains how LAM and UpLAM are wired into the repository — for their original training methodology, see the upstream references.

## Parameter Counts

| Model | Total parameters |
| --- | --- |
| `LAM` | `0.146 M` |
| `UpLAM` | `2.840 M` |

## Checkpoints

| Variant | Path | Notes |
| --- | --- | --- |
| `lam` | `src/lam_min/checkpoints/LAM.pth` | Upstream retained checkpoint |
| `uplam_dist` | `src/lam_min/checkpoints/UpLAM.pth` | Upstream retained checkpoint |
| `uplam_e2e_auxdis` | [`src/lam_min/checkpoints/e2e/uplam_e2e_auxdis.pth`](https://github.com/PhilippXXY/upsampler-lam-benchmarking/blob/main/src/lam_min/checkpoints/e2e/uplam_e2e_auxdis.pth) | Joint upsampler + LAM, no auxiliary loss |
| `uplam_e2e_upfroz` | [`src/lam_min/checkpoints/e2e/uplam_e2e_upfroz.pth`](https://github.com/PhilippXXY/upsampler-lam-benchmarking/blob/main/src/lam_min/checkpoints/e2e/uplam_e2e_upfroz.pth) | Frozen upsampler, LAM-only training |

End-to-end checkpoints were trained with the shared schedule documented in [Training Overview](training-overview.md).

## Practical Notes

- `LAM` expects the full 32-channel LOCATA path.
- `UpLAM` follows the 4-channel wrapper path.
- `lam` and `uplam_dist` are loaded directly in `src/infer.py`.
