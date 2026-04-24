# Docker Inference

Containerised inference without local Python environment management.

## Wrapper Interface

```bash
./scripts/docker-infer.sh [--runner infer|evaluate] [config_file] [additional_args...]
```

## Wrapper Arguments

| Input | Meaning |
| --- | --- |
| `--runner infer` | Run `src/infer.py` inside the container |
| `--runner evaluate` | Run `src/evaluate_inference.py` inside the container |
| `--evaluate` | Shorthand for `--runner evaluate` |
| `config_file` | Optional host-side inference config. If omitted, the embedded image uses its internal config |
| `additional_args...` | Extra arguments passed through to `infer.py` or `evaluate_inference.py` |

Reserved arguments:

- do not pass `--config` as an additional argument. Use the positional `config_file`
- do not pass `--base-config` as an additional argument to the evaluate runner. Use the positional `config_file`

## Two Operating Modes

| Mode | Image expectation | Config source | Dataset source |
| --- | --- | --- | --- |
| Internal config | Embedded image | `/app/config/inference_config.yaml` inside the image | Bundled LOCATA subset |
| External config | Standard or embedded image | Host YAML rewritten to container mount paths | Host-mounted data, except the special bundled-LOCATA case for embedded images |

## Common Commands

```bash
# Embedded image, internal config, bundled LOCATA subset
./scripts/docker-infer.sh

# External config on the standard image
./scripts/docker-infer.sh /path/to/inference_config.yaml

# Benchmark several retained targets in Docker
./scripts/docker-infer.sh --runner evaluate /path/to/inference_config.yaml --targets bicubiclam uplam

# Conservative embedded run
DOCKER_CPU_LIMIT=1 \
DOCKER_MEMORY_LIMIT=1536m \
DOCKER_CPU_THREADS=1 \
DOCKER_NUM_WORKERS_OVERRIDE=0 \
DOCKER_SHM_SIZE=256m \
./scripts/docker-infer.sh
```

## Docker Environment Variables

### Wrapper selection and image resolution

| Variable | Meaning |
| --- | --- |
| `DOCKER_IMAGE_REPO` | Base image repository used to derive default tags |
| `DOCKER_IMAGE` | Full image override Bypasses the standard/embedded default selection |
| `DOCKER_IMAGE_STANDARD` | Default image used for host-mounted runs |
| `DOCKER_IMAGE_EMBEDDED` | Default image used for bundled-data runs |
| `DOCKER_PULL` | If `true`, pull the image when it is not present locally |
| `DOCKER_PLATFORM` | Optional `docker run --platform` value, for example `linux/arm64` |

### Host path control

| Variable | Meaning |
| --- | --- |
| `DOCKER_OUTPUT_ROOT` | Host output directory mounted into `/app/output` |
| `DOCKER_LOGS_ROOT` | Host log directory mounted into `/app/logs` |
| `DOCKER_LOCATA_CACHE_ROOT` | Host cache directory used to persist the unpacked embedded LOCATA archive |

### Embedded-safe mode

| Variable | Meaning |
| --- | --- |
| `DOCKER_EMBEDDED_SAFE_MODE` | Enables or disables the resource-constrained embedded mode |
| `DOCKER_CPU_LIMIT` | Passed through as `docker run --cpus` in embedded-safe mode |
| `DOCKER_MEMORY_LIMIT` | Passed through as `docker run --memory` in embedded-safe mode |
| `DOCKER_SHM_SIZE` | Passed through as `docker run --shm-size` in embedded-safe mode |
| `DOCKER_NUM_WORKERS_OVERRIDE` | Rewrites `inference.num_workers` in the generated config |
| `DOCKER_CPU_THREADS` | Sets OpenMP/BLAS-style thread counts inside the container |

### Embedded-image environment variables

These are read by `scripts/docker-embedded-entrypoint.sh` rather than by the host wrapper:

| Variable | Meaning |
| --- | --- |
| `UPSAMPLER_LOCATA_ROOT` | Container-side root directory for the unpacked LOCATA bundle |
| `UPSAMPLER_LOCATA_ARCHIVE` | Location of the bundled LOCATA archive inside the image |
| `UPSAMPLER_LOCATA_ARCHIVE_SHA_FILE` | Checksum file used to decide whether the cache is still valid |
| `UPSAMPLER_LOCATA_READY_MARKER` | Marker file used after extraction |
| `UPSAMPLER_DEFAULT_CONFIG_PATH` | Default config path used when the entrypoint launches `infer.py` directly |

## Path Rewriting Behaviour

When the wrapper generates a container config, it rewrites:

- `dataset.data_audio_path`
- `dataset.data_ground_truth_path` for STARSS23 runs
- `inference.output_path`
- file logging paths under `inference.logging.handlers`
- `inference.num_workers` when `DOCKER_NUM_WORKERS_OVERRIDE` is set

## Evaluate Runner Notes

For `--runner evaluate`, the wrapper:

- rewrites `--output-csv` and `--output-plot` host paths to temporary container mount points
- passes the generated or mounted config via `--base-config`
- forwards extra evaluation arguments after that

Supported retained targets are documented in [Batch Benchmarking](batch-benchmarking.md).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Internal-config mode requires an embedded Docker image` | You ran the no-config path against a non-embedded image |
| `Audio path does not exist` | The host path from the external config could not be resolved |
| `Cannot access the Docker daemon` | Docker is stopped or the current user cannot access the socket |
| `403 Forbidden` when pulling from GHCR | Package visibility or Docker credentials are insufficient |
| `Docker image not found locally and DOCKER_PULL is disabled` | `DOCKER_PULL=false` and the image is not already present |
