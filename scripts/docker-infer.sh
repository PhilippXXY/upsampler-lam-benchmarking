#!/usr/bin/env bash
# =============================================================================
# docker-infer.sh - Wrapper script to run inference workflows in Docker
# =============================================================================
#
# Usage:
#   ./scripts/docker-infer.sh [--runner infer|evaluate] [config_file] [additional_args...]
#
# Example:
#   ./scripts/docker-infer.sh
#   ./scripts/docker-infer.sh /path/to/my_config.yaml
#   ./scripts/docker-infer.sh /path/to/my_config.yaml --device cuda
#   ./scripts/docker-infer.sh --runner evaluate
#   ./scripts/docker-infer.sh --runner evaluate /path/to/my_config.yaml --targets bicubiclam uplam
#
# Without an external config file, the container uses its internal config and
# bundled LOCATA archive/cache (embedded image only), while outputs are written to:
#   ./output
# relative to the directory from which this script is run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_REPO_DEFAULT="${DOCKER_IMAGE_REPO:-ghcr.io/philippxxy/upsampler-lam-benchmarking}"
IMAGE_NAME_STANDARD_DEFAULT="${DOCKER_IMAGE_STANDARD:-${IMAGE_REPO_DEFAULT}:latest}"
IMAGE_NAME_EMBEDDED_DEFAULT="${DOCKER_IMAGE_EMBEDDED:-${IMAGE_REPO_DEFAULT}:latest-embedded}"
IMAGE_NAME="${DOCKER_IMAGE:-}"
DOCKER_PULL="${DOCKER_PULL:-true}"
HOST_BASE_DIR="$(pwd)"
OUTPUT_ROOT_DEFAULT="${DOCKER_OUTPUT_ROOT:-${HOST_BASE_DIR}/output}"
LOGS_ROOT_DEFAULT="${DOCKER_LOGS_ROOT:-${HOST_BASE_DIR}/logs}"
LOCATA_CACHE_ROOT_DEFAULT="${DOCKER_LOCATA_CACHE_ROOT:-${REPO_ROOT}/.docker-locata-cache}"

CONTAINER_APP_ROOT="/app"
CONTAINER_INTERNAL_CONFIG="${CONTAINER_APP_ROOT}/config/inference_config.yaml"
CONTAINER_INTERNAL_LOCATA_PATH="${CONTAINER_APP_ROOT}/data/locata/eval"
CONTAINER_EXTERNAL_CONFIG="/config/inference_config.yaml"
CONTAINER_AUDIO_PATH="/data/audio"
CONTAINER_GT_PATH="/data/ground_truth"
CONTAINER_OUTPUT_PATH="${CONTAINER_APP_ROOT}/output"
CONTAINER_LOGS_PATH="${CONTAINER_APP_ROOT}/logs"
RUNNER_INFER="infer"
RUNNER_EVALUATE="evaluate"

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

usage() {
    local exit_code="${1:-1}"
    echo "Usage: $0 [--runner infer|evaluate] [config_file] [additional_args...]"
    echo ""
    echo "Arguments:"
    echo "  --runner         Workflow to run: 'infer' (default) or 'evaluate'"
    echo "  --evaluate       Shorthand for '--runner evaluate'"
    echo "  config_file       Optional path to a host inference config YAML file"
    echo "  additional_args   Additional arguments passed to infer.py or evaluate_inference.py"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_IMAGE      Override Docker image completely"
    echo "  DOCKER_IMAGE_STANDARD Standard image for host-mounted runs (default: ${IMAGE_NAME_STANDARD_DEFAULT})"
    echo "  DOCKER_IMAGE_EMBEDDED Embedded-data image for internal-data runs (default: ${IMAGE_NAME_EMBEDDED_DEFAULT})"
    echo "  DOCKER_PULL       Pull image only if missing locally (default: ${DOCKER_PULL})"
    echo "  DOCKER_PLATFORM   Optional Docker platform, e.g. linux/arm64"
    echo "  DOCKER_OUTPUT_ROOT Host output directory (default: ${OUTPUT_ROOT_DEFAULT})"
    echo "  DOCKER_LOGS_ROOT  Host logs directory (default: ${LOGS_ROOT_DEFAULT})"
    echo "  DOCKER_LOCATA_CACHE_ROOT Host cache directory for unpacked embedded LOCATA data (default: ${LOCATA_CACHE_ROOT_DEFAULT})"
    echo "  DOCKER_EMBEDDED_SAFE_MODE Enable Pi-safe defaults for embedded images only (default: auto)"
    echo "  DOCKER_CPU_LIMIT  Docker CPU quota for embedded safe mode, passed to --cpus (default: unset)"
    echo "  DOCKER_MEMORY_LIMIT Memory limit for embedded safe mode, passed to --memory (default: unset)"
    echo "  DOCKER_SHM_SIZE   Docker shared memory for embedded safe mode (default: 256m)"
    echo "  DOCKER_NUM_WORKERS_OVERRIDE Override inference.num_workers in embedded safe mode (default: 0)"
    echo "  DOCKER_CPU_THREADS CPU thread cap for embedded safe mode (default: 1)"
    echo ""
    echo "Example:"
    echo "  $0"
    echo "  $0 /home/user/my_config.yaml"
    echo "  $0 /home/user/my_config.yaml --device cuda:0"
    echo "  $0 --runner evaluate"
    echo "  $0 --runner evaluate /home/user/my_config.yaml --targets bicubiclam uplam"
    echo ""
    echo "Pull or build the required image first, for example:"
    echo "  docker pull ${IMAGE_NAME_STANDARD_DEFAULT}"
    echo "  docker pull ${IMAGE_NAME_EMBEDDED_DEFAULT}"
    echo ""
    echo "When no config file is provided, the image uses:"
    echo "  - internal config: ${CONTAINER_INTERNAL_CONFIG}"
    echo "  - internal LOCATA data: ${CONTAINER_INTERNAL_LOCATA_PATH}"
    echo "  - host output path: ${OUTPUT_ROOT_DEFAULT}"
    exit "$exit_code"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

is_truthy() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"

    case "$value" in
        1|true|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        log_error "Missing value for ${flag}"
        exit 1
    fi
}

extract_yaml_value() {
    local key="$1"
    local file="$2"
    grep -E "^[[:space:]]*${key}:" "$file" | head -1 | sed -E 's/^[^:]+:[[:space:]]*"?([^"#]+)"?.*/\1/' | sed 's/[[:space:]]*$//'
}

resolve_path() {
    local base_dir="$1"
    local path_value="$2"

    if [[ -z "$path_value" ]]; then
        return 0
    fi

    if [[ "$path_value" == /* ]]; then
        printf '%s\n' "$path_value"
        return 0
    fi

    printf '%s/%s\n' "$(cd "$base_dir" && pwd)" "$path_value"
}

validate_host_output_path() {
    local path_value="$1"

    if [[ "$path_value" =~ ^/(output|data|logs|app|config)$ ]] || [[ "$path_value" =~ ^/(output|data|logs|app|config)/ ]]; then
        log_error "output_path appears to be a container path: $path_value"
        log_error "Please specify a host path, e.g. /home/user/results"
        exit 1
    fi
}

cleanup() {
    if [[ -n "${DOCKER_CONFIG:-}" && -f "${DOCKER_CONFIG:-}" ]]; then
        rm -f "$DOCKER_CONFIG"
    fi
}
trap cleanup EXIT

RUNNER="$RUNNER_INFER"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage 0
            ;;
        --evaluate)
            RUNNER="$RUNNER_EVALUATE"
            shift
            ;;
        --runner)
            require_value "$1" "${2:-}"
            RUNNER="$2"
            shift 2
            ;;
        --runner=*)
            RUNNER="${1#*=}"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

case "$RUNNER" in
    "$RUNNER_INFER"|"$RUNNER_EVALUATE")
        ;;
    *)
        log_error "Unsupported runner: $RUNNER"
        log_error "Supported runners: $RUNNER_INFER, $RUNNER_EVALUATE"
        exit 1
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker is not installed or not on PATH."
    exit 1
fi

CONFIG_MODE="internal"
CONFIG_FILE=""
CONTAINER_CONFIG_PATH="$CONTAINER_INTERNAL_CONFIG"
HOST_OUTPUT_PATH="$OUTPUT_ROOT_DEFAULT"
HOST_LOGS_PATH="$LOGS_ROOT_DEFAULT"
DATASET_NAME="locata"
AUDIO_PATH=""
GT_PATH=""

if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG_FILE="$1"
    shift

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
    CONFIG_MODE="external"
    CONTAINER_CONFIG_PATH="$CONTAINER_EXTERNAL_CONFIG"

    DATASET_NAME="$(extract_yaml_value "data_set" "$CONFIG_FILE")"
    AUDIO_PATH="$(resolve_path "$HOST_BASE_DIR" "$(extract_yaml_value "data_audio_path" "$CONFIG_FILE")")"
    GT_PATH="$(resolve_path "$HOST_BASE_DIR" "$(extract_yaml_value "data_ground_truth_path" "$CONFIG_FILE")")"

    OUTPUT_PATH_RAW="$(extract_yaml_value "output_path" "$CONFIG_FILE")"
    if [[ -z "$OUTPUT_PATH_RAW" ]]; then
        log_warn "Could not extract 'output_path' from config, using default '${OUTPUT_ROOT_DEFAULT}'"
        HOST_OUTPUT_PATH="$OUTPUT_ROOT_DEFAULT"
        HOST_LOGS_PATH="$LOGS_ROOT_DEFAULT"
    else
        HOST_OUTPUT_PATH="$(resolve_path "$HOST_BASE_DIR" "$OUTPUT_PATH_RAW")"
        HOST_LOGS_PATH="${DOCKER_LOGS_ROOT:-$(dirname "$HOST_OUTPUT_PATH")/logs}"
    fi
fi

HOST_OUTPUT_PATH="$(resolve_path "$HOST_BASE_DIR" "$HOST_OUTPUT_PATH")"
HOST_LOGS_PATH="$(resolve_path "$HOST_BASE_DIR" "$HOST_LOGS_PATH")"

validate_host_output_path "$HOST_OUTPUT_PATH"

if [[ -z "$IMAGE_NAME" ]]; then
    if [[ "$CONFIG_MODE" == "internal" ]]; then
        IMAGE_NAME="$IMAGE_NAME_EMBEDDED_DEFAULT"
    else
        IMAGE_NAME="$IMAGE_NAME_STANDARD_DEFAULT"
    fi
fi

EMBEDDED_IMAGE_SELECTED="false"
case "$IMAGE_NAME" in
    "$IMAGE_NAME_EMBEDDED_DEFAULT"|*embedded*)
        EMBEDDED_IMAGE_SELECTED="true"
        ;;
esac

USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG="false"
if [[ "$CONFIG_MODE" == "external" && "$EMBEDDED_IMAGE_SELECTED" == "true" && "$DATASET_NAME" == "locata" ]]; then
    USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG="true"
fi

if [[ "$CONFIG_MODE" == "internal" && "$EMBEDDED_IMAGE_SELECTED" != "true" ]]; then
    log_error "Internal-config mode requires an embedded Docker image with bundled LOCATA data."
    log_error "Use the default embedded image, set DOCKER_IMAGE to an '*-embedded' tag, or provide an external config file."
    exit 1
fi

EMBEDDED_SAFE_MODE_DEFAULT="false"
if [[ "$EMBEDDED_IMAGE_SELECTED" == "true" ]]; then
    EMBEDDED_SAFE_MODE_DEFAULT="true"
fi

if is_truthy "${DOCKER_EMBEDDED_SAFE_MODE:-$EMBEDDED_SAFE_MODE_DEFAULT}"; then
    EMBEDDED_SAFE_MODE="true"
else
    EMBEDDED_SAFE_MODE="false"
fi

EMBEDDED_MEMORY_LIMIT=""
EMBEDDED_SHM_SIZE=""
EMBEDDED_NUM_WORKERS_OVERRIDE=""
EMBEDDED_CPU_THREADS=""
EMBEDDED_CPU_LIMIT=""

if [[ "$EMBEDDED_SAFE_MODE" == "true" ]]; then
    EMBEDDED_MEMORY_LIMIT="${DOCKER_MEMORY_LIMIT:-}"
    EMBEDDED_SHM_SIZE="${DOCKER_SHM_SIZE:-256m}"
    EMBEDDED_NUM_WORKERS_OVERRIDE="${DOCKER_NUM_WORKERS_OVERRIDE:-0}"
    EMBEDDED_CPU_THREADS="${DOCKER_CPU_THREADS:-1}"
    EMBEDDED_CPU_LIMIT="${DOCKER_CPU_LIMIT:-}"

    if [[ ! "$EMBEDDED_NUM_WORKERS_OVERRIDE" =~ ^[0-9]+$ ]]; then
        log_error "DOCKER_NUM_WORKERS_OVERRIDE must be a non-negative integer, got: $EMBEDDED_NUM_WORKERS_OVERRIDE"
        exit 1
    fi

    if [[ -n "$EMBEDDED_CPU_THREADS" && ! "$EMBEDDED_CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
        log_error "DOCKER_CPU_THREADS must be a positive integer, got: $EMBEDDED_CPU_THREADS"
        exit 1
    fi

    if [[ -n "$EMBEDDED_CPU_LIMIT" && ! "$EMBEDDED_CPU_LIMIT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        log_error "DOCKER_CPU_LIMIT must be a positive number, got: $EMBEDDED_CPU_LIMIT"
        exit 1
    fi
fi

if [[ "$CONFIG_MODE" == "external" ]]; then
    if [[ -z "$DATASET_NAME" ]]; then
        log_error "Could not extract 'data_set' from config file"
        exit 1
    fi

    if [[ "$USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG" != "true" ]]; then
        if [[ -z "$AUDIO_PATH" ]]; then
            log_error "Could not extract 'data_audio_path' from config file"
            exit 1
        fi

        if [[ ! -d "$AUDIO_PATH" ]]; then
            log_error "Audio path does not exist: $AUDIO_PATH"
            exit 1
        fi
    fi

    if [[ "$DATASET_NAME" == "starss23" ]]; then
        if [[ -z "$GT_PATH" ]]; then
            log_error "Could not extract 'data_ground_truth_path' from config file"
            exit 1
        fi

        if [[ ! -d "$GT_PATH" ]]; then
            log_error "Ground truth path does not exist: $GT_PATH"
            exit 1
        fi
    fi
fi

mkdir -p "$HOST_OUTPUT_PATH" "$HOST_LOGS_PATH"

MOUNT_ARGS=(
    -v "$HOST_OUTPUT_PATH:$CONTAINER_OUTPUT_PATH"
    -v "$HOST_LOGS_PATH:$CONTAINER_LOGS_PATH"
)
CONFIG_LOG_DESCRIPTION=""
CONFIG_TEMPLATE_PATH=""
DATA_AUDIO_CONTAINER_PATH="$CONTAINER_AUDIO_PATH"
HOST_LOCATA_CACHE_PATH=""

if [[ "$CONFIG_MODE" == "external" ]]; then
    CONFIG_TEMPLATE_PATH="$CONFIG_FILE"
    CONFIG_LOG_DESCRIPTION="$CONFIG_FILE"
else
    CONFIG_TEMPLATE_PATH="${REPO_ROOT}/config/inference_config.yaml"
    DATA_AUDIO_CONTAINER_PATH="$CONTAINER_INTERNAL_LOCATA_PATH"
fi

if [[ "$USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG" == "true" ]]; then
    DATA_AUDIO_CONTAINER_PATH="$CONTAINER_INTERNAL_LOCATA_PATH"
fi

if [[ "$EMBEDDED_IMAGE_SELECTED" == "true" ]]; then
    HOST_LOCATA_CACHE_PATH="$(resolve_path "$HOST_BASE_DIR" "$LOCATA_CACHE_ROOT_DEFAULT")"
    mkdir -p "$HOST_LOCATA_CACHE_PATH"
    MOUNT_ARGS+=(-v "$HOST_LOCATA_CACHE_PATH:${CONTAINER_APP_ROOT}/data/locata")
fi

GENERATE_CONFIG="false"
if [[ "$CONFIG_MODE" == "external" || -n "$EMBEDDED_NUM_WORKERS_OVERRIDE" ]]; then
    GENERATE_CONFIG="true"
fi

if [[ "$GENERATE_CONFIG" == "true" ]]; then
    if [[ ! -f "$CONFIG_TEMPLATE_PATH" ]]; then
        log_error "Config template not found: $CONFIG_TEMPLATE_PATH"
        exit 1
    fi

    DOCKER_CONFIG="$(mktemp)"
    SED_ARGS=(
        -E
        -e "s|^([[:space:]]*data_audio_path:[[:space:]]*).*$|\\1\"${DATA_AUDIO_CONTAINER_PATH}\"|"
        -e "s|^([[:space:]]*output_path:[[:space:]]*).*$|\\1\"${CONTAINER_OUTPUT_PATH}\"|"
        -e "s|^([[:space:]]*filename:[[:space:]]*)\"?logs/|\\1\"${CONTAINER_LOGS_PATH}/|"
    )

    if [[ -n "$EMBEDDED_NUM_WORKERS_OVERRIDE" ]]; then
        SED_ARGS+=(
            -e "s|^([[:space:]]*num_workers:[[:space:]]*).*$|\\1${EMBEDDED_NUM_WORKERS_OVERRIDE}|"
        )
    fi

    if [[ "$CONFIG_MODE" == "external" && "$DATASET_NAME" == "starss23" ]]; then
        SED_ARGS+=(
            -e "s|^([[:space:]]*data_ground_truth_path:[[:space:]]*).*$|\\1\"${CONTAINER_GT_PATH}\"|"
        )
        MOUNT_ARGS+=(-v "$GT_PATH:$CONTAINER_GT_PATH:ro")
    fi

    sed "${SED_ARGS[@]}" "$CONFIG_TEMPLATE_PATH" > "$DOCKER_CONFIG"
    MOUNT_ARGS+=(-v "$DOCKER_CONFIG:$CONTAINER_EXTERNAL_CONFIG:ro")
    CONTAINER_CONFIG_PATH="$CONTAINER_EXTERNAL_CONFIG"

    if [[ "$CONFIG_MODE" == "external" ]]; then
        CONFIG_LOG_DESCRIPTION="generated from $CONFIG_FILE"
        if [[ "$USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG" != "true" ]]; then
            MOUNT_ARGS+=(-v "$AUDIO_PATH:$CONTAINER_AUDIO_PATH:ro")
        fi
    else
        CONFIG_LOG_DESCRIPTION="generated from ${REPO_ROOT}/config/inference_config.yaml"
    fi
fi

if [[ "$CONFIG_MODE" == "external" ]]; then
    log_info "Using external config file: $CONFIG_FILE"
    if [[ "$USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG" == "true" ]]; then
        log_info "Using bundled LOCATA data: $CONTAINER_INTERNAL_LOCATA_PATH"
    fi
else
    if [[ "$GENERATE_CONFIG" == "true" ]]; then
        log_info "Using generated embedded-safe config: $CONTAINER_CONFIG_PATH"
    else
        log_info "Using internal config file: $CONTAINER_INTERNAL_CONFIG"
    fi
    log_info "Using bundled LOCATA data: $CONTAINER_INTERNAL_LOCATA_PATH"
fi
log_info "Runner: $RUNNER"
if [[ "$EMBEDDED_SAFE_MODE" == "true" ]]; then
    EMBEDDED_CPU_LIMIT_DISPLAY="${EMBEDDED_CPU_LIMIT:-all available}"
    EMBEDDED_MEMORY_LIMIT_DISPLAY="${EMBEDDED_MEMORY_LIMIT:-all available}"
    log_info "Embedded resource settings: cpu_limit=${EMBEDDED_CPU_LIMIT_DISPLAY}, memory_limit=${EMBEDDED_MEMORY_LIMIT_DISPLAY}, cpu_threads=${EMBEDDED_CPU_THREADS}, num_workers=${EMBEDDED_NUM_WORKERS_OVERRIDE}, shm=${EMBEDDED_SHM_SIZE}"
fi

log_info "Mounting volumes:"
if [[ "$CONFIG_MODE" == "external" ]]; then
    if [[ "$USE_BUNDLED_LOCATA_WITH_EXTERNAL_CONFIG" == "true" ]]; then
        log_info "  LOCATA data:   bundled image data at $CONTAINER_INTERNAL_LOCATA_PATH"
        if [[ -n "$HOST_LOCATA_CACHE_PATH" ]]; then
            log_info "  LOCATA cache:  $HOST_LOCATA_CACHE_PATH -> ${CONTAINER_APP_ROOT}/data/locata"
        fi
    else
        log_info "  Audio data:    $AUDIO_PATH -> $CONTAINER_AUDIO_PATH"
    fi
    if [[ "$DATASET_NAME" == "starss23" ]]; then
        log_info "  Ground truth:  $GT_PATH -> $CONTAINER_GT_PATH"
    fi
    if [[ -n "$CONFIG_LOG_DESCRIPTION" ]]; then
        log_info "  Config:        $CONFIG_LOG_DESCRIPTION -> $CONTAINER_CONFIG_PATH"
    fi
else
    log_info "  LOCATA data:   internal image data at $CONTAINER_INTERNAL_LOCATA_PATH"
    if [[ -n "$HOST_LOCATA_CACHE_PATH" ]]; then
        log_info "  LOCATA cache:  $HOST_LOCATA_CACHE_PATH -> ${CONTAINER_APP_ROOT}/data/locata"
    fi
    if [[ -n "$CONFIG_LOG_DESCRIPTION" ]]; then
        log_info "  Config:        $CONFIG_LOG_DESCRIPTION -> $CONTAINER_CONFIG_PATH"
    fi
fi
log_info "  Output:        $HOST_OUTPUT_PATH -> $CONTAINER_OUTPUT_PATH"
log_info "  Logs:          $HOST_LOGS_PATH -> $CONTAINER_LOGS_PATH"

log_info "Running Docker container..."
log_info "Image: $IMAGE_NAME"

DOCKER_INFO_OUTPUT=""
if ! DOCKER_INFO_OUTPUT="$(docker info 2>&1)"; then
    if [[ -n "$DOCKER_INFO_OUTPUT" ]]; then
        printf '%s\n' "$DOCKER_INFO_OUTPUT" >&2
    fi
    log_error "Cannot access the Docker daemon."
    if [[ "$DOCKER_INFO_OUTPUT" == *"permission denied while trying to connect to the docker API"* ]] || [[ "$DOCKER_INFO_OUTPUT" == *"/var/run/docker.sock"* ]]; then
        log_error "Your current user likely does not have permission to access /var/run/docker.sock."
        log_error "On Linux, add your user to the docker group and start a new login session:"
        log_error "  sudo usermod -aG docker \$USER"
        log_error "  newgrp docker"
        log_error "  # Or log out and back in"
    else
        log_error "Make sure Docker is running and the daemon is reachable."
    fi
    exit 1
fi

if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    log_info "Using local Docker image."
elif [[ "$DOCKER_PULL" == "true" ]]; then
    log_info "Local image not found, pulling from registry..."
    PULL_OUTPUT=""
    if ! PULL_OUTPUT="$(docker pull "$IMAGE_NAME" 2>&1)"; then
        if [[ -n "$PULL_OUTPUT" ]]; then
            printf '%s\n' "$PULL_OUTPUT" >&2
        fi
        log_error "Failed to pull image '$IMAGE_NAME'"
        log_error ""
        if [[ "$IMAGE_NAME" == ghcr.io/* ]]; then
            if [[ "$PULL_OUTPUT" == *"403 Forbidden"* || "$PULL_OUTPUT" == *"denied"* ]]; then
                log_error "GHCR denied access to the image."
                log_error "This usually means the package is private, or your Docker credential lacks package-read access."
                log_error "A plain 'docker login ghcr.io' can also reuse stale cached credentials."
                log_error ""
                log_error "Try refreshing credentials explicitly:"
                log_error "  docker logout ghcr.io"
                log_error "  echo \$GITHUB_TOKEN | docker login ghcr.io --username YOUR_USERNAME --password-stdin"
                log_error ""
                log_error "For private packages, the token typically needs package read access and repository access."
                log_error "If you own the package, also verify the package visibility/access settings on GitHub."
            elif [[ "$PULL_OUTPUT" == *"not found"* || "$PULL_OUTPUT" == *"manifest unknown"* ]]; then
                log_error "GHCR resolved the image tag, but the published manifest or blob content is missing."
                log_error "This usually means the registry artifact for that tag is incomplete or corrupted."
                log_error "Try a different tag, republish the image, or build it locally."
            else
                log_error "GHCR returned an unexpected pull error."
                log_error "Review the Docker error above for the exact registry response."
            fi
        else
            log_error "For private repositories, authenticate first:"
            log_error "  docker login ghcr.io"
            log_error "  # Or using a personal access token:"
            log_error "  echo \$GITHUB_TOKEN | docker login ghcr.io --username YOUR_USERNAME --password-stdin"
        fi
        exit 1
    elif [[ -n "$PULL_OUTPUT" ]]; then
        printf '%s\n' "$PULL_OUTPUT"
    fi
else
    log_error "Docker image '$IMAGE_NAME' not found locally and DOCKER_PULL is disabled."
    log_error "Pull or build the image first, for example:"
    log_error "  docker pull $IMAGE_NAME"
    exit 1
fi

RUN_ARGS=(--rm)

if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
    RUN_ARGS+=(--platform "$DOCKER_PLATFORM")
fi

if [[ -n "$EMBEDDED_CPU_LIMIT" ]]; then
    RUN_ARGS+=(--cpus "$EMBEDDED_CPU_LIMIT")
fi

if [[ -n "$EMBEDDED_MEMORY_LIMIT" ]]; then
    RUN_ARGS+=(--memory "$EMBEDDED_MEMORY_LIMIT")
fi

if [[ -n "$EMBEDDED_SHM_SIZE" ]]; then
    RUN_ARGS+=(--shm-size "$EMBEDDED_SHM_SIZE")
fi

RUN_ARGS+=(
    -e HOME=/tmp
    -e UV_CACHE_DIR=/tmp/uv-cache
    -e XDG_CACHE_HOME=/tmp/.cache
    -e TORCH_HOME=/tmp/torch
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
)

if [[ -n "$EMBEDDED_CPU_THREADS" ]]; then
    RUN_ARGS+=(
        -e OMP_NUM_THREADS="$EMBEDDED_CPU_THREADS"
        -e OPENBLAS_NUM_THREADS="$EMBEDDED_CPU_THREADS"
        -e MKL_NUM_THREADS="$EMBEDDED_CPU_THREADS"
        -e NUMEXPR_NUM_THREADS="$EMBEDDED_CPU_THREADS"
        -e BLIS_NUM_THREADS="$EMBEDDED_CPU_THREADS"
        -e MPLCONFIGDIR=/tmp/matplotlib
        -e PYTHONPYCACHEPREFIX=/tmp/pycache
    )
fi

if command -v id >/dev/null 2>&1; then
    HOST_USERNAME="${USER:-}"
    if [[ -z "$HOST_USERNAME" ]]; then
        HOST_USERNAME="$(id -un 2>/dev/null || true)"
    fi
    if [[ -z "$HOST_USERNAME" ]]; then
        HOST_USERNAME="appuser"
    fi

    RUN_ARGS+=(
        -e USER="$HOST_USERNAME"
        -e LOGNAME="$HOST_USERNAME"
        -e USERNAME="$HOST_USERNAME"
        -e LNAME="$HOST_USERNAME"
    )
    RUN_ARGS+=(--user "$(id -u):$(id -g)")
fi

COMMAND_ARGS=()

if [[ "$RUNNER" == "$RUNNER_INFER" ]]; then
    for arg in "$@"; do
        if [[ "$arg" == "--config" || "$arg" == "--config="* ]]; then
            log_error "Do not pass --config in additional arguments. Use the positional config_file instead."
            exit 1
        fi
    done

    COMMAND_ARGS=(--config "$CONTAINER_CONFIG_PATH")
    if [[ $# -gt 0 ]]; then
        COMMAND_ARGS+=("$@")
    fi
else
    EXTRA_EVAL_MOUNTS=()
    EVAL_ARGS=()
    EVAL_OUTPUT_INDEX=0

    while [[ $# -gt 0 ]]; do
        arg="$1"
        shift

        case "$arg" in
            --base-config|--base-config=*)
                log_error "Do not pass --base-config in additional arguments. Use the positional config_file instead."
                exit 1
                ;;
            --output-csv|--output-plot)
                require_value "$arg" "${1:-}"
                output_host_path="$(resolve_path "$HOST_BASE_DIR" "$1")"
                validate_host_output_path "$output_host_path"
                output_host_dir="$(dirname "$output_host_path")"
                mkdir -p "$output_host_dir"
                EVAL_OUTPUT_INDEX=$((EVAL_OUTPUT_INDEX + 1))
                output_container_dir="/benchmark-output-${EVAL_OUTPUT_INDEX}"
                output_container_path="${output_container_dir}/$(basename "$output_host_path")"
                EXTRA_EVAL_MOUNTS+=(-v "$output_host_dir:$output_container_dir")
                EVAL_ARGS+=("$arg" "$output_container_path")
                shift
                ;;
            --output-csv=*|--output-plot=*)
                output_host_path="$(resolve_path "$HOST_BASE_DIR" "${arg#*=}")"
                validate_host_output_path "$output_host_path"
                output_host_dir="$(dirname "$output_host_path")"
                mkdir -p "$output_host_dir"
                EVAL_OUTPUT_INDEX=$((EVAL_OUTPUT_INDEX + 1))
                output_container_dir="/benchmark-output-${EVAL_OUTPUT_INDEX}"
                output_container_path="${output_container_dir}/$(basename "$output_host_path")"
                EXTRA_EVAL_MOUNTS+=(-v "$output_host_dir:$output_container_dir")
                EVAL_ARGS+=("${arg%%=*}=$output_container_path")
                ;;
            *)
                EVAL_ARGS+=("$arg")
                ;;
        esac
    done

    if [[ ${#EXTRA_EVAL_MOUNTS[@]} -gt 0 ]]; then
        MOUNT_ARGS+=("${EXTRA_EVAL_MOUNTS[@]}")
    fi
    if [[ "$EMBEDDED_IMAGE_SELECTED" == "true" ]]; then
        COMMAND_ARGS=(
            python
            src/evaluate_inference.py
            --base-config "$CONTAINER_CONFIG_PATH"
        )
    else
        RUN_ARGS+=(--entrypoint python)
        COMMAND_ARGS=(
            src/evaluate_inference.py
            --base-config "$CONTAINER_CONFIG_PATH"
        )
    fi
    if [[ ${#EVAL_ARGS[@]} -gt 0 ]]; then
        COMMAND_ARGS+=("${EVAL_ARGS[@]}")
    fi
fi

docker run "${RUN_ARGS[@]}" \
    "${MOUNT_ARGS[@]}" \
    "$IMAGE_NAME" \
    "${COMMAND_ARGS[@]}"

if [[ "$RUNNER" == "$RUNNER_EVALUATE" ]]; then
    log_info "Benchmark evaluation completed."
else
    log_info "Inference completed."
fi
log_info "Results saved to: $HOST_OUTPUT_PATH"
