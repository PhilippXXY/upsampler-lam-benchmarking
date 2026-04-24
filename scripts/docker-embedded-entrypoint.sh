#!/usr/bin/env sh
set -eu

LOCATA_ROOT="${UPSAMPLER_LOCATA_ROOT:-/app/data/locata}"
LOCATA_ARCHIVE="${UPSAMPLER_LOCATA_ARCHIVE:-/opt/locata/dev.zip}"
LOCATA_ARCHIVE_SHA_FILE="${UPSAMPLER_LOCATA_ARCHIVE_SHA_FILE:-/opt/locata/dev.zip.sha256}"
LOCATA_READY_MARKER="${UPSAMPLER_LOCATA_READY_MARKER:-${LOCATA_ROOT}/.dev-ready}"
LOCATA_DEV_DIR="${LOCATA_ROOT}/dev"
DEFAULT_CONFIG_PATH="${UPSAMPLER_DEFAULT_CONFIG_PATH:-/app/config/inference_config.yaml}"

log() {
    printf '[embedded-entrypoint] %s\n' "$1"
}

ensure_locata_ready() {
    if [ ! -f "$LOCATA_ARCHIVE" ]; then
        log "archive missing at ${LOCATA_ARCHIVE}"
        exit 1
    fi

    mkdir -p "$LOCATA_ROOT"

    expected_checksum=""
    if [ -f "$LOCATA_ARCHIVE_SHA_FILE" ]; then
        expected_checksum="$(cat "$LOCATA_ARCHIVE_SHA_FILE")"
    fi

    marker_checksum=""
    if [ -f "$LOCATA_READY_MARKER" ]; then
        marker_checksum="$(cat "$LOCATA_READY_MARKER")"
    fi

    if [ -d "$LOCATA_DEV_DIR" ] && [ "$marker_checksum" = "$expected_checksum" ] && [ -n "$marker_checksum" ]; then
        return
    fi

    log "preparing bundled LOCATA dev data at ${LOCATA_DEV_DIR}"
    rm -rf "$LOCATA_DEV_DIR" "${LOCATA_ROOT}/__MACOSX" "$LOCATA_READY_MARKER"

    unzip -oq "$LOCATA_ARCHIVE" -d "$LOCATA_ROOT"
    rm -rf "${LOCATA_ROOT}/__MACOSX"
    find "$LOCATA_ROOT" -name '.DS_Store' -delete

    if [ ! -d "$LOCATA_DEV_DIR" ]; then
        log "expected directory ${LOCATA_DEV_DIR} was not created by archive extraction"
        exit 1
    fi

    if [ -n "$expected_checksum" ]; then
        printf '%s\n' "$expected_checksum" > "$LOCATA_READY_MARKER"
    else
        printf 'ready\n' > "$LOCATA_READY_MARKER"
    fi

    log "LOCATA extraction completed"
}

ensure_locata_ready

if [ "$#" -eq 0 ]; then
    set -- python src/infer.py --config "$DEFAULT_CONFIG_PATH"
elif [ "${1#-}" != "$1" ]; then
    set -- python src/infer.py "$@"
fi

exec "$@"
