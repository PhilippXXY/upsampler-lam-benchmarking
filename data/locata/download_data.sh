#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}"
TARGET_DIR_SET=0

usage() {
  echo "Usage: bash download_data.sh [--target-dir DIR] [DIR]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      if [[ $# -lt 2 ]]; then
        echo "[LOCATA] Missing value for --target-dir"
        usage
        exit 1
      fi
      TARGET_DIR="$2"
      TARGET_DIR_SET=1
      shift 2
      ;;
    --target-dir=*)
      TARGET_DIR="${1#*=}"
      TARGET_DIR_SET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[LOCATA] Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [[ "${TARGET_DIR_SET}" == "1" ]]; then
        echo "[LOCATA] Unexpected extra argument: $1"
        usage
        exit 1
      fi
      TARGET_DIR="$1"
      TARGET_DIR_SET=1
      shift
      ;;
  esac
done

mkdir -p "${TARGET_DIR}"
TARGET_DIR="$(cd "${TARGET_DIR}" && pwd)"
ARCHIVE_DIR="${TARGET_DIR}/archives"
mkdir -p "${ARCHIVE_DIR}"

download_file() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "[LOCATA] Skip existing: ${out}"
    return
  fi
  echo "[LOCATA] Downloading: ${url}"
  curl -L "${url}" -o "${out}"
}

extract_zip() {
  local zip_path="$1"
  echo "[LOCATA] Extracting: ${zip_path}"
  unzip -oq "${zip_path}" -d "${TARGET_DIR}"
}

count_wavs() {
  find "${TARGET_DIR}/mic_dev" -name "*.wav" 2>/dev/null | wc -l | tr -d ' '
}

echo "[LOCATA] Target directory: ${TARGET_DIR}"

download_file \
  "https://zenodo.org/records/3630471/files/dev.zip?download=1" \
  "${ARCHIVE_DIR}/dev.zip"
download_file \
  "https://zenodo.org/records/3630471/files/eval.zip?download=1" \
  "${ARCHIVE_DIR}/eval.zip"

extract_zip "${ARCHIVE_DIR}/dev.zip"
extract_zip "${ARCHIVE_DIR}/eval.zip"

num_wavs="$(count_wavs)"
echo "[LOCATA] Download complete. Found ${num_wavs} WAV files in ${TARGET_DIR}/mic_dev."
