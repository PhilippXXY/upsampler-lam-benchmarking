#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}"
TARGET_DIR_SET=0
REPO_URL="https://huggingface.co/datasets/PhilippXXY/AudibleLight_Eigenmike32-5_DCASE-STARSS23_Dataset"

usage() {
  echo "Usage: bash download_data.sh [--target-dir DIR] [DIR]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      if [[ $# -lt 2 ]]; then
        echo "[AudibleLight] Missing value for --target-dir"
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
      echo "[AudibleLight] Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [[ "${TARGET_DIR_SET}" == "1" ]]; then
        echo "[AudibleLight] Unexpected extra argument: $1"
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

count_wavs() {
  local split="$1"
  find "${TARGET_DIR}/em32_dev/${split}" -name "*.wav" 2>/dev/null | wc -l | tr -d ' '
}

count_csvs() {
  local split="$1"
  find "${TARGET_DIR}/metadata_dev/${split}" -name "*.csv" 2>/dev/null | wc -l | tr -d ' '
}

echo "[AudibleLight] Target directory: ${TARGET_DIR}"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "[AudibleLight] Existing git repository detected."
else
  echo "[AudibleLight] No .git directory found. Bootstrapping repository metadata from Hugging Face."
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  git clone "${REPO_URL}" "${tmp_dir}/repo"
  cp -R "${tmp_dir}/repo/.git" "${TARGET_DIR}/.git"
  cp -n "${tmp_dir}/repo/.gitattributes" "${TARGET_DIR}/.gitattributes" || true
  cp -n "${tmp_dir}/repo/.gitignore" "${TARGET_DIR}/.gitignore" || true
fi

if ! git -C "${TARGET_DIR}" lfs version >/dev/null 2>&1; then
  echo "[AudibleLight] git-lfs is required but not available."
  echo "Install git-lfs and rerun this script."
  exit 1
fi

echo "[AudibleLight] Pulling LFS files (this may take time and disk space)..."
git -C "${TARGET_DIR}" lfs install --local
git -C "${TARGET_DIR}" lfs pull

train_wavs="$(count_wavs dev-train)"
test_wavs="$(count_wavs dev-test)"
train_csvs="$(count_csvs dev-train)"
test_csvs="$(count_csvs dev-test)"

echo "[AudibleLight] Download complete."
echo "[AudibleLight]   em32_dev/dev-train:    ${train_wavs} WAV file(s)"
echo "[AudibleLight]   em32_dev/dev-test:     ${test_wavs} WAV file(s)"
echo "[AudibleLight]   metadata_dev/dev-train: ${train_csvs} CSV file(s) (ground truth)"
echo "[AudibleLight]   metadata_dev/dev-test:  ${test_csvs} CSV file(s) (ground truth)"

if [[ "${train_wavs}" -eq 0 ]]; then
  echo "[AudibleLight] WARNING: No WAV files found in dev-train. LFS pull may have failed."
fi
if [[ "${test_wavs}" -eq 0 ]]; then
  echo "[AudibleLight] WARNING: No WAV files found in dev-test. LFS pull may have failed."
fi
if [[ "${train_csvs}" -eq 0 ]]; then
  echo "[AudibleLight] WARNING: No CSV files found in metadata_dev/dev-train. Ground truth may be missing."
fi
if [[ "${test_csvs}" -eq 0 ]]; then
  echo "[AudibleLight] WARNING: No CSV files found in metadata_dev/dev-test. Ground truth may be missing."
fi

