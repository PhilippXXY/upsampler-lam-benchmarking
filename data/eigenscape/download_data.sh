#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}"
TARGET_DIR_SET=0

FORCE_PARK_REFRESH=0
KEEP_ARCHIVES=0

usage() {
  echo "Usage: bash download_data.sh [--target-dir DIR] [DIR] [--force-park-refresh] [--keep-archives]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      if [[ $# -lt 2 ]]; then
        echo "[EigenScape] Missing value for --target-dir"
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
    --force-park-refresh)
      FORCE_PARK_REFRESH=1
      shift
      ;;
    --keep-archives)
      KEEP_ARCHIVES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[EigenScape] Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [[ "${TARGET_DIR_SET}" == "1" ]]; then
        echo "[EigenScape] Unexpected extra argument: $1"
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
RAW_DIR="${TARGET_DIR}/raw"
mkdir -p "${ARCHIVE_DIR}" "${RAW_DIR}"

pick_extractor() {
  if command -v 7zz >/dev/null 2>&1; then
    echo "7zz"
    return
  fi
  if command -v 7z >/dev/null 2>&1; then
    echo "7z"
    return
  fi
  if command -v unzip >/dev/null 2>&1; then
    echo "unzip"
    return
  fi
  echo ""
}

EXTRACTOR="$(pick_extractor)"
if [[ -z "${EXTRACTOR}" ]]; then
  echo "[EigenScape] No extractor found. Install one of: 7zz, 7z, unzip."
  exit 1
fi

ensure_free_space() {
  local min_gb="${EIGENSCAPE_MIN_FREE_GB:-5}"
  local avail_kb
  avail_kb="$(df -Pk "${TARGET_DIR}" | awk 'NR==2 {print $4}')"
  local min_kb=$((min_gb * 1024 * 1024))
  if (( avail_kb < min_kb )); then
    local avail_gb
    avail_gb="$(awk -v kb="${avail_kb}" 'BEGIN {printf \"%.2f\", kb / 1024 / 1024}')"
    echo "[EigenScape] Not enough free space: ${avail_gb} GiB available."
    echo "[EigenScape] Need at least ${min_gb} GiB free to continue downloads safely."
    echo "[EigenScape] Hint: remove archive copies with: rm -f ${ARCHIVE_DIR}/*.zip"
    exit 1
  fi
}

get_remote_size_bytes() {
  local url="$1"
  curl -fLsI "${url}" | awk '
    BEGIN {IGNORECASE=1}
    /^content-length:/ {cl=$2}
    END {
      gsub("\r", "", cl)
      if (cl ~ /^[0-9]+$/) {
        print cl
      }
    }
  '
}

scene_raw_complete() {
  local scene="$1"
  for i in $(seq 1 8); do
    local idx
    idx="$(printf "%02d" "${i}")"
    local wav="${RAW_DIR}/${scene}-${idx}-Raw.wav"
    if [[ ! -s "${wav}" ]]; then
      return 1
    fi
  done
  return 0
}

download_file() {
  local url="$1"
  local out="$2"
  ensure_free_space

  local local_size=0
  if [[ -f "${out}" ]]; then
    local_size="$(wc -c < "${out}" | tr -d '[:space:]')"
  fi

  local remote_size=""
  remote_size="$(get_remote_size_bytes "${url}" || true)"

  if [[ -n "${remote_size}" && "${local_size}" -gt 0 ]]; then
    if (( local_size == remote_size )); then
      echo "[EigenScape] Skip existing (complete): ${out}"
      return
    fi
    if (( local_size > remote_size )); then
      echo "[EigenScape] Local file larger than remote, restarting: ${out}"
      rm -f "${out}"
      local_size=0
    fi
  fi

  if (( local_size > 0 )); then
    echo "[EigenScape] Resuming download: ${url} (from ${local_size} bytes)"
  else
    echo "[EigenScape] Downloading: ${url}"
  fi
  curl -fL --retry 5 --retry-delay 2 --continue-at - "${url}" -o "${out}"
}

download_with_fallback() {
  local primary_url="$1"
  local fallback_url="$2"
  local out="$3"
  local force_refresh="${4:-0}"
  if [[ "${force_refresh}" == "1" ]]; then
    echo "[EigenScape] Force refresh enabled for: ${out}"
    rm -f "${out}"
  fi

  echo "[EigenScape] Downloading (primary): ${primary_url}"
  if download_file "${primary_url}" "${out}"; then
    return
  fi

  echo "[EigenScape] Primary failed. Downloading (fallback): ${fallback_url}"
  download_file "${fallback_url}" "${out}"
}

extract_zip() {
  local zip_path="$1"
  echo "[EigenScape] Extracting: ${zip_path}"

  if [[ "${EXTRACTOR}" == "7zz" || "${EXTRACTOR}" == "7z" ]]; then
    # Keep progress visible: percentage + processed file entries.
    "${EXTRACTOR}" x -y -bb1 -bsp1 -bso1 -bse2 -o"${RAW_DIR}" "${zip_path}"
    return
  fi

  unzip -oq "${zip_path}" -d "${RAW_DIR}"

  if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
    rm -f "${zip_path}"
    echo "[EigenScape] Removed archive to save disk: ${zip_path}"
  fi
}

echo "[EigenScape] Target directory: ${TARGET_DIR}"
echo "[EigenScape] Using extractor: ${EXTRACTOR}"
echo "[EigenScape] Download mode: eigenscape_raw (32-channel Eigenmike WAV)."
echo "[EigenScape] Source layout follows soundata eigenscape_raw remotes."
echo "[EigenScape] Reference: https://soundata.readthedocs.io/en/latest/_modules/soundata/datasets/eigenscape_raw.html"
if [[ "${FORCE_PARK_REFRESH}" == "1" ]]; then
  echo "[EigenScape] Force Park refresh: enabled"
fi
if [[ "${KEEP_ARCHIVES}" == "1" ]]; then
  echo "[EigenScape] Keep archives: enabled"
else
  echo "[EigenScape] Keep archives: disabled (default)"
fi

AUDIO_URL_YORK="https://webfiles.york.ac.uk/INFODATA/eaeaac50-483e-408f-a391-01b02d4ff9c4/"
# Soundata eigenscape_raw fallback remote for Park.
PARK_ZENODO_URL="https://zenodo.org/record/1284156/files/Park.zip?download=1"
METADATA_URL="https://zenodo.org/record/1284156/files/Metadata-EigenScape.csv?download=1"

# Archives available on the York host.
for scene in Beach BusyStreet PedestrianZone; do
  if scene_raw_complete "${scene}"; then
    echo "[EigenScape] Skip ${scene}: raw files already present."
    continue
  fi
  archive="${ARCHIVE_DIR}/${scene}.zip"
  download_file "${AUDIO_URL_YORK}${scene}.zip" "${archive}"
  extract_zip "${archive}"
done

# Park has a fallback source in soundata remotes.
park_archive="${ARCHIVE_DIR}/Park.zip"
if [[ "${FORCE_PARK_REFRESH}" == "1" ]]; then
  rm -f "${RAW_DIR}/Park-"*-Raw.wav
fi
if [[ "${FORCE_PARK_REFRESH}" != "1" ]] && scene_raw_complete "Park"; then
  echo "[EigenScape] Skip Park: raw files already present."
else
  download_with_fallback "${AUDIO_URL_YORK}Park.zip" "${PARK_ZENODO_URL}" "${park_archive}" "${FORCE_PARK_REFRESH}"
  extract_zip "${park_archive}"
fi

# Files provided directly as WAVs on the York host.
for scene in QuietStreet ShoppingCentre TrainStation Woodland; do
  for i in $(seq 1 8); do
    idx="$(printf "%02d" "${i}")"
    file="${scene}-${idx}-Raw.wav"
    download_file "${AUDIO_URL_YORK}${file}" "${RAW_DIR}/${file}"
  done
done

download_file "${METADATA_URL}" "${TARGET_DIR}/Metadata-EigenScape.csv"

num_wavs="$(find "${RAW_DIR}" -name "*.wav" | wc -l | tr -d ' ')"
echo "[EigenScape] Download complete. Found ${num_wavs} WAV files in ${RAW_DIR}."
