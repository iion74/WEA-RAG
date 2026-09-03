#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
mkdir -p "${DATA_DIR}"

download() {
  local url="$1"
  local output="$2"
  if [[ -s "${output}" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[download] ${url}"
  curl --fail --location --retry 3 --output "${output}.part" "${url}"
  mv "${output}.part" "${output}"
}

BASE_URL="http://curtis.ml.cmu.edu/datasets/hotpot"
download "${BASE_URL}/hotpot_train_v1.1.json" "${DATA_DIR}/hotpot_train_v1.1.json"
download "${BASE_URL}/hotpot_dev_distractor_v1.json" "${DATA_DIR}/hotpot_dev_distractor_v1.json"
download "${BASE_URL}/hotpot_dev_fullwiki_v1.json" "${DATA_DIR}/hotpot_dev_fullwiki_v1.json"
