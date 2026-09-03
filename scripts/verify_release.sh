#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

sha256sum --check artifacts/SHA256SUMS
"${PYTHON_BIN}" analysis/validate_artifacts.py
"${PYTHON_BIN}" -m unittest discover -s tests -v

echo "[done] release artifacts and tests verified"
