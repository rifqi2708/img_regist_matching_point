#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYTHON_PACKAGES=(
  "numpy<1.24"
  nibabel
  pillow
  matplotlib
  itk
  itk-elastix
)

HARD_CODED_PATH_FILES=(
  run_dvf_point_matching.py
  registration_cycle_error.py
  run_single_pair_elastix.py
  verify_dvf_units_and_coordinates.py
  visualize_dvf_nifti.py
)

log() {
  printf '[setup] %s\n' "$1"
}

fail() {
  printf '[setup] ERROR: %s\n' "$1" >&2
  exit 1
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "Could not find '$PYTHON_BIN'. Install Python 3 first, or set PYTHON_BIN=/path/to/python3."
fi

cd "$ROOT_DIR"

log "Upgrading pip tooling"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

log "Installing Python dependencies"
"$PYTHON_BIN" -m pip install "${PYTHON_PACKAGES[@]}"

log "Verifying imports"
"$PYTHON_BIN" - <<'PY'
import itk
import matplotlib
import nibabel
import numpy
import PIL

print("Imports OK")
print(f"Python: {__import__('sys').version.split()[0]}")
print(f"ITK: {itk.Version.GetITKVersion()}")
print(f"NumPy: {numpy.__version__}")
print(f"Nibabel: {nibabel.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Pillow: {PIL.__version__}")
PY

log "Environment is ready"
printf '\n'
printf 'Note:\n'
printf '  Several scripts still contain machine-specific absolute paths.\n'
printf '  Update these before running on a remote environment:\n'
for file in "${HARD_CODED_PATH_FILES[@]}"; do
  printf '  - %s\n' "$file"
done
