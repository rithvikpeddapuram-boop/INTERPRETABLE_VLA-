#!/bin/bash
# ==============================================================
# LIBERO install -- run AFTER install_vla_env.sh has succeeded.
# Installs LIBERO into the SAME vla-interp environment (does not
# create a new one), skipping LIBERO's old torch/python pins since
# SmolVLA already requires newer versions.
#
# Usage:
#   conda activate vla-interp
#   bash install_libero.sh
# ==============================================================

set -e

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "vla-interp" ]; then
    echo "ERROR: vla-interp environment is not active."
    echo "Run: conda activate vla-interp"
    echo "Then re-run this script."
    exit 1
fi

echo "=============================================="
echo "STEP 1: Clone LIBERO"
echo "=============================================="
cd ~
if [ ! -d "$HOME/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd "$HOME/LIBERO"

echo "=============================================="
echo "STEP 2a: Ensure a modern, Python-3.12-compatible numpy is installed FIRST"
echo "(LIBERO's requirements.txt pins an old numpy that fails to build on 3.12)"
echo "=============================================="
pip install --upgrade "numpy>=1.26"

echo "=============================================="
echo "STEP 2b: Install LIBERO's remaining requirements, EXCLUDING"
echo "torch/torchvision/torchaudio (SmolVLA already installed newer, compatible"
echo "versions of these) and EXCLUDING numpy (already handled above)"
echo "=============================================="
grep -v -i -E '^torch|^torchvision|^torchaudio|^numpy' requirements.txt > requirements_filtered.txt
pip install -r requirements_filtered.txt

echo "=============================================="
echo "STEP 3: Install the libero package itself"
echo "=============================================="
pip install -e .

echo "=============================================="
echo "STEP 4: Download the libero_spatial task suite only (fastest to start)"
echo "=============================================="
python3 benchmark_scripts/download_libero_datasets.py --datasets libero_spatial

echo "=============================================="
echo "DONE."
echo "Next: run the test script (02_test_libero.py) to confirm the"
echo "simulation environment actually loads and steps correctly."
echo "=============================================="
