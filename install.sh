#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"

"${PYTHON_BIN}" -m pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
"${PYTHON_BIN}" -m pip install cython==0.29.36
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install \
    "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"

pushd models/dqdetr/ops >/dev/null
"${PYTHON_BIN}" setup.py build install
"${PYTHON_BIN}" test.py
popd >/dev/null
