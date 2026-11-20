#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Edit these values before evaluation, or set them in the environment.
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-cph}"  # cph | cps2d | pcb
SEED="${SEED:-42}"  # 42 | 43 | 44
DATASET_ROOT="${DATASET_ROOT:-/path/to/dataset}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dir/eval_${DATASET}_seed${SEED}}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "${DATASET}" in
    cph)
        CONFIG_FILE="config/DQ_5scale_cph_fixed900_pmap_ramp8.py"
        DATASET_FILE="coco"
        ;;
    cps2d)
        CONFIG_FILE="config/DQ_5scale_cps2d_fixed900_pmap_ramp8.py"
        DATASET_FILE="cps2d"
        ;;
    pcb)
        CONFIG_FILE="config/DQ_5scale_pcb_fixed900_pmap_ramp8.py"
        DATASET_FILE="coco"
        ;;
    *)
        echo "Unsupported dataset: ${DATASET}" >&2
        exit 1
        ;;
esac

CHECKPOINT_NAME="ssp_detr_${DATASET}_seed${SEED}.pth"
case "${CHECKPOINT_NAME}" in
    ssp_detr_cph_seed42.pth) CHECKPOINT_SHA256="118490f50ec068f1111aa009db80adb9680d3b68b111301722d6f71d7b6e9b13" ;;
    ssp_detr_cph_seed43.pth) CHECKPOINT_SHA256="3bb9c3ae177662707c9a62101b720fa9819f1e7b444d305fea850d2a0492d7e8" ;;
    ssp_detr_cph_seed44.pth) CHECKPOINT_SHA256="3e51dfd866e54cae4659a906463bc03048049bcd78b109c326444c89f38a40bd" ;;
    ssp_detr_cps2d_seed42.pth) CHECKPOINT_SHA256="93add214bf0542204408a532e57ef40a371fcfd007da265cfebd76c86e828735" ;;
    ssp_detr_cps2d_seed43.pth) CHECKPOINT_SHA256="0d560ecee94fc784f76fd7c905869cf9fa24f21cf4efafb466e9b607f6319c36" ;;
    ssp_detr_cps2d_seed44.pth) CHECKPOINT_SHA256="156d25de016404c7317c85f43a8be8c99c03e56fabc2f8142df21b3ba800abdf" ;;
    ssp_detr_pcb_seed42.pth) CHECKPOINT_SHA256="f48bdfd5aea9e9c8203d3132877cfa02e28dc382647a03d143885f9b0502507a" ;;
    ssp_detr_pcb_seed43.pth) CHECKPOINT_SHA256="1d822f2c31902fdbe762dbd25c574e06cb914798199632587f7db480675d08b4" ;;
    ssp_detr_pcb_seed44.pth) CHECKPOINT_SHA256="02a7db98445c230bf58d0cbf3b8178573157e2a7afc178dd05666893d910a2ab" ;;
    *)
        echo "Unsupported seed: ${SEED}" >&2
        exit 1
        ;;
esac

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset not found: ${DATASET_ROOT}" >&2
    exit 1
fi

CHECKPOINT="${WEIGHTS_DIR}/${CHECKPOINT_NAME}"
if [[ ! -f "${CHECKPOINT}" ]]; then
    mkdir -p "${WEIGHTS_DIR}"
    CHECKPOINT_URL="https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/${CHECKPOINT_NAME}"
    echo "Downloading ${CHECKPOINT_NAME}"
    curl --fail --location --retry 3 "${CHECKPOINT_URL}" --output "${CHECKPOINT}.part"
    mv "${CHECKPOINT}.part" "${CHECKPOINT}"
fi
printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${CHECKPOINT}" | sha256sum --check
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_BIN}" main_aitod.py \
    -c "${CONFIG_FILE}" \
    --dataset_file "${DATASET_FILE}" \
    --coco_path "${DATASET_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${NUM_WORKERS}" \
    --resume "${CHECKPOINT}" \
    --eval
