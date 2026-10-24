#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Edit these values before training.
PYTHON_BIN="python"
DATASET="cph"  # cph | cps2d | pcb
DATASET_ROOT="/path/to/dataset"
PRETRAIN_MODEL="/path/to/pretrain_model.pth"
OUTPUT_DIR="work_dir/train_${DATASET}"
GPU_IDS="0,1"
NUM_WORKERS=4
SEED=42
MASTER_PORT=29500

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

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset not found: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${PRETRAIN_MODEL}" ]]; then
    echo "Pretrained model not found: ${PRETRAIN_MODEL}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"

ARGS=(
    -c "${CONFIG_FILE}"
    --dataset_file "${DATASET_FILE}"
    --coco_path "${DATASET_ROOT}"
    --pretrain_model_path "${PRETRAIN_MODEL}"
    --output_dir "${OUTPUT_DIR}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --save_log
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        main_aitod.py "${ARGS[@]}"
else
    "${PYTHON_BIN}" main_aitod.py "${ARGS[@]}"
fi
