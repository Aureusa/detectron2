#!/bin/bash
# Training script for GRG detection

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/scripts"

# Default config
CONFIG="${PROJECT_ROOT}/configs/mask_rcnn_R_50_FPN_grg.yaml"

# Parse arguments
NUM_GPUS=1
RESUME=""
EVAL_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --eval-only)
            EVAL_ONLY="--eval-only"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "GRG Detection Training"
echo "======================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Resume: ${RESUME:-No}"
echo "Eval only: ${EVAL_ONLY:-No}"
echo "======================================"

# Run training
python train.py \
    --config-file "$CONFIG" \
    --num-gpus "$NUM_GPUS" \
    $RESUME \
    $EVAL_ONLY \
    "${@}"
