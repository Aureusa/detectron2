#!/bin/bash
# Training script for GRG detection

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/pipelines"

# Default config
CONFIG="${PROJECT_ROOT}/configs/model_configs/modified_rcnn.yaml"
DATASET_CONFIG="${PROJECT_ROOT}/configs/dataset_configs/dataset_b2s.yaml"

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
echo "Dataset config: $DATASET_CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Resume: ${RESUME:-No}"
echo "Eval only: ${EVAL_ONLY:-No}"
echo "======================================"

# Run training
python train.py \
    --config-file "$CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --num-gpus "$NUM_GPUS" \
    $RESUME \
    $EVAL_ONLY \
    "${@}"
