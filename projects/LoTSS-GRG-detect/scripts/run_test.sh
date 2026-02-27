#!/bin/bash
# Testing script launcher for GRG detection evaluation

set -e

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/scripts"

# Defaults
CONFIG="${PROJECT_ROOT}/configs/mask_rcnn_R_50_FPN_grg.yaml"
DATASET_CONFIG="${PROJECT_ROOT}/config/dataset_test.yaml"
WEIGHTS=""
OUTPUT_DIR=""
SCORE_THRESHOLD=""

# Parse arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset-config)
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --score-threshold)
            SCORE_THRESHOLD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_test.sh [options] [extra test.py args]"
            echo ""
            echo "Options:"
            echo "  --config PATH            Model config YAML"
            echo "  --dataset-config PATH    Dataset config YAML"
            echo "  --weights PATH           Checkpoint (.pth)"
            echo "  --output-dir PATH        Evaluation output directory"
            echo "  --score-threshold FLOAT  ROI score threshold"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "======================================"
echo "GRG Detection Evaluation"
echo "======================================"
echo "Config: $CONFIG"
echo "Dataset config: $DATASET_CONFIG"
echo "Weights: ${WEIGHTS:-From MODEL.WEIGHTS in config}"
echo "Output dir: ${OUTPUT_DIR:-From cfg.OUTPUT_DIR}"
echo "Score threshold: ${SCORE_THRESHOLD:-From cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}"
echo "======================================"

CMD=(python test.py --config-file "$CONFIG" --dataset-config "$DATASET_CONFIG")

if [[ -n "$WEIGHTS" ]]; then
    CMD+=(--weights "$WEIGHTS")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ -n "$SCORE_THRESHOLD" ]]; then
    CMD+=(--score-threshold "$SCORE_THRESHOLD")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
