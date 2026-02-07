# GRG Detection Training

## Quick Start

### 1. Training
```bash
cd scripts
chmod +x run_training.sh

# Train on single GPU
./run_training.sh

# Train on multiple GPUs
./run_training.sh --gpus 2

# Resume training from checkpoint
./run_training.sh --resume

# Evaluation only
./run_training.sh --eval-only
```

### 2. Or run directly with Python
```bash
cd scripts
python train.py --config-file ../configs/mask_rcnn_R_50_FPN_grg.yaml --num-gpus 1
```

## Configuration

### Model Config: `configs/mask_rcnn_R_50_FPN_grg.yaml`
- **Model**: Mask R-CNN with ResNet-50 FPN
- **Tasks**: Bounding box detection + instance segmentation
- **Precomputed proposals**: Enabled (loads from NPZ files)
- **Batch size**: 8 (adjust based on GPU memory)
- **Learning rate**: 0.001
- **Max iterations**: 25,000
- **Evaluation**: Every 1,000 iterations

### Dataset Config: `config/dataset.yaml`
- **Training data**: `/home/s4861264/project_data/full-dataset/train/`
- **Validation data**: `/home/s4861264/project_data/full-dataset/val/`
- **Proposals**: Loaded from `proposals/` subdirectory

## Key Features

1. **Custom Proposal Mapper**: Loads precomputed proposals from NPZ files
2. **Automatic Evaluation**: Validates on val set every 1,000 iterations
3. **COCO Metrics**: Reports bbox AP and segmentation AP
4. **Logging**: Comprehensive logging of training progress
5. **Checkpointing**: Saves checkpoints every 2,500 iterations

## Customization

### Adjust Training Parameters
Edit `configs/mask_rcnn_R_50_FPN_grg.yaml`:
- `SOLVER.IMS_PER_BATCH`: Batch size
- `SOLVER.BASE_LR`: Learning rate
- `SOLVER.MAX_ITER`: Training iterations
- `TEST.EVAL_PERIOD`: Evaluation frequency

### Change Dataset Paths
Edit `config/dataset.yaml`:
- `DATA_ROOT`: Root directory of your data

## Output

Training outputs saved to: `./output/grg_mask_rcnn/`
- Model checkpoints: `model_*.pth`
- Final model: `model_final.pth`
- Evaluation results: `inference/`
- TensorBoard logs: `events.out.tfevents.*`

## Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir ./output/grg_mask_rcnn
```
