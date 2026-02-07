#!/bin/bash
#SBATCH --job-name=train_grg_mask_rcnn
#SBATCH --mail-user="s4861264@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:1

# making sure we start with a clean module environment
module purge

TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"

echo "## Available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

source /home/s4861264/.envs/grg_project/bin/activate

echo "## Running training script"

/home/s4861264/detectron2/projects/LoTSS-GRG-detect/scripts/run_training.sh
