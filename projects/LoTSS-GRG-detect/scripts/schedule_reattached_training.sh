#!/bin/bash

# Wait 4 hours before doing sbatch again to ensure the first training job has enough time to finish

WAIT_TIME=14400 # 4 hours in seconds

echo "Waiting for $WAIT_TIME seconds (4 hours) before scheduling the next training job..."
sleep $WAIT_TIME

echo "Scheduling the next training job..."
sbatch /home/s4861264/detectron2/projects/LoTSS-GRG-detect/scripts/train_run_continue.slurm
echo "Next training job scheduled!"
