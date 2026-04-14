#!/bin/bash

QUEUE="gpu-short"
NATSKS=1
CPUS_PER_TASK=10
MEMORY="32G"
GPU="l4:1"

# Ask for time limit
read -p "Enter time limit for the job (e.g., 01:00:00 for 1 hour): " TIME_LIMIT

salloc --job-name=testing_nets --time=$TIME_LIMIT --partition=$QUEUE --ntasks=$NATSKS --cpus-per-task=$CPUS_PER_TASK --mem=$MEMORY --gres=gpu:$GPU
