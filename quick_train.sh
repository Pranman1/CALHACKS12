#!/bin/bash

# Quick Training Script for Skateboard Balancing
# This script sets up optimized parameters for fast training

echo "Starting quick training for skateboard balancing..."

# Set environment variables for Isaac Lab
export ISAACLAB_PATH="$(pwd)/unitree_rl_lab"
export PYTHONPATH="${ISAACLAB_PATH}/source:${PYTHONPATH}"

# Training parameters optimized for speed
TASK="Unitree-G1-29DOF-Skateboard-v0"
NUM_ENVS=8192
MAX_ITERATIONS=1000
BATCH_SIZE=16384
MINI_BATCH_SIZE=512
LEARNING_RATE=3e-4
SAVE_INTERVAL=100
LOG_INTERVAL=10

echo "Training parameters:"
echo "  Task: $TASK"
echo "  Environments: $NUM_ENVS"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"

# Run training with optimized parameters
python unitree_rl_lab/scripts/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --batch_size "$BATCH_SIZE" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --save_interval "$SAVE_INTERVAL" \
    --log_interval "$LOG_INTERVAL" \
    --headless \
    --num_envs "$NUM_ENVS"

echo "Training completed!"
