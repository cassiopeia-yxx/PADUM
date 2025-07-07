#!/bin/bash
# Training script for PADM (Pixel Adaptive Deep Unfolding Network)

# Set up Python environment
export PYTHONPATH=.:$PYTHONPATH

# Training parameters
CONFIG_FILE="Options/Deraining.yml"
EXPERIMENT_NAME="PADUM"
GPU_IDS="0,1,2,3"  # Comma-separated list of GPU IDs to use
BATCH_SIZE=16       # Batch size per GPU
NUM_WORKERS=8       # Number of data loading workers
LOG_FREQ=100        # How often (in iterations) to log training progress
SAVE_FREQ=50        # How often (in epochs) to save model checkpoints

# Create experiment directory
EXP_DIR="experiments/${EXPERIMENT_NAME}"
mkdir -p ${EXP_DIR}

# Start training
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=4321 basicsr/train.py \
    --opt ${CONFIG_FILE} \
    --launcher pytorch \
    --auto_resume \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --log_freq ${LOG_FREQ} \
    --save_freq ${SAVE_FREQ} \
    --gpu_ids ${GPU_IDS} \
    --exp_dir ${EXP_DIR}

# To monitor training with TensorBoard:
# tensorboard --logdir=experiments/tb_logger