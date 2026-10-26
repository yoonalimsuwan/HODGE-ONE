#!/bin/bash
#=============================================================================
# RUN DDP — Multi-GPU Training Script
#=============================================================================
# Usage:
#   ./run_ddp.sh                    # Auto-detect GPUs
#   ./run_ddp.sh 4                  # Use 4 GPUs
#   ./run_ddp.sh 4 cy               # 4 GPUs, Calabi-Yau model
#   ./run_ddp.sh 4 protein ./data/pdb/pdb_data.json
#=============================================================================

set -e

# Configuration
NGPUS=${1:-auto}
MODEL=${2:-cy}
DATA=${3:-./data/cicy/cicy_data.json}
EPOCHS=${4:-100}
BATCH_SIZE=${5:-16}
OUTPUT_DIR=${6:-./checkpoints}

# Auto-detect GPUs
if [ "$NGPUS" == "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NGPUS=1
    fi
fi

echo "=============================================="
echo " SESI DDP Training"
echo "=============================================="
echo " Model:      $MODEL"
echo " Data:       $DATA"
echo " GPUs:       $NGPUS"
echo " Epochs:     $EPOCHS"
echo " Batch size: $BATCH_SIZE (per GPU)"
echo " Output:     $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
if [ $NGPUS -gt 1 ]; then
    echo "Starting multi-GPU training with torchrun..."
    torchrun \
        --nproc_per_node=$NGPUS \
        --master_port=29500 \
        train_ddp.py \
        --model $MODEL \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR
else
    echo "Starting single-GPU training..."
    python train_ddp.py \
        --model $MODEL \
        --data $DATA \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR
fi

echo "=============================================="
echo " Training complete!"
echo "=============================================="
