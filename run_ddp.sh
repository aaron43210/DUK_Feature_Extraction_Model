#!/bin/bash
# DUK-EM 8-GPU DDP Training Launch Script
# Scaled for DGX Systems (8x A100/H100)

# High-Performance Configuration
export OMP_NUM_THREADS=1                 # Prevent CPU contention (1 thread per DDP process)
export MKL_NUM_THREADS=1                 # Consistent with OMP
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # DDP robustness
export CUDA_LAUNCH_BLOCKING=0            # Async kernel launches for speed
export NCCL_TIMEOUT=1800000              # 30-min timeout for large model syncs
export NCCL_P2P_DISABLE=0               # Force NVLink P2P
export NCCL_IB_DISABLE=0                # Force InfiniBand on DGX
export TORCH_DISTRIBUTED_DEBUG=DETAIL    # Better crash logs
export MASTER_PORT=$((29500 + RANDOM % 100))

# Training Parameters
if [ $# -eq 0 ]; then
    TRAIN_DIR="../DATA/MAP1 ../DATA/MAP2 ../DATA/MAP3 ../DATA/MAP4 ../DATA/MAP5 ../DATA/MAP6 ../DATA/MAP7 ../DATA/MAP8 ../DATA/MAP9"
else
    TRAIN_DIR="$@"
fi
EPOCHS=150
PER_GPU_BATCH=12   # 12 per GPU prevents OOM on SegFormer-B4
WORKERS=2          # 1 worker per GPU, prefetches 1 batch ahead

echo "=========================================================="
echo "LAUNCHING 8-GPU DDP TRAINING"
echo "   Data Root:    $TRAIN_DIR"
echo "   Global Batch: $((PER_GPU_BATCH * 8))"
echo "=========================================================="

MIN_GPUS=1

while true; do
    echo "=========================================================="
    echo "SCANNING FOR AVAILABLE GPUs..."

    FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '$2 > 20000 {print $1}' | \
        paste -sd "," -)

    if [ -z "$FREE_GPUS" ]; then
        echo "No free GPUs available (all >20GB utilized). Waiting 60s..."
        sleep 60
        continue
    fi

    NUM_FREE_GPUS=$(echo $FREE_GPUS | awk -F, '{print NF}')

    if [ "$NUM_FREE_GPUS" -lt "$MIN_GPUS" ]; then
        echo "Found $NUM_FREE_GPUS GPUs, need at least $MIN_GPUS. Waiting 60s..."
        sleep 60
        continue
    fi

    export CUDA_VISIBLE_DEVICES=$FREE_GPUS
    export MASTER_PORT=$((29500 + RANDOM % 100))

    echo "Found $NUM_FREE_GPUS free GPUs: [$FREE_GPUS]"
    echo "Launching elastic DDP training — global batch: $((PER_GPU_BATCH * NUM_FREE_GPUS))"
    echo "=========================================================="

    python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_FREE_GPUS \
        --master_port=$MASTER_PORT \
        train_engine/train_segmentation.py \
        --train_dirs $TRAIN_DIR \
        --epochs $EPOCHS \
        --batch_size $PER_GPU_BATCH \
        --num_workers $WORKERS \
        --checkpoint_dir check \
        --resume \
        --multi_gpu \
        --split_mode tile \
        --name dgx_ddp_v1_final

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully."
        break
    elif [ $EXIT_CODE -eq 3 ]; then
        echo "Epoch completed (one_epoch_only). Restarting for next epoch..."
        sleep 5
        continue
    else
        echo "Training crashed (exit code: $EXIT_CODE). Waiting 30s before restart..."
        sleep 30
    fi
done
