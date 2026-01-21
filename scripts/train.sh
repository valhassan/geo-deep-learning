#!/bin/bash

set -e  # Exit on any error

# Project settings
PROJECT_ROOT="/home/valhassa/training/geo-deep-learning"
TRAIN_SCRIPT="${PROJECT_ROOT}/geo_deep_learning/train.py"

CONFIG_PATH="${PROJECT_ROOT}/experiments/config/wds_segformer.yaml"

# Training parameters
NUM_GPUS=2
NUM_NODES=1

# Environment settings
export WDS_VERBOSE_CACHE=1
export GOPEN_VERBOSE=0
export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="0,1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

print_info() {
    echo "========================================="
    echo "$1"
    echo "========================================="
}

show_gpu_status() {
    print_info "GPU Status Before Training"
    nvidia-smi
    echo ""
}


kill_existing_processes() {
    print_info "Cleaning Up Existing Processes"
    
    # Kill any existing training processes
    pkill -f "train.py" || true
    pkill -f "torchrun" || true
    pkill -f "test.py" || true
    
    # Wait a moment for cleanup
    sleep 2
    
    echo "Process cleanup completed."
}

run_training() {
    print_info "Launch training on $NUM_GPUS GPUs..."
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    # Run training
    python "$TRAIN_SCRIPT" fit \
    --config "$CONFIG_PATH" \
    --trainer.num_nodes=$NUM_NODES
}

run_testing() {
    print_info "Launch testing on 1 GPU..."
    
    cd "$PROJECT_ROOT"

    python "$TRAIN_SCRIPT" test \
        --config "$CONFIG_PATH" \
        --trainer.devices=1 \
        --data.init_args.batch_size=32
}

cleanup_on_exit() {
    echo ""
    print_info "Training Completed/Interrupted"

    # Kill any existing testing processes
    kill_existing_processes
    
    # Show final GPU status
    echo "Final GPU status:"
    nvidia-smi
    
    echo ""
    echo "Training session ended."
}

# Set trap for cleanup
trap cleanup_on_exit EXIT

# Execute training

train() {
    print_info "Geo Deep Learning - Training Session"
    
    # Run all checks and setup
    show_gpu_status
    kill_existing_processes
    
    # Start training
    run_training
}

test() {
    print_info "Geo Deep Learning - Testing Session"
    
    # Run all checks and setup
    show_gpu_status
    kill_existing_processes
    
    # Start testing
    run_testing
}

case "$1" in
    "train")
        train
        ;;
    "test")
        test
        ;;
    *)
        echo "Usage: $0 [train|test]"
        exit 1
        ;;
esac