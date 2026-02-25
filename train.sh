#!/bin/bash

# ==============================================================================
# Multi-GPU Distributed Training Script for Geo Deep Learning
#
# QUICK USAGE:
#   ./train_ddp.sh              # Run distributed training (default)
#   ./train_ddp.sh train        # Same as above
#   ./train_ddp.sh test         # Run test on single GPU
#   ./train_ddp.sh validate     # Check if distributed setup works
#   ./train_ddp.sh clean        # Kill any stuck training processes
#   ./train_ddp.sh status       # Show GPU status
#   ./train_ddp.sh help         # Show detailed help
#
# ==============================================================================

set -e  # Exit on any error

# ==============================================================================
# Configuration
# ==============================================================================

# Project paths
PROJECT_ROOT="/home/valhassa/Projects/geo-deep-learning"
# CONFIG_FILE="${PROJECT_ROOT}/configs/exp/wds_dofa.yaml"
# CONFIG_FILE="${PROJECT_ROOT}/configs/exp/wds_segformer.yaml"
CONFIG_FILE="${PROJECT_ROOT}/experiments/config_files/wds_dinov3.yaml"
# CONFIG_FILE="${PROJECT_ROOT}/configs/segformer_config_RGB.yaml"
# CONFIG_FILE="${PROJECT_ROOT}/configs/unetplus_config_RGB.yaml"
# CONFIG_FILE="${PROJECT_ROOT}/configs/exp/armando_config_RGB.yaml"
# CONFIG_FILE="${PROJECT_ROOT}/configs/exp/ssl_mit.yaml"
TRAIN_SCRIPT="${PROJECT_ROOT}/geo_deep_learning/train.py"

# Distributed training settings
NUM_GPUS=2
NUM_NODES=1
MASTER_PORT=12355

# Environment settings
export WDS_VERBOSE_CACHE=1
export GOPEN_VERBOSE=0
export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="0,1"  # Explicitly set which GPUs to use
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Optional: NCCL debugging (uncomment if you have issues)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
# ==============================================================================
# Functions
# ==============================================================================

print_header() {
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check if project directory exists
    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "❌ Project directory not found: $PROJECT_ROOT"
        exit 1
    fi

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Check if training script exists
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "❌ Training script not found: $TRAIN_SCRIPT"
        exit 1
    fi

    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. Are NVIDIA drivers installed?"
        exit 1
    fi

    # Check number of available GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        echo "❌ Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
        exit 1
    fi

    echo "✅ All prerequisites check passed"
    echo "   - Project root: $PROJECT_ROOT"
    echo "   - Config file: $CONFIG_FILE"
    echo "   - Training script: $TRAIN_SCRIPT"
    echo "   - Available GPUs: $GPU_COUNT"
    echo "   - Using GPUs: $NUM_GPUS"
}

show_gpu_status() {
    print_header "GPU Status Before Training"
    nvidia-smi
    echo ""
}

kill_existing_processes() {
    print_header "Cleaning Up Existing Processes"

    # Kill any existing training processes
    pkill -f "train.py" || true
    pkill -f "torchrun" || true

    # Wait a moment for cleanup
    sleep 2

    echo "✅ Process cleanup completed"
}

start_monitoring() {
    print_header "Starting GPU Monitoring"

    # Start GPU monitoring in background
    echo "Starting nvidia-smi monitoring..."
    echo "You can check GPU utilization with: watch -n 1 nvidia-smi"
    echo ""
}

run_training() {
    print_header "Starting Distributed Training"

    echo "Configuration:"
    echo "  - GPUs per node: $NUM_GPUS"
    echo "  - Number of nodes: $NUM_NODES"
    echo "  - Master port: $MASTER_PORT"
    echo "  - Config file: $CONFIG_FILE"
    echo ""

    # Change to project directory
    cd "$PROJECT_ROOT"

    # Run distributed training
    echo "🚀 Launching distributed training..."
    echo ""

    # torchrun \
    #     --nproc_per_node=$NUM_GPUS \
    #     --nnodes=$NUM_NODES \
    #     --master_port=$MASTER_PORT \
    #     "$TRAIN_SCRIPT" fit \
    #     --config "$CONFIG_FILE"
    python "$TRAIN_SCRIPT" fit \
    --config "$CONFIG_FILE" \
    --trainer.devices=$NUM_GPUS \
    --trainer.num_nodes=$NUM_NODES
}

# ==============================================================================
# Error handling
# ==============================================================================

cleanup_on_exit() {
    echo ""
    print_header "Training Completed/Interrupted"

    # Show final GPU status
    echo "Final GPU status:"
    nvidia-smi

    echo ""
    echo "Training session ended."
}

# Set trap for cleanup
trap cleanup_on_exit EXIT

# ==============================================================================
# Main execution
# ==============================================================================

main() {
    print_header "Geo Deep Learning - Distributed Training"

    # Run all checks and setup
    check_prerequisites
    show_gpu_status
    kill_existing_processes
    start_monitoring

    # Start training
    run_training
}

# ==============================================================================
# Additional utility functions (can be called separately)
# ==============================================================================

# Function to run test on single GPU
test_single_gpu() {
    print_header "Running test on single GPU"

    cd "$PROJECT_ROOT"

    # Temporarily modify config or create single GPU version
    echo "🧪 Running test on single GPU..."

    python "$TRAIN_SCRIPT" test \
        --config "$CONFIG_FILE" \
        --trainer.devices=1 \
        --data.init_args.batch_size=16
}

# Function to validate distributed setup
validate_distributed() {
    print_header "Validating Distributed Setup"

    echo "🔍 Testing if torchrun can launch your training script..."

    cd "$PROJECT_ROOT"

    # Test with a very short run
    timeout 30s torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --master_port=$MASTER_PORT \
        "$TRAIN_SCRIPT" fit \
        --config "$CONFIG_FILE" \
        --trainer.max_epochs=1 \
        --data.init_args.epoch_size=2 \
        --trainer.limit_train_batches=1 || echo "Validation completed (timeout expected)"
}

# ==============================================================================
# Command line interface
# ==============================================================================

case "${1:-main}" in
    "main"|"train")
        main
        ;;
    "test")
        test_single_gpu
        ;;
    "validate")
        validate_distributed
        ;;
    "clean")
        kill_existing_processes
        ;;
    "status")
        show_gpu_status
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  main, train     - Run full distributed training (default)"
        echo "  test            - Test with single GPU"
        echo "  validate        - Validate distributed setup"
        echo "  clean           - Kill existing training processes"
        echo "  status          - Show current GPU status"
        echo "  help            - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run distributed training"
        echo "  $0 test               # Test single GPU"
        echo "  $0 validate           # Check distributed setup"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
