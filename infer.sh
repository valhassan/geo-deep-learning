#!/bin/bash

set -e  # Exit on any error

PROJECT_ROOT="/home/valhassa/Projects/geo-deep-learning"
INFERENCE_SCRIPT="${PROJECT_ROOT}/geo_deep_learning/infer.py"

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="${PROJECT_ROOT}"

print_header() {
    echo "========================================="
    echo "$1"
    echo "========================================="
}

show_gpu_status() {
    print_header "GPU Status Before Inference"
    nvidia-smi
    echo ""
}

kill_existing_processes() {
    print_header "Cleaning Up Existing Processes"

    # Kill any existing inference processes
    pkill -f "infer.py" || true

    # Wait a moment for cleanup
    sleep 2

    echo "✅ Process cleanup completed"
}

# Run the functions
kill_existing_processes
show_gpu_status

python $INFERENCE_SCRIPT \
    --checkpoint /export/sata01/wspace/test_dir/multi/all_rgb_data/RGB_4class_Segformer_b5_VA_20230915.pth.tar \
    --input /home/valhassa/Projects/geo-deep-learning/data/JAM-HM-SAN-CW-20251111-BU-21-B2_True_Ortho.tif \
    --output /home/valhassa/Projects/geo-deep-learning/data/out/pred.tif \
    --mean 0.405 0.432 0.397 \
    --std 0.164 0.173 0.153
