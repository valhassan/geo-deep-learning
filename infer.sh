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
    --checkpoint /export/sata01/wspace/test_dir/multi/all_rgb_data/geoaware_epoch_45_val_loss_0.527.ckpt \
    --input /home/valhassa/Projects/geo-deep-learning/experiments/data/NS2-058651012010_01_P001-WV02-RGB.tif \
    --output /home/valhassa/Projects/geo-deep-learning/experiments/data/out/ns2_wv2_static_segformer_rgb_geoaware.tif \
    --mean 0.1014 0.1360 0.1296 \
    --std 0.1102 0.1230 0.1107
    # --mean 0.405 0.432 0.397 \
    # --std 0.164 0.173 0.153
    # --mean 0.1014 0.1360 0.1296 0.2604 \
    # --std 0.1102 0.1230 0.1107 0.2099
