#!/bin/bash

set -e  # Exit on any error

PROJECT_ROOT="/home/valhassa/dev/geo-deep-learning"

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="${PROJECT_ROOT}"

INFERENCE_SCRIPT="${PROJECT_ROOT}/geo_deep_learning/infer.py"
COMPARE_SCRIPT="${PROJECT_ROOT}/geo_deep_learning/compare_models.py"




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

# python -m geo_deep_learning.tools.patch_checkpoint \
#   /export/sata01/wspace/test_dir/multi/all_rgb_data/dofav2_geoaware_epoch_47_val_loss_0.599.ckpt \
#   -o /export/sata01/wspace/test_dir/multi/all_rgb_data/dofav2_geoaware_patched.ckpt \
#   --no-weights-only

# python -m geo_deep_learning.tools.export_model \
#   --model dofa \
#   --checkpoint /export/sata01/wspace/test_dir/multi/all_rgb_data/dofav2_geoaware_patched.ckpt \
#   --output dofav2_geoaware_patched.pt2


python $INFERENCE_SCRIPT \
    --checkpoint /export/sata01/wspace/test_dir/multi/demo/pt2/dofav2_47_0.599.pt2 \
    --input /export/sata01/wspace/test_dir/multi/demo/image/ns2_small_rgbn.tif \
    --output /export/sata01/wspace/test_dir/multi/demo/prediction/ns2_small_rgbn_tta_10_32_32_mix_zoom_out_0.5.tif \
    --mean 0.1014 0.1360 0.1296 0.2604 \
    --std 0.1102 0.1230 0.1107 0.2099 \
    --wavelengths 0.6599999999999999 0.5449999999999999 0.48 0.8325 \
    --num-classes 5 \
    --batch-size 8 \
    --radiometric-tta \
    --geometric-tta \
    --zoom-out-tta \
    --zoom-out-scale 0.5

# python $COMPARE_SCRIPT \
#   --checkpoint /export/sata01/wspace/test_dir/multi/all_rgb_data/dofav2.ckpt \
#   --exported dofav2.pt2 \
