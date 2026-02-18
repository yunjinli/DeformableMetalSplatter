#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinysam

# Configuration - modify these paths as needed
PLY_PATH="/Users/cedric/Documents/example_data/sear_steak/point_cloud.ply"
MODEL_PATH="/Users/cedric/Documents/example_data/sear_steak/deform.pth"
OUTPUT_PATH="/Users/cedric/Documents/example_data/sear_steak/mask.bin"

# Build the command
CMD="python generate_deformation_mask.py $PLY_PATH $MODEL_PATH $OUTPUT_PATH --smooth --threshold 0.1 --max-t 1.0 --num-samples 10 "
echo "Running: $CMD"
eval $CMD
