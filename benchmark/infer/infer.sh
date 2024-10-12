#!/bin/sh
set -x
set -e

input_rgb_root=/path/to/input/RGB/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] input RGB
saved_root=/path/to/saved/root_directory # The parent directory that saving [sintel, scannet, KITTI, bonn, NYUv2] prediction
gpus=0,1,2,3 # Using 4 GPU, you can adjust it according to your device


# infer sintel
python benchmark/infer/infer_batch.py \
    --meta_path ./eval/csv/meta_sintel.csv \
    --saved_root ${saved_root} \
    --saved_dataset_folder results_sintel \
    --input_rgb_root ${input_rgb_root} \
    --process_length 50 \
    --gpus ${gpus} \
    --dataset sintel \

# infer scannet
python benchmark/infer/infer_batch.py \
    --meta_path ./eval/csv/meta_scannet_test.csv \
    --saved_root ${saved_root} \
    --saved_dataset_folder results_scannet \
    --input_rgb_root ${input_rgb_root} \
    --process_length 90 \
    --gpus ${gpus} \
    --dataset scannet \

# infer kitti
python benchmark/infer/infer_batch.py \
    --meta_path ./eval/csv/meta_kitti_val.csv \
    --saved_root ${saved_root} \
    --saved_dataset_folder results_kitti \
    --input_rgb_root ${input_rgb_root} \
    --process_length 110 \
    --gpus ${gpus} \
    --dataset kitti \

# infer bonn
python benchmark/infer/infer_batch.py \
    --meta_path ./eval/csv/meta_bonn.csv \
    --saved_root ${saved_root} \
    --saved_dataset_folder results_bonn \
    --input_rgb_root ${input_rgb_root} \
    --process_length 110 \
    --gpus ${gpus} \
    --dataset bonn \

# infer nyu
python benchmark/infer/infer_batch.py \
    --meta_path ./eval/csv/meta_nyu_test.csv \
    --saved_root ${saved_root} \
    --saved_dataset_folder results_nyu \
    --input_rgb_root ${input_rgb_root} \
    --process_length 1 \
    --gpus ${gpus} \
    --overlap 0 \
    --dataset nyu \
