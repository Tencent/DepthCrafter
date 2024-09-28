#!/bin/sh
set -x
set -e

pred_disp_root=/path/to/saved/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] prediction
gt_disp_root=/path/to/gt_depth/root_directory # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] ground truth

# eval sintel
python benchmark/eval/eval.py \
    --meta_path ./eval/csv/meta_sintel.csv \
    --dataset_max_depth 70 \
    --dataset sintel \
    --seq_len 50 \
    --pred_disp_root ${pred_disp_root} \
    --gt_disp_root ${gt_disp_root} \

# eval scannet
python benchmark/eval/eval.py \
    --meta_path ./eval/csv/meta_scannet_test.csv \
    --dataset_max_depth 10 \
    --dataset scannet \
    --seq_len 90 \
    --pred_disp_root ${pred_disp_root} \
    --gt_disp_root ${gt_disp_root} \

# eval kitti
python benchmark/eval/eval.py \
    --meta_path ./eval/csv/meta_kitti_val.csv \
    --dataset_max_depth 80 \
    --dataset kitti \
    --seq_len 110 \
    --pred_disp_root ${pred_disp_root} \
    --gt_disp_root ${gt_disp_root} \

# eval bonn
python benchmark/eval/eval.py \
    --meta_path ./eval/csv/meta_bonn.csv \
    --dataset_max_depth 10 \
    --dataset bonn \
    --seq_len 110 \
    --pred_disp_root ${pred_disp_root} \
    --gt_disp_root ${gt_disp_root} \

# eval nyu
python benchmark/eval/eval.py \
    --meta_path ./eval/csv/meta_nyu_test.csv \
    --dataset_max_depth 10 \
    --dataset nyu \
    --seq_len 1 \
    --pred_disp_root ${pred_disp_root} \
    --gt_disp_root ${gt_disp_root} \
