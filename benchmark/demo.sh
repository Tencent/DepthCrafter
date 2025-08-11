#!/bin/sh
set -x
set -e

test_case=$1
gpu_id=$2
process_length=$3
saved_root=$4
saved_dataset_folder=$5
overlap=$6
dataset=$7

PYTHONPATH=. python run.py \
  --video-path ${test_case} \
  --save-folder ${saved_root}/${saved_dataset_folder} \
  --process-length ${process_length} \
  --dataset ${dataset} \
  --overlap ${overlap}