# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# # Data loading based on https://github.com/NVIDIA/flownet2-pytorch


import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import imageio


# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"


def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def extract_sintel(
    root,
    depth_root,
    sample_len=-1,
    csv_save_path="",
    datatset_name="",
    saved_rgb_dir="",
    saved_disp_dir="",
):
    scenes_names = os.listdir(root)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = os.listdir(os.path.join(root, seq_name))
        all_img_names = [x for x in all_img_names if x.endswith(".png")]
        all_img_names.sort()
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0][-4:]))

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step} / {seq_len // step}")

            video_imgs = []
            video_depths = []

            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(root, seq_name, all_img_names[idx])
                depth_path = osp.join(
                    depth_root, seq_name, all_img_names[idx][:-3] + "dpt"
                )

                depth = depth_read(depth_path)
                disp = depth

                video_depths.append(disp)
                video_imgs.append(np.array(Image.open(im_path)))

            disp_video = np.array(video_depths)[:, None]
            img_video = np.array(video_imgs)[..., 0:3]

            data_root = saved_rgb_dir + datatset_name
            disp_root = saved_disp_dir + datatset_name
            os.makedirs(data_root, exist_ok=True)
            os.makedirs(disp_root, exist_ok=True)

            img_video_dir = data_root
            disp_video_dir = disp_root

            img_video_path = os.path.join(img_video_dir, f"{seq_name}_rgb_left.mp4")
            disp_video_path = os.path.join(disp_video_dir, f"{seq_name}_disparity.npz")

            imageio.mimsave(
                img_video_path, img_video, fps=15, quality=10, macro_block_size=1
            )
            np.savez(disp_video_path, disparity=disp_video)

            sample = {}
            sample["filepath_left"] = os.path.join(
                f"{datatset_name}/{seq_name}_rgb_left.mp4"
            )
            sample["filepath_disparity"] = os.path.join(
                f"{datatset_name}/{seq_name}_disparity.npz"
            )

            all_samples.append(sample)

    filename_ = csv_save_path
    os.makedirs(os.path.dirname(filename_), exist_ok=True)
    fields = ["filepath_left", "filepath_disparity"]
    with open(filename_, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)

    print(f"{filename_} has been saved.")


if __name__ == "__main__":
    extract_sintel(
        root="path/to/Sintel-Depth/training_image/clean",
        depth_root="path/to/Sintel-Depth/MPI-Sintel-depth-training-20150305/training/depth",
        saved_rgb_dir="./benchmark/datasets/",
        saved_disp_dir="./benchmark/datasets/",
        csv_save_path=f"./benchmark/datasets/sintel.csv",
        sample_len=-1,
        datatset_name="sintel",
    )
