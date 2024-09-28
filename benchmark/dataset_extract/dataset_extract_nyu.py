import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import imageio


def _read_image(img_rel_path) -> np.ndarray:
    image_to_read = img_rel_path
    image = Image.open(image_to_read)
    image = np.asarray(image)
    return image


def depth_read(filename):
    depth_in = _read_image(filename)
    depth_decoded = depth_in / 1000.0
    return depth_decoded


def extract_nyu(
    root,
    depth_root,
    csv_save_path="",
    datatset_name="",
    filename_ls_path="",
    saved_rgb_dir="",
    saved_disp_dir="",
):
    with open(filename_ls_path, "r") as f:
        filenames = [s.split() for s in f.readlines()]

    all_samples = []
    for i, pair_names in enumerate(tqdm(filenames)):
        img_name = pair_names[0]
        filled_depth_name = pair_names[2]

        im_path = osp.join(root, img_name)
        depth_path = osp.join(depth_root, filled_depth_name)

        depth = depth_read(depth_path)
        disp = depth

        video_depths = [disp]
        video_imgs = [np.array(Image.open(im_path))]

        disp_video = np.array(video_depths)[:, None]
        img_video = np.array(video_imgs)[..., 0:3]

        disp_video = disp_video[:, :, 45:471, 41:601]
        img_video = img_video[:, 45:471, 41:601, :]

        data_root = saved_rgb_dir + datatset_name
        disp_root = saved_disp_dir + datatset_name
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(disp_root, exist_ok=True)

        img_video_dir = data_root
        disp_video_dir = disp_root

        img_video_path = os.path.join(img_video_dir, f"{img_name[:-4]}_rgb_left.mp4")
        disp_video_path = os.path.join(disp_video_dir, f"{img_name[:-4]}_disparity.npz")

        dir_name = os.path.dirname(img_video_path)
        os.makedirs(dir_name, exist_ok=True)
        dir_name = os.path.dirname(disp_video_path)
        os.makedirs(dir_name, exist_ok=True)

        imageio.mimsave(
            img_video_path, img_video, fps=15, quality=10, macro_block_size=1
        )
        np.savez(disp_video_path, disparity=disp_video)

        sample = {}
        sample["filepath_left"] = os.path.join(
            f"{datatset_name}/{img_name[:-4]}_rgb_left.mp4"
        )
        sample["filepath_disparity"] = os.path.join(
            f"{datatset_name}/{img_name[:-4]}_disparity.npz"
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
    extract_nyu(
        root="path/to/NYUv2/",
        depth_root="path/to/NYUv2/",
        filename_ls_path="path/to/NYUv2/filename_list_test.txt",
        saved_rgb_dir="./benchmark/datasets/",
        saved_disp_dir="./benchmark/datasets/",
        csv_save_path=f"./benchmark/datasets/NYUv2.csv",
        datatset_name="NYUv2",
    )
