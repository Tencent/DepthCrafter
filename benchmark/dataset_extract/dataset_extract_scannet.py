import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import imageio


def _read_image(img_rel_path) -> np.ndarray:
    image_to_read = img_rel_path
    image = Image.open(image_to_read)  # [H, W, rgb]
    image = np.asarray(image)
    return image


def depth_read(filename):
    depth_in = _read_image(filename)
    depth_decoded = depth_in / 1000.0
    return depth_decoded


def extract_scannet(
    root,
    sample_len=-1,
    csv_save_path="",
    datatset_name="",
    scene_number=16,
    scene_frames_len=120,
    stride=1,
    saved_rgb_dir="",
    saved_disp_dir="",
):
    scenes_names = os.listdir(root)
    scenes_names = sorted(scenes_names)[:scene_number]
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = os.listdir(osp.join(root, seq_name, "color"))
        all_img_names = [x for x in all_img_names if x.endswith(".jpg")]
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0]))
        all_img_names = all_img_names[:scene_frames_len:stride]
        print(f"sequence frame number: {len(all_img_names)}")

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step + 1} / {seq_len//step}")

            video_imgs = []
            video_depths = []

            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(root, seq_name, "color", all_img_names[idx])
                depth_path = osp.join(
                    root, seq_name, "depth", all_img_names[idx][:-3] + "png"
                )

                depth = depth_read(depth_path)
                disp = depth

                video_depths.append(disp)
                video_imgs.append(np.array(Image.open(im_path)))

            disp_video = np.array(video_depths)[:, None]
            img_video = np.array(video_imgs)[..., 0:3]

            disp_video = disp_video[:, :, 8:-8, 11:-11]
            img_video = img_video[:, 8:-8, 11:-11, :]

            data_root = saved_rgb_dir + datatset_name
            disp_root = saved_disp_dir + datatset_name
            os.makedirs(data_root, exist_ok=True)
            os.makedirs(disp_root, exist_ok=True)

            img_video_dir = data_root
            disp_video_dir = disp_root

            img_video_path = os.path.join(img_video_dir, f"{seq_name}_rgb_left.mp4")
            disp_video_path = os.path.join(disp_video_dir, f"{seq_name}_disparity.npz")

            imageio.mimsave(
                img_video_path, img_video, fps=15, quality=9, macro_block_size=1
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
    extract_scannet(
        root="path/to/ScanNet_v2/raw/scans_test",
        saved_rgb_dir="./benchmark/datasets/",
        saved_disp_dir="./benchmark/datasets/",
        csv_save_path=f"./benchmark/datasets/scannet.csv",
        sample_len=-1,
        datatset_name="scannet",
        scene_number=100,
        scene_frames_len=90 * 3,
        stride=3,
    )
