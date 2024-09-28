import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import imageio


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255

    depth = depth_png.astype(np.float64) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def extract_kitti(
    root,
    depth_root,
    sample_len=-1,
    csv_save_path="",
    datatset_name="",
    saved_rgb_dir="",
    saved_disp_dir="",
    start_frame=0,
    end_frame=110,
):
    scenes_names = os.listdir(depth_root)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = os.listdir(
            osp.join(depth_root, seq_name, "proj_depth/groundtruth/image_02")
        )
        all_img_names = [x for x in all_img_names if x.endswith(".png")]
        print(f"sequence frame number: {len(all_img_names)}")

        all_img_names.sort()
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0][-4:]))
        all_img_names = all_img_names[start_frame:end_frame]

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
                im_path = osp.join(
                    root, seq_name[0:10], seq_name, "image_02/data", all_img_names[idx]
                )
                depth_path = osp.join(
                    depth_root,
                    seq_name,
                    "proj_depth/groundtruth/image_02",
                    all_img_names[idx],
                )

                depth = depth_read(depth_path)
                disp = depth

                video_depths.append(disp)
                video_imgs.append(np.array(Image.open(im_path)))

            disp_video = np.array(video_depths)[:, None]
            img_video = np.array(video_imgs)[..., 0:3]

            def even_or_odd(num):
                if num % 2 == 0:
                    return num
                else:
                    return num - 1

            height = disp_video.shape[-2]
            width = disp_video.shape[-1]
            height = even_or_odd(height)
            width = even_or_odd(width)
            disp_video = disp_video[:, :, 0:height, 0:width]
            img_video = img_video[:, 0:height, 0:width]

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
            sample["filepath_left"] = os.path.join(f"KITTI/{seq_name}_rgb_left.mp4")
            sample["filepath_disparity"] = os.path.join(
                f"KITTI/{seq_name}_disparity.npz"
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
    extract_kitti(
        root="path/to/KITTI/raw_data",
        depth_root="path/to/KITTI/data_depth_annotated/val",
        saved_rgb_dir="./benchmark/datasets/",
        saved_disp_dir="./benchmark/datasets/",
        csv_save_path=f"./benchmark/datasets/KITTI.csv",
        sample_len=-1,
        datatset_name="KITTI",
        start_frame=0,
        end_frame=110,
    )
