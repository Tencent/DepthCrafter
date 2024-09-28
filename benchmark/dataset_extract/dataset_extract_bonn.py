import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import imageio
import csv


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array

    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255

    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def extract_bonn(
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
        # load all images
        all_img_names = os.listdir(osp.join(depth_root, seq_name, "rgb"))
        all_img_names = [x for x in all_img_names if x.endswith(".png")]
        print(f"sequence frame number: {len(all_img_names)}")

        # for not zero padding image name
        all_img_names.sort()
        all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0][-4:]))
        all_img_names = all_img_names[start_frame:end_frame]

        all_depth_names = os.listdir(osp.join(depth_root, seq_name, "depth"))
        all_depth_names = [x for x in all_depth_names if x.endswith(".png")]
        print(f"sequence depth number: {len(all_depth_names)}")

        # for not zero padding image name
        all_depth_names.sort()
        all_depth_names = sorted(
            all_depth_names, key=lambda x: int(x.split(".")[0][-4:])
        )
        all_depth_names = all_depth_names[start_frame:end_frame]

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

            # for idx in range(ref_idx, ref_idx + step):
            for idx in range(ref_idx, ref_e):
                im_path = osp.join(root, seq_name, "rgb", all_img_names[idx])
                depth_path = osp.join(
                    depth_root, seq_name, "depth", all_depth_names[idx]
                )

                depth = depth_read(depth_path)
                disp = depth

                video_depths.append(disp)
                video_imgs.append(np.array(Image.open(im_path)))

            disp_video = np.array(video_depths)[:, None]  # [:, 0:1, :, :, 0]
            img_video = np.array(video_imgs)[..., 0:3]  # [:, 0, :, :, 0:3]

            print(disp_video.max(), disp_video.min())

            def even_or_odd(num):
                if num % 2 == 0:
                    return num
                else:
                    return num - 1

            # print(disp_video.shape)
            # print(img_video.shape)
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
                img_video_path, img_video, fps=15, quality=9, macro_block_size=1
            )
            np.savez(disp_video_path, disparity=disp_video)

            sample = {}
            sample["filepath_left"] = os.path.join(
                f"{datatset_name}/{seq_name}_rgb_left.mp4"
            )  # img_video_path
            sample["filepath_disparity"] = os.path.join(
                f"{datatset_name}/{seq_name}_disparity.npz"
            )  # disp_video_path

            all_samples.append(sample)

    # save csv file

    filename_ = csv_save_path
    os.makedirs(os.path.dirname(filename_), exist_ok=True)
    fields = ["filepath_left", "filepath_disparity"]
    with open(filename_, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)

    print(f"{filename_} has been saved.")


if __name__ == "__main__":
    extract_bonn(
        root="path/to/Bonn-RGBD",
        depth_root="path/to/Bonn-RGBD",
        saved_rgb_dir="./benchmark/datasets/",
        saved_disp_dir="./benchmark/datasets/",
        csv_save_path=f"./benchmark/datasets/bonn.csv",
        sample_len=-1,
        datatset_name="bonn",
        start_frame=30,
        end_frame=140,
    )
