import logging
from fire import Fire

from depthcrafter.inference import DepthCrafterInference

logging.basicConfig(level=logging.INFO)


def main(
    video_path: str,
    save_folder: str = "./demo_output",
    unet_path: str = "tencent/DepthCrafter",
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    process_length: int = -1,
    cpu_offload: str = "model",
    target_fps: int = -1,
    seed: int = 42,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    overlap: int = 25,
    max_res: int = 1024,
    dataset: str = "open",
    save_npz: bool = False,
    save_exr: bool = False,
    track_time: bool = False,
):
    """
    Main function to run DepthCrafter inference.

    Args:
        video_path (str): Path to the input video(s), separated by comma.
        save_folder (str): Folder to save output.
        unet_path (str): Path to the UNet model.
        pre_train_path (str): Path to the pre-trained model.
        process_length (int): Maximum number of frames to process.
        cpu_offload (str): CPU offload strategy.
        target_fps (int): Target FPS for output video.
        seed (int): Random seed.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Guidance scale.
        window_size (int): Window size for sliding window inference.
        overlap (int): Overlap between windows.
        max_res (int): Maximum resolution.
        dataset (str): Dataset name for resolution settings.
        save_npz (bool): Whether to save depth map as .npz.
        save_exr (bool): Whether to save depth map as .exr.
        track_time (bool): Whether to track execution time.
    """
    depthcrafter_inference = DepthCrafterInference(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
    )
    # process the videos, the video paths are separated by comma
    video_paths = video_path.split(",")
    for video in video_paths:
        depthcrafter_inference.infer(
            video,
            num_inference_steps,
            guidance_scale,
            save_folder=save_folder,
            window_size=window_size,
            process_length=process_length,
            overlap=overlap,
            max_res=max_res,
            dataset=dataset,
            target_fps=target_fps,
            seed=seed,
            track_time=track_time,
            save_npz=save_npz,
            save_exr=save_exr,
        )
        depthcrafter_inference.clear_cache()


if __name__ == "__main__":
    Fire(main)
