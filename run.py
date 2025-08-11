import gc
import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
from fire import Fire

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import vis_sequence_depth, save_video, read_video_frames


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float32,
        )

        # Determine the target device
        # Note: MPS doesn't support Conv3D operations required by this model
        # So we use CPU for Apple Silicon devices
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        # Handle model placement and offloading
        if cpu_offload is not None and torch.cuda.is_available():
            # CPU offloading only works with CUDA
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            # For MPS or CPU, just move the entire model to the device
            self.pipe.to(device)
        # enable attention slicing and xformers memory efficient attention (only for CUDA)
        if torch.cuda.is_available():
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Xformers not enabled: {e}")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        num_denoising_steps: int,
        guidance_scale: float,
        save_folder: str = "./demo_output",
        window_size: int = 110,
        process_length: int = 195,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = 15,
        seed: int = 42,
        track_time: bool = True,
        save_npz: bool = False,
        save_exr: bool = False,
        max_frames: int = -1,
        start_frame: int = 0,
    ):
        set_seed(seed)

        # Use max_frames if specified, otherwise use process_length
        frame_limit = max_frames if max_frames > 0 else process_length
        
        frames, target_fps = read_video_frames(
            video,
            frame_limit,
            target_fps,
            max_res,
            dataset,
            start_frame=start_frame,
        )
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video(res, save_path + "_depth.mp4", fps=target_fps)
        save_video(vis, save_path + "_vis.mp4", fps=target_fps)
        save_video(frames, save_path + "_input.mp4", fps=target_fps)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)
        if save_exr:
            import OpenEXR
            import Imath

            os.makedirs(save_path, exist_ok=True)
            print(f"==> saving EXR results to {save_path}")
            # Iterate over each frame and save as a separate EXR file
            for i, frame in enumerate(res):
                output_exr = f"{save_path}/frame_{i:04d}.exr"

                # Prepare EXR header for each frame
                header = OpenEXR.Header(frame.shape[1], frame.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }

                # Create EXR file and write the frame
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": frame.tobytes()})
                exr_file.close()

        return [
            save_path + "_input.mp4",
            save_path + "_vis.mp4",
            save_path + "_depth.mp4",
        ]

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            max_res=max_res,
            process_length=process_length,
        )
        # clear the cache for the next video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res_path[:2]


def main(
    video_path: str,
    save_folder: str = "./demo_output",
    unet_path: str = "tencent/DepthCrafter",
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    process_length: int = -1,
    cpu_offload: str = None,
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
    max_frames: int = -1,
    start_frame: int = 0,
):
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
    )
    # process the videos, the video paths are separated by comma
    video_paths = video_path.split(",")
    for video in video_paths:
        depthcrafter_demo.infer(
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
            max_frames=max_frames,
            start_frame=start_frame,
        )
        # clear the cache for the next video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # running configs
    # the most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # the most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    # Use `max_frames` to limit the number of frames to process (e.g., --max-frames 50)
    # Use `start_frame` to start from a specific frame (e.g., --start-frame 100)
    Fire(main)
