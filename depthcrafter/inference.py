import gc
import logging
import os
from typing import List, Optional

import numpy as np
import torch
from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import read_video_frames, save_video, vis_sequence_depth

logger = logging.getLogger(__name__)


class DepthCrafterInference:
    """
    Inference class for DepthCrafter.
    """

    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: Optional[str] = "model",
        device: str = "cuda",
    ):
        """
        Initialize the DepthCrafter inference pipeline.

        Args:
            unet_path (str): Path to the UNet model.
            pre_train_path (str): Path to the pre-trained model.
            cpu_offload (Optional[str]): CPU offload strategy ("model", "sequential", or None).
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        logger.info(f"Loading UNet from {unet_path}")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        logger.info(f"Loading pipeline from {pre_train_path}")
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        if cpu_offload is not None:
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            logger.warning(f"Xformers is not enabled: {e}")

        self.pipe.enable_attention_slicing()

    def infer(
        self,
        video_path: str,
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
    ) -> List[str]:
        """
        Run inference on a video.

        Args:
            video_path (str): Path to the input video.
            num_denoising_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale.
            save_folder (str): Folder to save output.
            window_size (int): Window size for sliding window inference.
            process_length (int): Maximum number of frames to process.
            overlap (int): Overlap between windows.
            max_res (int): Maximum resolution.
            dataset (str): Dataset name for resolution settings.
            target_fps (int): Target FPS for output video.
            seed (int): Random seed.
            track_time (bool): Whether to track execution time.
            save_npz (bool): Whether to save depth map as .npz.
            save_exr (bool): Whether to save depth map as .exr.

        Returns:
            List[str]: List of paths to saved files.
        """
        set_seed(seed)

        frames, target_fps = read_video_frames(
            video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
        )

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

        res = res.sum(-1) / res.shape[-1]
        res = (res - res.min()) / (res.max() - res.min())
        vis = vis_sequence_depth(res)

        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video_path))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_video(res, save_path + "_depth.mp4", fps=target_fps)
        save_video(vis, save_path + "_vis.mp4", fps=target_fps)
        save_video(frames, save_path + "_input.mp4", fps=target_fps)

        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)

        if save_exr:
            self._save_exr(res, save_path)

        return [
            save_path + "_input.mp4",
            save_path + "_vis.mp4",
            save_path + "_depth.mp4",
        ]

    def _save_exr(self, res: np.ndarray, save_path: str):
        """
        Save results as EXR files.
        """
        try:
            import OpenEXR
            import Imath
        except ImportError:
            logger.error("OpenEXR or Imath not installed. Skipping EXR saving.")
            return

        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving EXR results to {save_path}")

        for i, frame in enumerate(res):
            output_exr = f"{save_path}/frame_{i:04d}.exr"
            header = OpenEXR.Header(frame.shape[1], frame.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": frame.tobytes()})
            exr_file.close()

    def clear_cache(self):
        """Clear CUDA cache."""
        gc.collect()
        torch.cuda.empty_cache()
