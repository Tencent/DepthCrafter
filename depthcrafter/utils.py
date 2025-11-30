import logging
import tempfile
from typing import List, Optional, Tuple, Union

import matplotlib
import mediapy
import numpy as np
import PIL.Image
import torch
from decord import VideoReader, cpu

logger = logging.getLogger(__name__)

dataset_res_dict = {
    "sintel": [448, 1024],
    "scannet": [640, 832],
    "KITTI": [384, 1280],
    "bonn": [512, 640],
    "NYUv2": [448, 640],
}


def read_video_frames(
    video_path: str,
    process_length: int,
    target_fps: int,
    max_res: int,
    dataset: str = "open",
) -> Tuple[np.ndarray, int]:
    """
    Read video frames from a file, resize and downsample them.

    Args:
        video_path (str): Path to the video file.
        process_length (int): Maximum number of frames to process.
        target_fps (int): Target FPS for the output.
        max_res (int): Maximum resolution (height or width).
        dataset (str): Dataset name for resolution settings.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the frames (numpy array) and the actual FPS.
    """
    if dataset == "open":
        logger.info(f"Processing video: {video_path}")
        vid = VideoReader(video_path, ctx=cpu(0))
        logger.info(
            f"Original video shape: {(len(vid), *vid.get_batch([0]).shape[1:])}"
        )
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    logger.info(
        f"Downsampled shape: {(len(frames_idx), *vid.get_batch([0]).shape[1:])}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    logger.info(
        f"Final processing shape: {(len(frames_idx), *vid.get_batch([0]).shape[1:])}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames, fps


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image], np.ndarray],
    output_video_path: Optional[str] = None,
    fps: int = 10,
    crf: int = 18,
) -> str:
    """
    Save video frames to a file.

    Args:
        video_frames (Union[List[np.ndarray], List[PIL.Image.Image], np.ndarray]): List of frames or numpy array.
        output_video_path (Optional[str]): Path to save the video. If None, a temporary file is created.
        fps (int): Frames per second.
        crf (int): Constant Rate Factor for encoding quality.

    Returns:
        str: Path to the saved video.
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames, np.ndarray):
        # If it's a numpy array, we assume it's already in the correct format or needs simple conversion
        if video_frames.dtype != np.uint8:
            video_frames = (video_frames * 255).astype(np.uint8)
    elif isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path


class ColorMapper:
    """
    A color mapper to map depth values to a certain colormap.
    """

    def __init__(self, colormap: str = "inferno"):
        """
        Initialize the ColorMapper.

        Args:
            colormap (str): Name of the colormap to use.
        """
        self.colormap = torch.tensor(matplotlib.colormaps[colormap].colors)

    def apply(
        self,
        image: torch.Tensor,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply the colormap to an image.

        Args:
            image (torch.Tensor): Input image tensor.
            v_min (Optional[float]): Minimum value for normalization.
            v_max (Optional[float]): Maximum value for normalization.

        Returns:
            torch.Tensor: Color-mapped image.
        """
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        # Clamp values to be within valid range for indexing
        image = torch.clamp(image, 0, 255)
        image = self.colormap[image]
        return image


def vis_sequence_depth(
    depths: np.ndarray, v_min: Optional[float] = None, v_max: Optional[float] = None
) -> np.ndarray:
    """
    Visualize a sequence of depth maps.

    Args:
        depths (np.ndarray): Input depth maps.
        v_min (Optional[float]): Minimum value for normalization.
        v_max (Optional[float]): Maximum value for normalization.

    Returns:
        np.ndarray: Visualized depth maps.
    """
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
