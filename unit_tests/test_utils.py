import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import os
import tempfile
import mediapy
from depthcrafter.utils import (
    read_video_frames,
    save_video,
    ColorMapper,
    vis_sequence_depth,
)


@pytest.fixture
def dummy_video_path():
    # Create a dummy video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
        # Write some dummy bytes so the file is not empty (though decord is mocked anyway)
        f.write(b"dummy video content")

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@patch("depthcrafter.utils.VideoReader")
def test_read_video_frames(mock_video_reader, dummy_video_path):
    # Mock VideoReader
    mock_vr_instance = MagicMock()
    mock_video_reader.return_value = mock_vr_instance

    # Mock video properties
    mock_vr_instance.__len__.return_value = 10
    mock_vr_instance.get_avg_fps.return_value = 10.0

    # Mock get_batch
    # Shape: [batch, height, width, channels]
    mock_batch = MagicMock()
    mock_batch.shape = (1, 32, 32, 3)
    mock_batch.asnumpy.return_value = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    mock_vr_instance.get_batch.return_value = mock_batch

    # Test call with dummy path (even though mocked, good to have valid path string)
    frames, fps = read_video_frames(
        dummy_video_path, process_length=10, target_fps=10, max_res=32
    )

    assert fps == 10
    assert isinstance(frames, np.ndarray)
    # Check if VideoReader was called
    mock_video_reader.assert_called()


@patch("depthcrafter.utils.VideoReader")
def test_read_video_frames_dataset(mock_video_reader, dummy_video_path):
    # Mock VideoReader
    mock_vr_instance = MagicMock()
    mock_video_reader.return_value = mock_vr_instance
    mock_vr_instance.__len__.return_value = 10
    mock_vr_instance.get_avg_fps.return_value = 10.0

    mock_batch = MagicMock()
    mock_batch.shape = (1, 32, 32, 3)
    mock_batch.asnumpy.return_value = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    mock_vr_instance.get_batch.return_value = mock_batch

    # Test with dataset="sintel"
    # sintel resolution is [448, 1024] (height, width)
    frames, fps = read_video_frames(
        dummy_video_path, process_length=10, target_fps=10, max_res=32, dataset="sintel"
    )

    # Check if VideoReader was initialized with specific width/height
    # Note: We use pytest.any() for ctx because it's a decord.cpu(0) object
    call_args = mock_video_reader.call_args
    assert call_args is not None
    assert call_args[1]["width"] == 1024
    assert call_args[1]["height"] == 448


@patch("depthcrafter.utils.mediapy.write_video")
def test_save_video(mock_write_video):
    frames = np.zeros((10, 32, 32, 3), dtype=np.float32)
    output_path = save_video(frames, "output.mp4", fps=10)

    assert output_path == "output.mp4"
    mock_write_video.assert_called_once()

    # Test with temp file
    output_path_temp = save_video(frames, None, fps=10)
    assert output_path_temp.endswith(".mp4")


def test_color_mapper():
    mapper = ColorMapper(colormap="inferno")
    image = torch.rand((32, 32))
    colored_image = mapper.apply(image)

    assert colored_image.shape == (32, 32, 3)
    assert isinstance(colored_image, torch.Tensor)


def test_vis_sequence_depth():
    depths = np.random.rand(10, 32, 32).astype(np.float32)
    vis = vis_sequence_depth(depths)

    assert isinstance(vis, np.ndarray)
    assert vis.shape == (10, 32, 32, 3)  # Assuming RGB output
