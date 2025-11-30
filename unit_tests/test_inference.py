import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import os
import tempfile
from depthcrafter.inference import DepthCrafterInference


@pytest.fixture
def dummy_video_path():
    # Create a dummy video file (empty is fine since we mock read_video_frames)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@patch("depthcrafter.inference.DepthCrafterPipeline")
@patch("depthcrafter.inference.DiffusersUNetSpatioTemporalConditionModelDepthCrafter")
def test_init(mock_unet_cls, mock_pipeline_cls):
    mock_unet = MagicMock()
    mock_unet_cls.from_pretrained.return_value = mock_unet

    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

    # Test default (model offload)
    inference = DepthCrafterInference(
        unet_path="dummy_unet",
        pre_train_path="dummy_pretrain",
        cpu_offload="model",
        device="cpu",
    )
    mock_pipeline.enable_model_cpu_offload.assert_called()

    # Test sequential offload
    inference = DepthCrafterInference(
        unet_path="dummy_unet",
        pre_train_path="dummy_pretrain",
        cpu_offload="sequential",
        device="cpu",
    )
    mock_pipeline.enable_sequential_cpu_offload.assert_called()

    # Test no offload
    inference = DepthCrafterInference(
        unet_path="dummy_unet",
        pre_train_path="dummy_pretrain",
        cpu_offload=None,
        device="cpu",
    )
    mock_pipeline.to.assert_called_with("cpu")

    # Test invalid offload
    with pytest.raises(ValueError):
        DepthCrafterInference(
            unet_path="dummy_unet",
            pre_train_path="dummy_pretrain",
            cpu_offload="invalid",
            device="cpu",
        )


@patch("depthcrafter.inference.DepthCrafterPipeline")
@patch("depthcrafter.inference.DiffusersUNetSpatioTemporalConditionModelDepthCrafter")
def test_clear_cache(mock_unet_cls, mock_pipeline_cls):
    mock_pipeline_cls.from_pretrained.return_value = MagicMock()
    inference = DepthCrafterInference("dummy", "dummy")

    with (
        patch("depthcrafter.inference.gc.collect") as mock_gc,
        patch("depthcrafter.inference.torch.cuda.empty_cache") as mock_cuda,
    ):
        inference.clear_cache()
        mock_gc.assert_called_once()
        mock_cuda.assert_called_once()


@patch("depthcrafter.inference.DepthCrafterPipeline")
@patch("depthcrafter.inference.DiffusersUNetSpatioTemporalConditionModelDepthCrafter")
def test_save_exr(mock_unet_cls, mock_pipeline_cls):
    mock_pipeline_cls.from_pretrained.return_value = MagicMock()
    inference = DepthCrafterInference("dummy", "dummy")

    # Mock OpenEXR and Imath
    with patch.dict("sys.modules", {"OpenEXR": MagicMock(), "Imath": MagicMock()}):
        import OpenEXR

        res = np.random.rand(2, 32, 32).astype(np.float32)
        with patch("depthcrafter.inference.os.makedirs") as mock_makedirs:
            inference._save_exr(res, "output_path")

            mock_makedirs.assert_called_with("output_path", exist_ok=True)
            assert OpenEXR.OutputFile.call_count == 2


@patch("depthcrafter.inference.DepthCrafterPipeline")
@patch("depthcrafter.inference.DiffusersUNetSpatioTemporalConditionModelDepthCrafter")
@patch("depthcrafter.inference.read_video_frames")
@patch("depthcrafter.inference.vis_sequence_depth")
@patch("depthcrafter.inference.save_video")
@patch("depthcrafter.inference.os.makedirs")
@patch("depthcrafter.inference.np.savez_compressed")
def test_infer(
    mock_savez,
    mock_makedirs,
    mock_save_video,
    mock_vis,
    mock_read_video,
    mock_unet_cls,
    mock_pipeline_cls,
    dummy_video_path,
):
    # Setup mocks
    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

    inference = DepthCrafterInference("dummy", "dummy", cpu_offload=None, device="cpu")

    # Mock read_video_frames
    frames = np.random.rand(10, 32, 32, 3).astype(np.float32)
    mock_read_video.return_value = (frames, 30)

    # Mock pipeline output
    mock_output = MagicMock()
    mock_output.frames = [np.random.rand(10, 32, 32, 3)]
    mock_pipeline.return_value = mock_output

    # Mock vis
    mock_vis.return_value = np.random.rand(10, 32, 32, 3)

    # Run infer
    result_paths = inference.infer(
        video_path=dummy_video_path,
        num_denoising_steps=1,
        guidance_scale=1.0,
        save_folder="output",
        save_npz=True,
        save_exr=True,
    )

    assert len(result_paths) == 3
    mock_read_video.assert_called()
    mock_pipeline.assert_called()
    mock_vis.assert_called()
    assert mock_save_video.call_count == 3  # depth, vis, input
    mock_savez.assert_called()
    # _save_exr is called internally, we can check if it logs error or runs if modules are present
    # Since we didn't mock OpenEXR here, it should log error and return, which is fine.
