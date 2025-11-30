import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import numpy as np
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline


@pytest.fixture
def pipeline():
    vae = MagicMock()
    text_encoder = MagicMock()
    tokenizer = MagicMock()
    unet = MagicMock()
    scheduler = MagicMock()
    feature_extractor = MagicMock()
    image_encoder = MagicMock()

    return DepthCrafterPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )


def test_check_inputs(pipeline):
    video = torch.randn(2, 3, 32, 32)
    # Should pass with tensor
    pipeline.check_inputs(video, 32, 32)

    # Should pass with numpy array
    video_np = np.random.randn(2, 32, 32, 3)
    pipeline.check_inputs(video_np, 32, 32)

    # Should fail due to type
    with pytest.raises(ValueError):
        pipeline.check_inputs("invalid", 32, 32)

    # Should fail due to dimensions
    with pytest.raises(ValueError):
        pipeline.check_inputs(video, 30, 32)


@patch("depthcrafter.depth_crafter_ppl._resize_with_antialiasing")
def test_encode_video(mock_resize, pipeline):
    video = torch.randn(2, 3, 32, 32)  # batch, channels, height, width
    mock_resize.return_value = video

    # Mock feature extractor output
    mock_pixel_values = MagicMock()
    mock_pixel_values.pixel_values = torch.randn(2, 3, 32, 32)
    pipeline.feature_extractor.return_value = mock_pixel_values

    # Mock image encoder output
    mock_image_embeds = MagicMock()
    mock_image_embeds.image_embeds = torch.randn(2, 1024)
    pipeline.image_encoder.return_value = mock_image_embeds

    embeddings = pipeline.encode_video(video)

    assert embeddings.shape == (2, 1024)


def test_encode_vae_video(pipeline):
    video = torch.randn(2, 3, 32, 32)

    # Mock VAE encode
    mock_latent_dist = MagicMock()
    mock_latent_dist.mode.return_value = torch.randn(2, 4, 4, 4)
    mock_encoder_output = MagicMock()
    mock_encoder_output.latent_dist = mock_latent_dist
    pipeline.vae.encode.return_value = mock_encoder_output

    latents = pipeline.encode_vae_video(video)

    assert latents.shape == (2, 4, 4, 4)


def test_pipeline_call(pipeline):
    # Mock components and methods
    pipeline.check_inputs = MagicMock()
    pipeline.encode_video = MagicMock(return_value=torch.randn(10, 1024))
    pipeline.encode_vae_video = MagicMock(return_value=torch.randn(10, 4, 32, 32))
    pipeline._get_add_time_ids = MagicMock(return_value=torch.randn(1, 3))
    pipeline.prepare_latents = MagicMock(return_value=torch.randn(1, 10, 4, 32, 32))
    pipeline.decode_latents = MagicMock(return_value=torch.randn(1, 10, 32, 32, 3))
    pipeline.video_processor = MagicMock()
    pipeline.video_processor.postprocess_video.return_value = torch.randn(
        1, 10, 32, 32, 3
    )

    # Mock scheduler
    pipeline.scheduler.timesteps = [0]
    pipeline.scheduler.order = 1
    pipeline.scheduler.init_noise_sigma = 1.0
    pipeline.scheduler.sigmas = [1.0]
    pipeline.scheduler.scale_model_input.return_value = torch.randn(1, 10, 4, 32, 32)
    pipeline.scheduler.step.return_value.prev_sample = torch.randn(1, 10, 4, 32, 32)

    # Mock UNet
    pipeline.unet.config.sample_size = 32
    pipeline.unet.config.in_channels = 4
    pipeline.unet.return_value = (torch.randn(1, 10, 4, 32, 32),)

    # Mock VAE
    pipeline.vae_scale_factor = 8
    pipeline.vae.dtype = torch.float32
    pipeline.vae.config.force_upcast = False

    # Input video
    video = torch.randn(10, 3, 32, 32)

    # Run pipeline
    output = pipeline(
        video=video,
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=1.0,
        window_size=10,
        decode_chunk_size=2,
        output_type="pt",
        return_dict=True,
    )

    assert output.frames.shape == (1, 10, 32, 32, 3)
    pipeline.check_inputs.assert_called()
    pipeline.encode_video.assert_called()
    pipeline.encode_vae_video.assert_called()
    pipeline.unet.assert_called()


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.Event")
@patch("torch.cuda.synchronize")
@patch("torch.cuda.empty_cache")
def test_pipeline_call_cuda_logic(
    mock_empty_cache, mock_sync, mock_event, mock_cuda_available, pipeline
):
    # Mock components and methods
    pipeline.check_inputs = MagicMock()
    pipeline.encode_video = MagicMock(return_value=torch.randn(10, 1024))
    pipeline.encode_vae_video = MagicMock(return_value=torch.randn(10, 4, 32, 32))
    pipeline._get_add_time_ids = MagicMock(return_value=torch.randn(1, 3))
    pipeline.prepare_latents = MagicMock(return_value=torch.randn(1, 10, 4, 32, 32))
    pipeline.decode_latents = MagicMock(return_value=torch.randn(1, 10, 32, 32, 3))
    pipeline.video_processor = MagicMock()
    pipeline.video_processor.postprocess_video.return_value = torch.randn(
        1, 10, 32, 32, 3
    )

    # Mock scheduler
    pipeline.scheduler.timesteps = [0]
    pipeline.scheduler.order = 1
    pipeline.scheduler.init_noise_sigma = 1.0
    pipeline.scheduler.sigmas = [1.0]
    pipeline.scheduler.scale_model_input.return_value = torch.randn(1, 10, 4, 32, 32)
    pipeline.scheduler.step.return_value.prev_sample = torch.randn(1, 10, 4, 32, 32)

    # Mock UNet
    pipeline.unet.config.sample_size = 32
    pipeline.unet.config.in_channels = 4
    pipeline.unet.return_value = (torch.randn(1, 10, 4, 32, 32),)

    # Mock VAE
    pipeline.vae_scale_factor = 8
    pipeline.vae.dtype = torch.float32
    pipeline.vae.config.force_upcast = False

    # Input video
    video = torch.randn(10, 3, 32, 32)

    # Run pipeline with track_time=True
    output = pipeline(
        video=video,
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=1.0,
        window_size=10,
        decode_chunk_size=2,
        output_type="pt",
        return_dict=True,
        track_time=True,
    )

    assert output.frames.shape == (1, 10, 32, 32, 3)

    # Verify CUDA calls
    assert mock_event.called
    assert mock_sync.called
    assert mock_empty_cache.called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_gpu_integration():
    """Test GPU tensor operations in the pipeline with minimal overhead."""
    device = torch.device("cuda")

    # Create pipeline with fully mocked components
    vae = MagicMock()
    unet = MagicMock()
    scheduler = MagicMock()
    feature_extractor = MagicMock()
    image_encoder = MagicMock()

    pipeline = DepthCrafterPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )

    # Mock all pipeline methods to bypass actual computation
    pipeline.check_inputs = MagicMock()
    pipeline.encode_video = MagicMock(return_value=torch.randn(10, 1024, device=device))
    pipeline.encode_vae_video = MagicMock(
        return_value=torch.randn(10, 4, 4, 4, device=device)
    )
    pipeline._get_add_time_ids = MagicMock(
        return_value=torch.randn(1, 3, device=device)
    )
    pipeline.prepare_latents = MagicMock(
        return_value=torch.randn(1, 10, 4, 4, 4, device=device)
    )
    pipeline.decode_latents = MagicMock(
        return_value=torch.randn(1, 10, 32, 32, 3, device=device)
    )

    # Mock video_processor
    pipeline.video_processor = MagicMock()
    pipeline.video_processor.postprocess_video.return_value = torch.randn(
        1, 10, 32, 32, 3, device=device
    )

    # Mock scheduler
    pipeline.scheduler.timesteps = [0]
    pipeline.scheduler.order = 1
    pipeline.scheduler.init_noise_sigma = 1.0
    pipeline.scheduler.sigmas = [1.0]
    pipeline.scheduler.scale_model_input.return_value = torch.randn(
        1, 10, 4, 4, 4, device=device
    )
    pipeline.scheduler.step.return_value.prev_sample = torch.randn(
        1, 10, 4, 4, 4, device=device
    )

    # Mock UNet
    pipeline.unet.config.sample_size = 32
    pipeline.unet.config.in_channels = 4
    pipeline.unet.return_value = (torch.randn(1, 10, 4, 4, 4, device=device),)

    # Mock VAE
    pipeline.vae_scale_factor = 8
    pipeline.vae.dtype = torch.float32
    pipeline.vae.config.force_upcast = False

    # Input video (small for speed)
    video = torch.randn(10, 3, 32, 32)

    # Run pipeline
    output = pipeline(
        video=video,
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=1.0,
        window_size=10,
        decode_chunk_size=2,
        output_type="pt",
        return_dict=True,
    )

    # Verify output is on GPU
    assert output.frames.device.type == "cuda"
    assert output.frames.shape == (1, 10, 32, 32, 3)

    # Verify methods were called
    pipeline.encode_video.assert_called()
    pipeline.encode_vae_video.assert_called()
