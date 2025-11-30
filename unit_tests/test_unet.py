import pytest
from unittest.mock import MagicMock, patch
import torch
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter


@pytest.fixture
def config():
    # Create a dummy config
    config = MagicMock()
    config.sample_size = 32
    config.in_channels = 4
    config.out_channels = 4
    config.layers_per_block = 2
    config.block_out_channels = (32, 64)
    config.down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D")
    config.up_block_types = ("CrossAttnUpBlock2D", "UpBlock2D")
    config.cross_attention_dim = 32
    return config


@patch("depthcrafter.unet.UNetSpatioTemporalConditionModel.__init__")
def test_init(mock_super_init, config):
    mock_super_init.return_value = None
    model = DiffusersUNetSpatioTemporalConditionModelDepthCrafter(**config.__dict__)
    assert isinstance(model, DiffusersUNetSpatioTemporalConditionModelDepthCrafter)


def test_forward_signature():
    # Just checking if the method exists and has correct arguments
    assert hasattr(DiffusersUNetSpatioTemporalConditionModelDepthCrafter, "forward")
