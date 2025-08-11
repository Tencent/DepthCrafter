# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DepthCrafter is a deep learning project for generating temporally consistent long depth sequences from open-world videos. It uses a diffusion-based model built on Stable Video Diffusion to estimate depth maps without requiring camera poses or optical flow.

## Architecture

### Core Components

1. **Main Pipeline (`depthcrafter/depth_crafter_ppl.py`)**: Implements the DepthCrafterPipeline extending diffusers for depth estimation
2. **UNet Model (`depthcrafter/unet.py`)**: Custom spatio-temporal UNet for depth prediction
3. **Inference Scripts**:
   - `run.py`: Main CLI for single video inference
   - `app.py`: Gradio web interface
   - `benchmark/infer/infer_batch.py`: Batch processing for benchmarks

### Key Directories

- `depthcrafter/`: Core model implementation
- `benchmark/`: Dataset evaluation scripts and CSV metadata
- `examples/`: Sample video files for testing
- `visualization/`: Point cloud visualization tools

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Single Video Inference

High-resolution (requires ~26GB GPU memory):
```bash
python run.py --video-path examples/example_01.mp4
```

Low-resolution (requires ~9GB GPU memory):
```bash
python run.py --video-path examples/example_01.mp4 --max-res 512
```

### Gradio Demo
```bash
gradio app.py
```

### Benchmark Evaluation

Run inference on all datasets:
```bash
bash benchmark/infer/infer.sh
```

Evaluate results:
```bash
bash benchmark/eval/eval.sh
```

### Key Parameters

- `--process-length`: Number of frames to process (default: 195)
- `--window-size`: Sliding window size (default: 110)
- `--overlap`: Frame overlap between windows (default: 25)
- `--max-res`: Maximum resolution (default: 1024)
- `--num-denoising-steps`: Denoising steps (default: 5)
- `--guidance-scale`: Guidance scale for inference (default: 1.0)
- `--save-npz`: Save depth as NPZ file
- `--save-exr`: Save depth as EXR file

## Model Loading

The model uses two key components from Hugging Face:
1. DepthCrafter UNet: `tencent/DepthCrafter`
2. Base diffusion model: `stabilityai/stable-video-diffusion-img2vid-xt`

## Dependencies

Key dependencies:
- PyTorch 2.0.1
- Diffusers 0.29.1
- Transformers 4.41.2
- XFormers 0.0.20 (for memory efficient attention)
- OpenEXR 3.2.4 (for EXR output)

## Performance Notes

- v1.0.1 improvements: ~4x faster inference (465ms/frame vs 1914ms/frame at 1024x576)
- Memory optimization options via `--cpu-offload` parameter:
  - `"model"`: Standard CPU offloading
  - `"sequential"`: Sequential offloading (slower but saves more memory)