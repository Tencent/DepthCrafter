# DepthCrafter for macOS (Apple Silicon & Intel)

This is a modified version of DepthCrafter optimized for macOS, with full support for Apple Silicon (M1/M2/M3) and Intel Macs. The modifications enable CPU-based processing since MPS (Metal Performance Shaders) doesn't support Conv3D operations required by the model.

## 🍎 Key Modifications for macOS

### 1. **CPU-Only Processing**
- Removed CUDA dependencies
- Disabled MPS due to Conv3D limitations
- Uses CPU for all computations (slower but fully functional)

### 2. **FP32 Precision**
- Changed from FP16 to FP32 for CPU compatibility
- Ensures numerical stability on CPU

### 3. **Enhanced Video Processing**
- FFmpeg-based video handling with fallback support
- Automatic format conversion to MP4 (HEVC/H.264)
- Smart video trimming and frame extraction
- Progress indicators for all conversions

### 4. **Interactive CLI Interface**
- User-friendly terminal UI
- Preset management system
- Visual progress tracking

## 📋 Requirements

### System Requirements
- **macOS**: 10.15 (Catalina) or later
- **Python**: 3.8 - 3.11
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space for models and processing

### Software Dependencies
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg (required)
brew install ffmpeg

# Install Python via Homebrew (if needed)
brew install python@3.11
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Tencent/DepthCrafter.git
cd DepthCrafter
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for macOS)
pip install torch==2.0.1 torchvision==0.15.2

# Install other requirements
pip install diffusers==0.29.1
pip install transformers==4.41.2
pip install accelerate==0.30.1
pip install numpy==1.26.4
pip install matplotlib==3.8.4
pip install mediapy==1.2.0
pip install fire==0.6.0
pip install opencv-python==4.9.0.80
pip install gradio  # For web UI (optional)

# Optional: Install decord for better video processing
# Note: May require additional setup
# pip install decord
```

### 4. Download Model Weights
The models will be automatically downloaded from Hugging Face on first run:
- `tencent/DepthCrafter` - Main UNet model
- `stabilityai/stable-video-diffusion-img2vid-xt` - Base diffusion model

## 💻 Usage

### Option 1: Interactive CLI (Recommended)
```bash
# Launch the interactive interface
python interactive_cli.py

# Or use the launcher
./depthcrafter_ui
```

The interactive CLI provides:
- Step-by-step guided workflow
- Video preview and information
- Quality presets (Fast/Balanced/High)
- Frame range selection
- Preset save/load functionality

### Option 2: Command Line

#### Basic Usage
```bash
# Process entire video at 512px resolution
python run.py --video-path input.mp4 --max-res 512

# Process with custom settings
python run.py --video-path input.mp4 \
    --max-res 768 \
    --num-inference-steps 10 \
    --guidance-scale 1.2
```

#### Video Trimming
```bash
# Process first 50 frames only (faster for testing)
python run.py --video-path input.mp4 --max-frames 50 --max-res 512

# Process frames 100-200
python run.py --video-path input.mp4 --start-frame 100 --max-frames 100 --max-res 512

# Process specific time range (e.g., seconds 10-20)
python run.py --video-path input.mp4 --start-frame 300 --max-frames 300 --max-res 512
# (assuming 30fps: start at 10s = frame 300, 10s duration = 300 frames)
```

#### Output Options
```bash
# Save depth data in multiple formats
python run.py --video-path input.mp4 \
    --save-npz \           # Save as NPZ file
    --save-exr \           # Save as EXR sequence
    --save-folder output/  # Custom output directory
```

### Option 3: Web UI (Gradio)
```bash
python app.py
# Opens browser at http://localhost:7860
```

## ⚙️ Parameters

### Resolution Settings
- `--max-res`: Maximum resolution (512/768/1024)
  - 512: ~9GB RAM, fastest
  - 768: ~15GB RAM, balanced
  - 1024: ~26GB RAM, highest quality

### Quality Settings
- `--num-inference-steps`: Denoising steps (1-25, default: 5)
- `--guidance-scale`: Guidance strength (0.5-2.0, default: 1.0)

### Video Settings
- `--max-frames`: Limit number of frames to process
- `--start-frame`: Starting frame index
- `--target-fps`: Output video FPS (default: 15)
- `--process-length`: Alternative to max-frames

### Processing Settings
- `--window-size`: Sliding window size (default: 110)
- `--overlap`: Frame overlap between windows (default: 25)
- `--seed`: Random seed for reproducibility

## 📁 Output Files

The script generates the following outputs in the specified folder:

```
output_folder/
├── videoname_input.mp4   # Preprocessed/trimmed input
├── videoname_vis.mp4     # Colored depth visualization
├── videoname_depth.mp4   # Raw depth video
├── videoname.npz         # (Optional) Numpy depth data
└── frame_XXXX.exr        # (Optional) EXR depth frames
```

## 🎯 Performance Tips

### Memory Management
1. **Start with low resolution** (512px) for testing
2. **Use frame limits** to process shorter segments
3. **Close other applications** to free up RAM
4. **Monitor Activity Monitor** for memory usage

### Speed Optimization
```bash
# Fastest settings (lower quality)
python run.py --video-path input.mp4 \
    --max-res 512 \
    --num-inference-steps 3 \
    --max-frames 50

# Balanced settings
python run.py --video-path input.mp4 \
    --max-res 768 \
    --num-inference-steps 5 \
    --max-frames 100

# Best quality (slowest)
python run.py --video-path input.mp4 \
    --max-res 1024 \
    --num-inference-steps 10
```

### Processing Times (Approximate)
On M1 MacBook Pro (16GB RAM) for 150 frames:
- 512px: ~30 minutes
- 768px: ~60 minutes
- 1024px: ~120 minutes

*Note: Intel Macs will be slower. Apple Silicon (M1/M2/M3) provides better CPU performance.*

## 🎬 Supported Video Formats

### Input Formats
- MP4, MOV, AVI, MKV, WEBM, FLV, and most formats supported by FFmpeg
- Automatic conversion to MP4 for compatibility

### Automatic Optimizations
- Converts to HEVC/H.264 codec
- Adjusts to 15 FPS for consistency
- Maintains aspect ratio
- Removes audio tracks

## 🔧 Troubleshooting

### Common Issues

#### 1. "Torch not compiled with CUDA enabled"
This is expected on macOS. The code has been modified to use CPU instead.

#### 2. "Conv3D is not supported on MPS"
This is why we use CPU processing. MPS doesn't support 3D convolutions yet.

#### 3. Memory Errors
- Reduce `--max-res` to 512
- Process fewer frames with `--max-frames`
- Close other applications
- Consider upgrading RAM

#### 4. FFmpeg Errors
```bash
# Verify FFmpeg installation
ffmpeg -version

# Reinstall if needed
brew reinstall ffmpeg
```

#### 5. Slow Processing
This is normal for CPU processing. Tips:
- Use lower resolution (512px)
- Process shorter segments
- Run overnight for long videos
- Consider cloud GPU services for faster processing

### Check System Resources
```bash
# Monitor CPU and memory usage
top

# Check available disk space
df -h

# Check Python memory usage
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')"
```

## 🆕 Features Added for macOS

1. **Automatic Video Trimming**
   - Extract specific frame ranges before processing
   - Reduces memory usage and processing time

2. **Smart Format Conversion**
   - Automatic conversion to compatible MP4
   - Preserves quality while ensuring compatibility

3. **Progress Indicators**
   - Real-time conversion progress
   - Processing status updates

4. **Interactive CLI**
   - User-friendly interface
   - No need to remember commands
   - Visual feedback and validation

5. **Preset System**
   - Save frequently used settings
   - Share configurations with team

## 📊 Comparison with Original

| Feature | Original | macOS Version |
|---------|----------|---------------|
| GPU Support | CUDA | CPU only |
| MPS Support | No | No (Conv3D limitation) |
| Precision | FP16 | FP32 |
| Video Handling | Decord | FFmpeg + fallbacks |
| Trimming | Manual | Automatic |
| Interface | CLI only | CLI + Interactive UI |
| Presets | No | Yes |

## 🤝 Contributing

Contributions to improve macOS compatibility are welcome! Areas of interest:
- MPS optimization when Conv3D support is added
- Memory usage optimization
- Processing speed improvements
- Additional video format support

## 📝 License

This macOS version maintains the same license as the original DepthCrafter project.

## 🙏 Acknowledgments

- Original DepthCrafter team at Tencent AI Lab
- PyTorch team for CPU optimizations
- FFmpeg for robust video processing

## 📮 Support

For macOS-specific issues:
1. Check this README first
2. Search existing issues
3. Create a new issue with:
   - macOS version
   - Hardware (Intel/M1/M2/M3)
   - Python version
   - Error messages
   - Command used

---

**Note:** This is a CPU-based implementation optimized for macOS. For faster processing, consider using the original version on a CUDA-capable GPU or cloud services.