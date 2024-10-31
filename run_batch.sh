#!/bin/bash

# Define the Docker image name
IMAGE_NAME="depthcrafter"

# Define the batch directory path
BATCH_DIR="$(pwd)/batch"

# Check if the batch directory exists
if [ ! -d "$BATCH_DIR" ]; then
  echo "Batch directory does not exist. Please create a 'batch' directory and add video files."
  exit 1
fi

# Ensure the Hugging Face cache directory exists
CACHE_DIR="$HOME/.cache/huggingface"
if [ ! -d "$CACHE_DIR" ]; then
  echo "Creating Hugging Face cache directory at $CACHE_DIR"
  mkdir -p "$CACHE_DIR"
fi

# Collect all video file paths in the batch directory, prefix with "input/", and create a comma-separated list
VIDEO_PATHS=$(find "$BATCH_DIR" -type f ! -name ".gitignore" -exec basename {} \; | sed 's|^|input/|' | tr '\n' ',' | sed 's/,$//')

# Check if any video files are found
if [ -z "$VIDEO_PATHS" ]; then
  echo "No video files found in the batch directory."
  exit 1
fi

# Run the Docker container with GPU support, bind mounts, and video files as arguments
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
   -v "$CACHE_DIR":/root/.cache/huggingface \
   -v $(pwd)/output:/workspace/DepthCrafter/output \
   -v "$BATCH_DIR":/workspace/DepthCrafter/input \
   -it --rm "$IMAGE_NAME" --video-path "$VIDEO_PATHS"
