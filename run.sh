#!/bin/bash

# Ensure the script receives at least one argument
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <video_path1> [<video_path2> ... <video_pathN>]"
  exit 1
fi

# Define the Docker image name
IMAGE_NAME="depthcrafter"

# Ensure the Hugging Face cache directory exists
CACHE_DIR="$HOME/.cache/huggingface"
if [ ! -d "$CACHE_DIR" ]; then
  echo "Creating Hugging Face cache directory at $CACHE_DIR"
  mkdir -p "$CACHE_DIR"
fi

# Run the Docker container with GPU support and bind mounts for Hugging Face cache and video directory
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
   -v "$CACHE_DIR":/root/.cache/huggingface \
   -v $(pwd)/output:/workspace/DepthCrafter/output \
   -v $(pwd)/input:/workspace/DepthCrafter/input \
   -it --rm "$IMAGE_NAME" "$@"
