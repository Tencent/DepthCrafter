#!/bin/bash

# Activate the virtual environment
source /opt/venv/bin/activate

# Run the Python script with the passed video path(s)
python run.py --video-path "input/$@"
