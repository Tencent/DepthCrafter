from typing import Union, List
import tempfile
import numpy as np
import PIL.Image
import matplotlib.cm as cm
import mediapy
import torch
import subprocess
import json
import os
import sys
import re
import warnings
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

dataset_res_dict = {
    "sintel": [448, 1024],
    "scannet": [640, 832],
    "KITTI": [384, 1280],
    "bonn": [512, 640],
    "NYUv2": [448, 640],
}


def get_video_info_ffmpeg(video_path):
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        
        # Parse frame rate
        fps_str = stream['r_frame_rate']
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        return {
            'width': int(stream['width']),
            'height': int(stream['height']),
            'fps': fps,
            'nb_frames': int(stream.get('nb_frames', 0))
        }
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        # Fallback: get basic info without frame count
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        
        fps_str = stream['r_frame_rate']
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        # Estimate frame count
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration_info = json.loads(duration_result.stdout)
        duration = float(duration_info['format']['duration'])
        
        return {
            'width': int(stream['width']),
            'height': int(stream['height']),
            'fps': fps,
            'nb_frames': int(duration * fps)
        }


def read_video_frames_ffmpeg(video_path, process_length, target_fps, max_res, dataset="open"):
    """Read video frames using ffmpeg."""
    # Convert to absolute path
    video_path = os.path.abspath(video_path)
    
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")
    
    print("==> processing video directly with ffmpeg: ", video_path)
    
    # Get video info
    video_info = get_video_info_ffmpeg(video_path)
    original_width = video_info['width']
    original_height = video_info['height']
    original_fps = video_info['fps']
    total_frames = video_info['nb_frames']
    
    print(f"==> original video shape: ({total_frames}, {original_height}, {original_width}, 3)")
    
    if dataset == "open":
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]
    
    fps = original_fps if target_fps == -1 else target_fps
    stride = round(original_fps / fps)
    stride = max(stride, 1)
    
    # Calculate which frames to extract
    frames_idx = list(range(0, total_frames, stride))
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    
    print(f"==> downsampled shape: ({len(frames_idx)}, {height}, {width}, 3), with stride: {stride}")
    print(f"==> final processing shape: ({len(frames_idx)}, {height}, {width}, 3)")
    
    # Build ffmpeg command to extract frames
    # Simplified approach: extract at target fps and scale
    vf_filters = []
    
    # Add fps filter to get the right frame rate
    if stride > 1:
        vf_filters.append(f"fps={fps}")
    
    # Add scaling
    vf_filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
    vf_filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
    
    # Limit frames if needed
    if process_length != -1:
        vf_filters.append(f"select='lt(n\,{process_length})'")
    
    vf_string = ','.join(vf_filters)
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', vf_string,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-v', 'error',
        '-'
    ]
    
    # Run ffmpeg and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
    
    # Convert raw RGB data to numpy array
    frames = np.frombuffer(stdout, dtype=np.uint8)
    frames = frames.reshape((-1, height, width, 3))
    frames = frames.astype(np.float32) / 255.0
    
    # Ensure we have the expected number of frames
    if frames.shape[0] != len(frames_idx):
        print(f"Warning: Expected {len(frames_idx)} frames, got {frames.shape[0]}")
    
    return frames, fps


def get_video_duration(input_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        input_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except:
        return None

def show_progress(current_time, total_duration, width=50):
    """Display a progress bar for video conversion."""
    if total_duration is None or total_duration == 0:
        return
    
    progress = min(current_time / total_duration, 1.0)
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percent = progress * 100
    
    # Clear the line and print progress
    sys.stdout.write(f'\rConverting: [{bar}] {percent:.1f}% ({current_time:.1f}s/{total_duration:.1f}s)')
    sys.stdout.flush()

def convert_to_mp4(input_path, output_path=None, target_fps=15, max_frames=None, start_frame=0):
    """Convert video to MP4 format matching the example videos' settings.
    
    Args:
        input_path: Path to input video
        output_path: Path to output MP4 (if None, creates temp file)
        target_fps: Target frame rate (default: 15)
        max_frames: Maximum number of frames to extract (if None, extract all)
        start_frame: Starting frame number (default: 0)
    """
    # Convert to absolute path to avoid path issues
    input_path = os.path.abspath(input_path)
    
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input video file not found: {input_path}")
    
    if output_path is None:
        # Create a temporary MP4 file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_file.name
        temp_file.close()
    else:
        output_path = os.path.abspath(output_path)
    
    # Check if input is already MP4 with correct codec
    if input_path.lower().endswith('.mp4'):
        # Check if it's actually a valid MP4 that can be read
        try:
            # Quick probe to see if it's readable and has correct codec
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=codec_name', '-of', 'json', input_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            codec = info['streams'][0]['codec_name']
            # If it's already HEVC or H264, and readable, return original
            if codec in ['hevc', 'h264']:
                return input_path
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
            # If not readable or wrong codec, proceed with conversion
            pass
    
    # Determine if we're trimming the video
    trimming = max_frames is not None or start_frame > 0
    
    if trimming:
        print(f"Trimming and converting video to MP4: {os.path.basename(input_path)}")
        print(f"  Extracting frames {start_frame} to {start_frame + (max_frames or 'end')}")
    else:
        print(f"Converting video to MP4 format: {os.path.basename(input_path)}")
    
    print(f"  Input path: {input_path}")
    print(f"  Output path: {output_path}")
    print(f"  File exists: {os.path.exists(input_path)}")
    print(f"  File size: {os.path.getsize(input_path) / (1024*1024):.1f} MB" if os.path.exists(input_path) else "")
    
    # Get video info for progress tracking and trimming
    video_info = get_video_info_ffmpeg(input_path)
    original_fps = video_info.get('fps', 30)
    duration = get_video_duration(input_path)
    
    # Calculate time ranges if trimming
    if trimming:
        start_time = start_frame / original_fps if start_frame > 0 else 0
        if max_frames:
            # IMPORTANT: Use original_fps to calculate duration, not target_fps
            # We want to extract max_frames from the original video
            duration_time = max_frames / original_fps
            # Adjust duration for progress bar
            duration = min(duration_time, duration - start_time if duration else duration_time)
        else:
            duration_time = None
    
    def run_ffmpeg_with_progress(cmd, codec_name):
        """Run ffmpeg command with progress tracking."""
        # Add progress output to the command
        # Need to insert -progress and -stats before the input file (-i)
        try:
            i_index = cmd.index('-i')
            cmd_with_progress = cmd[:i_index] + ['-progress', 'pipe:1', '-stats'] + cmd[i_index:]
        except ValueError:
            # If -i not found, add at position 2 (after ffmpeg)
            cmd_with_progress = cmd[:1] + ['-progress', 'pipe:1', '-stats'] + cmd[1:]
        
        process = subprocess.Popen(
            cmd_with_progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Pattern to match time from ffmpeg progress output
        time_pattern = re.compile(r'out_time_ms=(\d+)')
        stderr_lines = []
        
        # Read stderr in background
        import threading
        def read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)
        
        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()
        
        for line in process.stdout:
            match = time_pattern.search(line)
            if match:
                current_time_ms = int(match.group(1))
                current_time = current_time_ms / 1_000_000  # Convert microseconds to seconds
                show_progress(current_time, duration)
        
        # Wait for process to complete
        process.wait()
        stderr_thread.join(timeout=1)
        
        if process.returncode == 0:
            print(f"\n✓ Video successfully converted to MP4 ({codec_name})")
            return True
        else:
            stderr = ''.join(stderr_lines)
            print(f"\n✗ {codec_name} conversion failed")
            # Print relevant error messages
            if 'Unknown encoder' in stderr or 'not found' in stderr:
                print(f"  Error: {codec_name} encoder not available in ffmpeg")
            elif 'Invalid' in stderr or 'Error' in stderr:
                # Extract error lines
                error_lines = [line.strip() for line in stderr_lines if 'Error' in line or 'Invalid' in line]
                if error_lines:
                    print(f"  Error details: {error_lines[0]}")
            return False
    
    # Build base command
    def build_command(codec, codec_lib, preset='medium', crf='23', use_target_fps=True):
        cmd = ['ffmpeg']
        
        # Add trimming options BEFORE input (for fast seek)
        if trimming:
            if start_frame > 0:
                # Use format HH:MM:SS.mmm for better compatibility
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = start_time % 60
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                cmd.extend(['-ss', time_str])
            if max_frames:
                # Duration also in time format
                hours = int(duration_time // 3600)
                minutes = int((duration_time % 3600) // 60)
                seconds = duration_time % 60
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                cmd.extend(['-t', duration_str])
        
        cmd.extend(['-i', input_path])
        
        # Video encoding options
        cmd.extend([
            '-c:v', codec_lib,
            '-preset', preset,
            '-crf', crf,
            '-pix_fmt', 'yuv420p',
        ])
        
        # Only set output frame rate if requested and different from input
        # This prevents frame duplication/interpolation
        if use_target_fps and target_fps != -1:
            cmd.extend(['-r', str(target_fps)])
        
        if codec == 'hevc':
            cmd.extend(['-tag:v', 'hev1'])
        
        # Add metadata for trimmed videos
        if trimming:
            cmd.extend([
                '-metadata', f'title=Trimmed from {os.path.basename(input_path)}',
                '-metadata', f'comment=Frames {start_frame}-{start_frame + (max_frames or "end")} at {target_fps}fps',
            ])
        
        cmd.extend([
            '-an',  # No audio
            '-movflags', '+faststart',
            '-y',
            output_path
        ])
        
        return cmd
    
    # Build ffmpeg command for conversion matching example videos
    # First try with HEVC (H.265) like the examples
    # Don't change fps when trimming to preserve frame count
    use_target_fps = not trimming or target_fps == -1
    cmd_hevc = build_command('hevc', 'libx265', use_target_fps=use_target_fps)
    
    # Try HEVC first
    if run_ffmpeg_with_progress(cmd_hevc, 'HEVC'):
        return output_path
    
    print("Falling back to H.264...")
    
    # Fallback to H.264 if HEVC fails (better compatibility)
    cmd_h264 = build_command('h264', 'libx264', use_target_fps=use_target_fps)
    
    if run_ffmpeg_with_progress(cmd_h264, 'H.264'):
        return output_path
    
    # If both conversions failed, try a more basic conversion
    print("\nTrying basic MP4 conversion with default settings...")
    
    cmd_basic = build_command('h264', 'libx264', preset='fast', crf='28', use_target_fps=use_target_fps)
    
    if run_ffmpeg_with_progress(cmd_basic, 'H.264 (basic)'):
        return output_path
    
    # Last resort: try with minimal options
    print("\nTrying minimal conversion...")
    cmd_minimal = ['ffmpeg']
    if trimming:
        if start_frame > 0:
            # Use time format for compatibility
            hours = int(start_time // 3600)
            minutes = int((start_time % 3600) // 60)
            seconds = start_time % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            cmd_minimal.extend(['-ss', time_str])
        if max_frames:
            # Duration in time format
            hours = int(duration_time // 3600)
            minutes = int((duration_time % 3600) // 60)
            seconds = duration_time % 60
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            cmd_minimal.extend(['-t', duration_str])
    cmd_minimal.extend([
        '-i', input_path,
        '-c:v', 'libx264',
        '-an',
        '-y',
        output_path
    ])
    
    process = subprocess.run(cmd_minimal, capture_output=True, text=True)
    if process.returncode == 0:
        print("✓ Video converted with minimal settings")
        return output_path
    else:
        print(f"✗ Minimal conversion also failed")
        print(f"Error: {process.stderr[:500]}...")
        raise RuntimeError(f"Failed to convert video to MP4. Please check if ffmpeg is properly installed and the input video is valid.")

def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open", skip_conversion=False, start_frame=0):
    """Read video frames with MP4 conversion and fallback to ffmpeg if decord fails.
    
    Args:
        video_path: Path to input video
        process_length: Number of frames to process (-1 for all)
        target_fps: Target frame rate (-1 to keep original)
        max_res: Maximum resolution
        dataset: Dataset type for resolution presets
        skip_conversion: Skip MP4 conversion if True
        start_frame: Starting frame for trimming (default: 0)
    """
    
    # Convert to absolute path
    video_path = os.path.abspath(video_path)
    
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")
    
    # Convert to MP4 first if needed
    converted_path = None
    original_path = video_path
    
    # Convert non-MP4 files or problematic files to MP4 (unless skip_conversion is True)
    # Use 15 fps by default (matching example videos) unless specified
    default_fps = 15 if target_fps == -1 else target_fps
    
    # Determine if we need to trim or convert
    needs_conversion = not video_path.lower().endswith('.mp4')
    needs_trimming = process_length > 0 and process_length != -1
    
    if not skip_conversion and (needs_conversion or needs_trimming):
        try:
            # If trimming is needed, always convert (even MP4s) to create a trimmed version
            if needs_trimming:
                print(f"\nCreating trimmed video: {process_length} frames starting from frame {start_frame}")
                # For trimming, use -1 for target_fps to keep original fps
                # This prevents frame count changes
                converted_path = convert_to_mp4(
                    video_path, 
                    target_fps=-1,  # Keep original fps to maintain frame count
                    max_frames=process_length,
                    start_frame=start_frame
                )
                video_path = converted_path
                # After trimming, we don't need to limit frames again
                process_length = -1
            elif needs_conversion:
                converted_path = convert_to_mp4(video_path, target_fps=default_fps)
                video_path = converted_path
        except RuntimeError as e:
            print(f"\nWarning: MP4 conversion/trimming failed: {e}")
            print("Attempting to process the original video directly...")
            video_path = original_path
    
    # Try using decord first if available
    if DECORD_AVAILABLE:
        try:
            if dataset == "open":
                print("==> processing video: ", video_path)
                vid = VideoReader(video_path, ctx=cpu(0))
                print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
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
            print(
                f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
            )
            if process_length != -1 and process_length < len(frames_idx):
                frames_idx = frames_idx[:process_length]
            print(
                f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
            )
            frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
            
            # Clean up temporary file if created
            if converted_path and converted_path != original_path:
                try:
                    os.remove(converted_path)
                except:
                    pass
            
            return frames, fps
            
        except Exception as e:
            print(f"Decord failed to read video: {e}")
            # If decord fails on MP4, try converting again with different settings
            if not skip_conversion and video_path == original_path:  # Only convert if we haven't already
                print("Attempting to convert video to MP4...")
                default_fps = 15 if target_fps == -1 else target_fps
                try:
                    # Try conversion with trimming if needed
                    if process_length > 0 and process_length != -1:
                        converted_path = convert_to_mp4(
                            video_path, 
                            target_fps=default_fps,
                            max_frames=process_length,
                            start_frame=start_frame
                        )
                        process_length = -1  # Reset since video is now trimmed
                    else:
                        converted_path = convert_to_mp4(video_path, target_fps=default_fps)
                    video_path = converted_path
                except RuntimeError:
                    print("Conversion failed, using original video")
                    video_path = original_path
            print("Falling back to ffmpeg direct processing...")
    
    # Fallback to ffmpeg
    try:
        frames, fps = read_video_frames_ffmpeg(video_path, process_length, target_fps, max_res, dataset)
    finally:
        # Clean up temporary file if created
        if converted_path and converted_path != original_path:
            try:
                os.remove(converted_path)
            except:
                pass
    
    return frames, fps


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    crf: int = 18,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path


class ColorMapper:
    # a color mapper to map depth values to a certain colormap
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
