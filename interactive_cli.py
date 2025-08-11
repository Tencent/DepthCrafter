#!/usr/bin/env python3
"""
Interactive CLI for DepthCrafter
A user-friendly command-line interface for video depth estimation
"""

import os
import sys
import glob
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import shutil

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("═" * 60)
    print("     DepthCrafter Interactive CLI     ")
    print("     Generate Depth Maps from Videos     ")
    print("═" * 60)
    print(f"{Colors.ENDC}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.YELLOW}▶ {title}{Colors.ENDC}")
    print("-" * 40)

def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,codec_name',
            '-show_entries', 'format=duration,size',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            stream = info['streams'][0]
            format_info = info['format']
            
            # Parse frame rate
            fps_str = stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            return {
                'width': int(stream.get('width', 0)),
                'height': int(stream.get('height', 0)),
                'fps': fps,
                'frames': int(stream.get('nb_frames', 0)),
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'codec': stream.get('codec_name', 'unknown')
            }
    except:
        return None

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def select_video() -> Optional[str]:
    """Let user select a video file"""
    print_section("Select Video File")
    
    # Option 1: Recent files
    recent_videos = []
    for pattern in ['*.mp4', '*.webm', '*.avi', '*.mov', '*.mkv']:
        recent_videos.extend(glob.glob(pattern))
        recent_videos.extend(glob.glob(f"examples/{pattern}"))
    
    if recent_videos:
        print(f"{Colors.GREEN}Found videos:{Colors.ENDC}")
        for i, video in enumerate(recent_videos[:10], 1):
            print(f"  {i}. {video}")
        print(f"  {Colors.CYAN}0. Enter custom path{Colors.ENDC}")
        
        choice = input(f"\n{Colors.BLUE}Select video (1-{len(recent_videos[:10])} or 0): {Colors.ENDC}")
        
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(recent_videos[:10]):
                return recent_videos[idx - 1]
    
    # Option 2: Custom path
    video_path = input(f"{Colors.BLUE}Enter video path: {Colors.ENDC}").strip()
    
    if video_path.startswith('"') and video_path.endswith('"'):
        video_path = video_path[1:-1]
    
    if os.path.exists(video_path):
        return video_path
    else:
        print(f"{Colors.RED}Error: File not found!{Colors.ENDC}")
        return None

def display_video_info(video_path: str, info: Dict[str, Any]):
    """Display video information"""
    print_section("Video Information")
    print(f"  📹 File: {Colors.CYAN}{os.path.basename(video_path)}{Colors.ENDC}")
    print(f"  📐 Resolution: {info['width']}x{info['height']}")
    print(f"  🎬 Codec: {info['codec']}")
    print(f"  ⏱️  Duration: {format_duration(info['duration'])}")
    print(f"  🎞️  Frames: {info['frames']} @ {info['fps']:.1f} fps")
    print(f"  💾 Size: {format_size(info['size'])}")

def get_processing_options() -> Dict[str, Any]:
    """Get processing options from user"""
    print_section("Processing Options")
    
    # Presets
    print(f"\n{Colors.GREEN}Quality Presets:{Colors.ENDC}")
    print("  1. 🚀 Fast (512px, 5 steps) - ~2 min for 150 frames")
    print("  2. ⚖️  Balanced (768px, 5 steps) - ~4 min for 150 frames")
    print("  3. 🎯 High Quality (1024px, 10 steps) - ~8 min for 150 frames")
    print("  4. 🎨 Custom settings")
    
    preset = input(f"\n{Colors.BLUE}Select preset (1-4): {Colors.ENDC}").strip()
    
    options = {}
    
    if preset == '1':
        options['max_res'] = 512
        options['num_inference_steps'] = 5
        options['guidance_scale'] = 1.0
    elif preset == '2':
        options['max_res'] = 768
        options['num_inference_steps'] = 5
        options['guidance_scale'] = 1.0
    elif preset == '3':
        options['max_res'] = 1024
        options['num_inference_steps'] = 10
        options['guidance_scale'] = 1.2
    else:
        # Custom settings
        print(f"\n{Colors.YELLOW}Custom Settings:{Colors.ENDC}")
        
        max_res = input(f"  Max resolution ({Colors.CYAN}512/768/1024{Colors.ENDC}) [512]: ").strip()
        options['max_res'] = int(max_res) if max_res else 512
        
        steps = input(f"  Inference steps ({Colors.CYAN}1-25{Colors.ENDC}) [5]: ").strip()
        options['num_inference_steps'] = int(steps) if steps else 5
        
        guidance = input(f"  Guidance scale ({Colors.CYAN}0.5-2.0{Colors.ENDC}) [1.0]: ").strip()
        options['guidance_scale'] = float(guidance) if guidance else 1.0
    
    return options

def get_frame_range(video_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get frame range options from user"""
    print_section("Frame Range")
    
    total_frames = video_info['frames']
    fps = video_info['fps']
    
    print(f"Total frames: {total_frames} ({format_duration(total_frames/fps)})")
    print(f"\n{Colors.GREEN}Options:{Colors.ENDC}")
    print("  1. 🎬 Process entire video")
    print("  2. 🎞️  First N frames")
    print("  3. ✂️  Custom range")
    print("  4. ⏱️  Time-based selection")
    
    choice = input(f"\n{Colors.BLUE}Select option (1-4): {Colors.ENDC}").strip()
    
    if choice == '2':
        n = input(f"  Number of frames to process: ").strip()
        return {'max_frames': int(n), 'start_frame': 0}
    elif choice == '3':
        start = input(f"  Start frame (0-{total_frames}): ").strip()
        end = input(f"  End frame (0-{total_frames}): ").strip()
        start_frame = int(start) if start else 0
        end_frame = int(end) if end else total_frames
        return {'max_frames': end_frame - start_frame, 'start_frame': start_frame}
    elif choice == '4':
        start_time = input(f"  Start time (seconds): ").strip()
        duration = input(f"  Duration (seconds): ").strip()
        start_frame = int(float(start_time) * fps) if start_time else 0
        max_frames = int(float(duration) * fps) if duration else -1
        return {'max_frames': max_frames, 'start_frame': start_frame}
    else:
        return {'max_frames': -1, 'start_frame': 0}

def get_output_options() -> Dict[str, Any]:
    """Get output options from user"""
    print_section("Output Options")
    
    options = {}
    
    # Output folder
    default_folder = "./demo_output"
    folder = input(f"Output folder [{Colors.CYAN}{default_folder}{Colors.ENDC}]: ").strip()
    options['save_folder'] = folder if folder else default_folder
    
    # Output formats
    print(f"\n{Colors.GREEN}Additional outputs:{Colors.ENDC}")
    save_npz = input(f"  Save NPZ depth data (y/n) [n]: ").strip().lower()
    options['save_npz'] = save_npz == 'y'
    
    save_exr = input(f"  Save EXR depth files (y/n) [n]: ").strip().lower()
    options['save_exr'] = save_exr == 'y'
    
    # FPS
    target_fps = input(f"\nTarget FPS ({Colors.CYAN}-1 for auto{Colors.ENDC}) [15]: ").strip()
    options['target_fps'] = int(target_fps) if target_fps else 15
    
    return options

def build_command(video_path: str, options: Dict[str, Any]) -> str:
    """Build the command to run"""
    cmd = ["python", "run.py", "--video-path", video_path]
    
    # Add all options
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        else:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))
    
    return " ".join(cmd)

def run_processing(command: str) -> bool:
    """Run the processing command"""
    print_section("Processing")
    print(f"{Colors.CYAN}Command:{Colors.ENDC}")
    print(f"  {command}")
    print()
    
    confirm = input(f"{Colors.YELLOW}Start processing? (y/n): {Colors.ENDC}").strip().lower()
    
    if confirm != 'y':
        print(f"{Colors.RED}Cancelled.{Colors.ENDC}")
        return False
    
    print(f"\n{Colors.GREEN}Processing started...{Colors.ENDC}")
    print("=" * 60)
    
    try:
        # Run the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("=" * 60)
            print(f"{Colors.GREEN}✓ Processing completed successfully!{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}✗ Processing failed with error code {process.returncode}{Colors.ENDC}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Processing interrupted by user.{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        return False

def save_preset(options: Dict[str, Any]):
    """Save current settings as a preset"""
    print_section("Save Preset")
    
    name = input(f"Preset name: ").strip()
    if not name:
        return
    
    preset_file = f".depthcrafter_preset_{name}.json"
    
    with open(preset_file, 'w') as f:
        json.dump(options, f, indent=2)
    
    print(f"{Colors.GREEN}✓ Preset saved to {preset_file}{Colors.ENDC}")

def load_preset() -> Optional[Dict[str, Any]]:
    """Load a saved preset"""
    presets = glob.glob(".depthcrafter_preset_*.json")
    
    if not presets:
        print(f"{Colors.YELLOW}No saved presets found.{Colors.ENDC}")
        return None
    
    print_section("Load Preset")
    for i, preset in enumerate(presets, 1):
        name = preset.replace(".depthcrafter_preset_", "").replace(".json", "")
        print(f"  {i}. {name}")
    
    choice = input(f"\n{Colors.BLUE}Select preset (1-{len(presets)}): {Colors.ENDC}").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(presets):
            with open(presets[idx], 'r') as f:
                return json.load(f)
    
    return None

def main():
    """Main interactive CLI loop"""
    print_header()
    print(f"{Colors.GREEN}Welcome to DepthCrafter Interactive CLI!{Colors.ENDC}")
    print("This tool will help you create depth maps from your videos.\n")
    
    while True:
        print(f"\n{Colors.CYAN}Main Menu:{Colors.ENDC}")
        print("  1. 🎥 Process a video")
        print("  2. 📁 Load preset")
        print("  3. 📚 View examples")
        print("  4. ❓ Help")
        print("  5. 🚪 Exit")
        
        choice = input(f"\n{Colors.BLUE}Select option (1-5): {Colors.ENDC}").strip()
        
        if choice == '1':
            # Process video workflow
            video_path = select_video()
            if not video_path:
                continue
            
            # Get video info
            info = get_video_info(video_path)
            if info:
                display_video_info(video_path, info)
            else:
                print(f"{Colors.YELLOW}Warning: Could not read video information{Colors.ENDC}")
                info = {'frames': -1, 'fps': 30}
            
            # Get options
            options = {}
            
            # Check if user wants to load preset
            if input(f"\n{Colors.BLUE}Load preset? (y/n): {Colors.ENDC}").strip().lower() == 'y':
                preset = load_preset()
                if preset:
                    options.update(preset)
            
            if not options:
                options.update(get_processing_options())
                options.update(get_frame_range(info))
                options.update(get_output_options())
                
                # Offer to save as preset
                if input(f"\n{Colors.BLUE}Save these settings as preset? (y/n): {Colors.ENDC}").strip().lower() == 'y':
                    save_preset(options)
            
            # Build and run command
            command = build_command(video_path, options)
            success = run_processing(command)
            
            if success:
                output_folder = options.get('save_folder', './demo_output')
                print(f"\n{Colors.GREEN}Output files saved to: {output_folder}{Colors.ENDC}")
            
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            print_header()
            
        elif choice == '2':
            # Load preset
            preset = load_preset()
            if preset:
                print(f"{Colors.GREEN}Preset loaded successfully!{Colors.ENDC}")
                print(json.dumps(preset, indent=2))
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            print_header()
            
        elif choice == '3':
            # View examples
            print_section("Example Commands")
            print(f"{Colors.GREEN}Quick test (50 frames):{Colors.ENDC}")
            print("  python run.py --video-path video.mp4 --max-frames 50 --max-res 512")
            print(f"\n{Colors.GREEN}High quality:{Colors.ENDC}")
            print("  python run.py --video-path video.mp4 --max-res 1024 --num-inference-steps 10")
            print(f"\n{Colors.GREEN}Custom range:{Colors.ENDC}")
            print("  python run.py --video-path video.mp4 --start-frame 100 --max-frames 200")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            print_header()
            
        elif choice == '4':
            # Help
            print_section("Help")
            print("DepthCrafter generates depth maps from videos using AI.")
            print("\n📋 Requirements:")
            print("  • Python 3.8+")
            print("  • PyTorch 2.0+")
            print("  • FFmpeg")
            print("  • ~8GB GPU memory (512px) or ~26GB (1024px)")
            print("\n🎯 Tips:")
            print("  • Start with low resolution (512px) for testing")
            print("  • Use frame limits to test on short segments")
            print("  • Save presets for frequently used settings")
            print("  • Check output folder for _vis.mp4 (visualization)")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            print_header()
            
        elif choice == '5':
            print(f"\n{Colors.GREEN}Thank you for using DepthCrafter!{Colors.ENDC}")
            break
        else:
            print(f"{Colors.RED}Invalid option. Please try again.{Colors.ENDC}")

if __name__ == "__main__":
    try:
        # Check if ffmpeg is available
        if shutil.which('ffmpeg') is None:
            print(f"{Colors.RED}Error: FFmpeg is not installed or not in PATH{Colors.ENDC}")
            print("Please install FFmpeg first: https://ffmpeg.org/download.html")
            sys.exit(1)
        
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Exiting...{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.ENDC}")
        sys.exit(1)