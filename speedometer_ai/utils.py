"""
Utility functions for video processing and data handling
"""

import cv2
import csv
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


def extract_frames_from_video(video_path: Path, 
                            output_dir: Path, 
                            fps: float = 3.0) -> List[Path]:
    """
    Extract frames from video using FFmpeg with timestamp-based filenames
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Frames per second to extract
        
    Returns:
        List of extracted frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command - extract frames numbered sequentially first
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-y',  # Overwrite output files
        str(output_dir / 'temp_frame_%06d.png')
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Get list of created frames and rename with timestamps
        temp_frames = sorted(output_dir.glob('temp_frame_*.png'))
        frame_files = []
        
        for i, temp_frame in enumerate(temp_frames):
            timestamp = i / fps
            new_name = f"frame_t{timestamp:06.2f}s.png"
            new_path = output_dir / new_name
            temp_frame.rename(new_path)
            frame_files.append(new_path)
        
        return frame_files
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")


def save_results_to_csv(results: List[Dict], output_path: Path, cost_info: Dict = None) -> None:
    """
    Save analysis results to CSV
    
    Args:
        results: List of analysis results
        output_path: Path to save CSV file
        cost_info: Optional cost information to include in metadata
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if not results:
            return
        
        # Add cost info as comments at the top if provided
        if cost_info:
            f.write(f"# Analysis Cost Summary\n")
            f.write(f"# Model: {cost_info['model']}\n")
            f.write(f"# API Calls: {cost_info['api_calls']}\n")
            f.write(f"# Total Cost: ${cost_info['total_cost_usd']:.6f} USD\n")
            f.write(f"# Input Tokens: {cost_info['input_tokens']}\n")
            f.write(f"# Output Tokens: {cost_info['output_tokens']}\n")
            f.write(f"#\n")
            
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'timestamp', 'speed', 'filename', 'success', 'interpolated', 'smoothed', 'anomaly_corrected', 'response'
        ])
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'frame': result['frame'],
                'timestamp': round(result['timestamp'], 2),
                'speed': result['speed'] if result['speed'] is not None else '',
                'filename': result['filename'],
                'success': result['success'],
                'interpolated': result.get('interpolated', False),
                'smoothed': result.get('smoothed', False),
                'anomaly_corrected': result.get('anomaly_corrected', False),
                'response': result['response']
            })


def load_results_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load results from CSV file"""
    return pd.read_csv(csv_path)


def create_speed_chart(results: List[Dict], output_path: Path) -> None:
    """
    Create speed progression chart
    
    Args:
        results: Analysis results
        output_path: Path to save chart
    """
    try:
        import matplotlib.pyplot as plt
        
        # Filter successful results
        successful = [r for r in results if r['success']]
        
        if not successful:
            print("No successful readings to plot")
            return
        
        timestamps = [r['timestamp'] for r in successful]
        speeds = [r['speed'] for r in successful]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, speeds, 'b-o', linewidth=2, markersize=4)
        plt.title('Speed Progression Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (km/h)')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        max_idx = speeds.index(max(speeds))
        min_idx = speeds.index(min(speeds))
        
        plt.annotate(f'Max: {max(speeds)} km/h', 
                    xy=(timestamps[max_idx], speeds[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.annotate(f'Min: {min(speeds)} km/h', 
                    xy=(timestamps[min_idx], speeds[min_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for chart generation")


def print_analysis_summary(results: List[Dict], stats: Dict) -> None:
    """Print formatted analysis summary"""
    print("\n" + "="*60)
    print("SPEEDOMETER ANALYSIS RESULTS")
    print("="*60)
    
    print(f"📊 SUMMARY:")
    print(f"   • Total frames: {stats.get('total_frames', 0)}")
    print(f"   • Successful readings: {stats.get('successful_readings', 0)}")
    print(f"   • Success rate: {stats.get('success_rate', 0):.1f}%")
    print(f"   • Duration: {stats.get('duration', 0):.1f} seconds")
    
    if 'min_speed' in stats:
        print(f"   • Speed range: {stats['min_speed']}-{stats['max_speed']} km/h")
        print(f"   • Average speed: {stats['avg_speed']:.1f} km/h")
    
    print(f"\n📈 DETAILED TIMELINE:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        speed_str = f"{result['speed']:3d}" if result['speed'] else "N/A"
        print(f"   {status} {result['timestamp']:5.1f}s: {speed_str} km/h")


def print_cost_summary(cost_info: Dict) -> None:
    """Print formatted cost summary"""
    print(f"\n💰 COST BREAKDOWN:")
    print(f"   • Model: {cost_info['model']}")
    print(f"   • API calls: {cost_info['api_calls']}")
    print(f"   • Input tokens: {cost_info['input_tokens']:,}")
    print(f"   • Output tokens: {cost_info['output_tokens']:,}")
    print(f"   • Input cost: ${cost_info['input_cost_usd']:.6f} USD")
    print(f"   • Output cost: ${cost_info['output_cost_usd']:.6f} USD")
    print(f"   • Total cost: ${cost_info['total_cost_usd']:.6f} USD")
    
    if cost_info['total_cost_usd'] < 0.01:
        print(f"   💡 Very affordable! Less than 1 cent.")
    elif cost_info['total_cost_usd'] < 0.10:
        print(f"   💡 Very reasonable cost, less than 10 cents.")
    else:
        print(f"   ⚠️  Consider optimizing frames/fps for cost reduction.")


def validate_video_file(video_path: Path) -> bool:
    """Validate that video file exists and is readable"""
    if not video_path.exists():
        return False
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path))
        ret = cap.isOpened()
        cap.release()
        return ret
    except:
        return False


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def interpolate_missing_speeds(results: List[Dict], max_gap_size: int = 3) -> List[Dict]:
    """
    Fill in missing speed readings using intelligent interpolation
    
    Args:
        results: List of analysis results with potential gaps
        max_gap_size: Maximum consecutive missing values to interpolate
        
    Returns:
        Results with interpolated speed values
    """
    if not results:
        return results
    
    # Create a copy to avoid modifying the original
    interpolated_results = [result.copy() for result in results]
    
    # Initialize interpolated flag for all results
    for result in interpolated_results:
        if 'interpolated' not in result:
            result['interpolated'] = False
    
    # Extract speeds and timestamps for successful readings
    speeds = []
    timestamps = []
    indices = []
    
    for i, result in enumerate(interpolated_results):
        if result['success'] and result['speed'] is not None:
            speeds.append(result['speed'])
            timestamps.append(result['timestamp'])
            indices.append(i)
    
    if len(speeds) < 2:
        return interpolated_results  # Need at least 2 points to interpolate
    
    # Find gaps and interpolate (NEVER interpolate the first frame)
    for i, result in enumerate(interpolated_results):
        # Skip the first frame - it must always be the original AI reading
        if i == 0:
            continue
            
        if not result['success'] or result['speed'] is None:
            # Find the gap boundaries
            gap_start, gap_end = _find_gap_boundaries(interpolated_results, i, max_gap_size)
            
            if gap_start is not None and gap_end is not None:
                # Interpolate speeds for this gap
                gap_length = gap_end - gap_start - 1
                if gap_length <= max_gap_size:
                    start_speed = interpolated_results[gap_start]['speed']
                    end_speed = interpolated_results[gap_end]['speed']
                    start_time = interpolated_results[gap_start]['timestamp']
                    end_time = interpolated_results[gap_end]['timestamp']
                    
                    # Linear interpolation with realistic speed constraints
                    for j in range(gap_start + 1, gap_end):
                        progress = (interpolated_results[j]['timestamp'] - start_time) / (end_time - start_time)
                        interpolated_speed = start_speed + (end_speed - start_speed) * progress
                        
                        # Apply realistic constraints (no sudden jumps > 50 km/h)
                        max_change = 50 * (interpolated_results[j]['timestamp'] - interpolated_results[j-1]['timestamp'])
                        prev_speed = interpolated_results[j-1]['speed'] or start_speed
                        
                        if abs(interpolated_speed - prev_speed) > max_change:
                            if interpolated_speed > prev_speed:
                                interpolated_speed = prev_speed + max_change
                            else:
                                interpolated_speed = prev_speed - max_change
                        
                        # Ensure reasonable speed bounds
                        interpolated_speed = max(0, min(300, interpolated_speed))
                        
                        interpolated_results[j]['speed'] = round(interpolated_speed)
                        interpolated_results[j]['success'] = True
                        interpolated_results[j]['interpolated'] = True
                        original_response = interpolated_results[j]['response']
                        interpolated_results[j]['response'] = f"INTERPOLATED: AI could not read, estimated {round(interpolated_speed)} km/h based on surrounding values | Original: {original_response}"
    
    return interpolated_results


def _find_gap_boundaries(results: List[Dict], gap_index: int, max_gap_size: int) -> tuple:
    """
    Find the valid speed readings before and after a gap
    
    Returns:
        Tuple of (start_index, end_index) or (None, None) if gap is too large
    """
    # Find valid reading before the gap
    start_index = None
    for i in range(gap_index - 1, -1, -1):
        if results[i]['success'] and results[i]['speed'] is not None:
            start_index = i
            break
    
    # Find valid reading after the gap
    end_index = None
    gap_size = 0
    for i in range(gap_index, len(results)):
        if results[i]['success'] and results[i]['speed'] is not None:
            end_index = i
            break
        gap_size += 1
        if gap_size > max_gap_size:
            break
    
    # Only return boundaries if gap is within acceptable size
    if (start_index is not None and end_index is not None and 
        end_index - start_index - 1 <= max_gap_size):
        return start_index, end_index
    
    return None, None


def detect_and_correct_anomalies(results: List[Dict], max_change_per_second: float = 40.0) -> List[Dict]:
    """
    Detect and correct anomalous speed readings (e.g., OCR misreads 90 as 60)
    
    Args:
        results: List of analysis results
        max_change_per_second: Maximum realistic speed change per second (km/h/s)
        
    Returns:
        Results with corrected anomalous readings
    """
    if not results or len(results) < 3:
        return results
    
    corrected_results = [result.copy() for result in results]
    
    # Initialize anomaly_corrected flag
    for result in corrected_results:
        if 'anomaly_corrected' not in result:
            result['anomaly_corrected'] = False
    
    # Process each frame to detect anomalies (NEVER correct the first frame)
    # Start from index 1 and explicitly exclude first frame from any corrections
    for i in range(1, len(corrected_results) - 1):
        current = corrected_results[i]
        prev_frame = corrected_results[i - 1]
        next_frame = corrected_results[i + 1]
        
        # Only check successful readings
        if not (current['success'] and prev_frame['success'] and next_frame['success']):
            continue
            
        if current['speed'] is None or prev_frame['speed'] is None or next_frame['speed'] is None:
            continue
        
        # Calculate time differences
        dt_prev = current['timestamp'] - prev_frame['timestamp']
        dt_next = next_frame['timestamp'] - current['timestamp']
        
        if dt_prev <= 0 or dt_next <= 0:
            continue
        
        # Calculate speed changes
        change_from_prev = abs(current['speed'] - prev_frame['speed'])
        change_to_next = abs(next_frame['speed'] - current['speed'])
        
        # Calculate maximum allowed changes based on time intervals
        max_change_prev = max_change_per_second * dt_prev
        max_change_next = max_change_per_second * dt_next
        
        # Detect anomaly: current reading is inconsistent with both neighbors
        if (change_from_prev > max_change_prev and change_to_next > max_change_next):
            
            # Check if this looks like a digit misread (common OCR errors)
            corrected_speed = _detect_digit_misread(current['speed'], prev_frame['speed'], next_frame['speed'])
            
            if corrected_speed is not None:
                # Apply correction
                original_speed = current['speed']
                corrected_results[i]['speed'] = corrected_speed
                corrected_results[i]['anomaly_corrected'] = True
                original_response = current['response']
                corrected_results[i]['response'] = f"ANOMALY CORRECTED: AI read {original_speed} km/h, likely misread digit, corrected to {corrected_speed} km/h based on surrounding data | Original: {original_response}"
                
            else:
                # If we can't correct it, interpolate between neighbors
                interpolated_speed = round((prev_frame['speed'] + next_frame['speed']) / 2)
                
                # Only apply if the interpolated value is significantly different
                if abs(interpolated_speed - current['speed']) > 20:
                    original_speed = current['speed']
                    corrected_results[i]['speed'] = interpolated_speed
                    corrected_results[i]['anomaly_corrected'] = True
                    original_response = current['response']
                    corrected_results[i]['response'] = f"ANOMALY CORRECTED: AI read {original_speed} km/h, unrealistic change, interpolated to {interpolated_speed} km/h | Original: {original_response}"
    
    return corrected_results


def _detect_digit_misread(current_speed: int, prev_speed: int, next_speed: int) -> Optional[int]:
    """
    Detect common OCR digit misreads and suggest corrections
    
    Common misreads:
    - 0 ↔ 6, 8, 9
    - 1 ↔ 7
    - 2 ↔ 5, 8
    - 3 ↔ 8
    - 5 ↔ 6, 8
    - 6 ↔ 8, 9
    - 8 ↔ 9
    """
    # Common digit confusion pairs
    digit_confusions = {
        '0': ['6', '8', '9'],
        '1': ['7'],
        '2': ['5', '8'],
        '3': ['8'],
        '5': ['2', '6', '8'],
        '6': ['0', '5', '8', '9'],
        '7': ['1'],
        '8': ['0', '2', '3', '5', '6', '9'],
        '9': ['0', '6', '8']
    }
    
    current_str = str(current_speed)
    expected_speed = round((prev_speed + next_speed) / 2)
    expected_str = str(expected_speed)
    
    # Try correcting each digit
    for pos in range(min(len(current_str), len(expected_str))):
        if pos < len(current_str) and pos < len(expected_str):
            current_digit = current_str[pos]
            expected_digit = expected_str[pos]
            
            # Check if this could be a digit confusion
            if (current_digit in digit_confusions and 
                expected_digit in digit_confusions[current_digit]):
                
                # Try the correction
                corrected_str = current_str[:pos] + expected_digit + current_str[pos+1:]
                corrected_speed = int(corrected_str)
                
                # Verify the correction makes sense
                change_to_corrected_prev = abs(corrected_speed - prev_speed)
                change_to_corrected_next = abs(corrected_speed - next_speed)
                change_original_prev = abs(current_speed - prev_speed)
                change_original_next = abs(current_speed - next_speed)
                
                # If correction is more consistent with neighbors, use it
                if (change_to_corrected_prev + change_to_corrected_next < 
                    change_original_prev + change_original_next):
                    return corrected_speed
    
    # Try swapping adjacent digits (e.g., 96 instead of 69)
    if len(current_str) >= 2:
        # Try swapping first two digits
        swapped = current_str[1] + current_str[0] + current_str[2:]
        try:
            swapped_speed = int(swapped)
            if 50 <= swapped_speed <= 300:  # Reasonable speed range
                change_swapped_prev = abs(swapped_speed - prev_speed)
                change_swapped_next = abs(swapped_speed - next_speed)
                change_original_prev = abs(current_speed - prev_speed)
                change_original_next = abs(current_speed - next_speed)
                
                if (change_swapped_prev + change_swapped_next < 
                    change_original_prev + change_original_next):
                    return swapped_speed
        except ValueError:
            pass
    
    return None


def smooth_speed_data(results: List[Dict], window_size: int = 3) -> List[Dict]:
    """
    Apply smoothing to reduce noise in speed readings
    
    Args:
        results: List of analysis results
        window_size: Size of smoothing window (odd number recommended)
        
    Returns:
        Results with smoothed speed values
    """
    if not results or window_size < 3:
        return results
    
    smoothed_results = [result.copy() for result in results]
    
    # Initialize smoothed flag for all results
    for result in smoothed_results:
        if 'smoothed' not in result:
            result['smoothed'] = False
    
    # Extract successful speed readings
    successful_indices = []
    speeds = []
    
    for i, result in enumerate(smoothed_results):
        if result['success'] and result['speed'] is not None:
            successful_indices.append(i)
            speeds.append(result['speed'])
    
    if len(speeds) < window_size:
        return smoothed_results
    
    # Apply moving average smoothing
    half_window = window_size // 2
    
    for i, speed_idx in enumerate(successful_indices):
        if i >= half_window and i < len(speeds) - half_window:
            # Calculate moving average
            window_start = i - half_window
            window_end = i + half_window + 1
            window_speeds = speeds[window_start:window_end]
            
            # Remove outliers (speeds that differ by more than 30 km/h from median)
            median_speed = np.median(window_speeds)
            filtered_speeds = [s for s in window_speeds if abs(s - median_speed) <= 30]
            
            if filtered_speeds:
                smoothed_speed = round(np.mean(filtered_speeds))
                
                if smoothed_speed != speeds[i]:
                    smoothed_results[speed_idx]['speed'] = smoothed_speed
                    smoothed_results[speed_idx]['smoothed'] = True
                    original_response = smoothed_results[speed_idx]['response']
                    smoothed_results[speed_idx]['response'] = f"SMOOTHED: AI read {speeds[i]} km/h, adjusted to {smoothed_speed} km/h to reduce noise | Original: {original_response}"
    
    return smoothed_results


def apply_video_crop(video_path: Path, x1: int, y1: int, x2: int, y2: int, progress_callback=None) -> Optional[Path]:
    """
    Crop a video to the specified rectangle using FFmpeg
    
    Args:
        video_path: Path to the input video file
        x1, y1: Top-left coordinates of crop area
        x2, y2: Bottom-right coordinates of crop area
        
    Returns:
        Path to the cropped video file, or None if cropping failed
    """
    try:
        # Calculate crop dimensions
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Ensure even dimensions for video encoding compatibility
        if crop_width % 2 != 0:
            crop_width -= 1
        if crop_height % 2 != 0:
            crop_height -= 1
            
        # Create output path
        output_path = video_path.parent / f"{video_path.stem}_cropped{video_path.suffix}"
        
        # Try hardware acceleration first, fallback to software
        hw_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-hwaccel', 'videotoolbox',  # M2 hardware acceleration
            '-i', str(video_path),  # Input video
            '-filter:v', f'crop={crop_width}:{crop_height}:{x1}:{y1}',  # Crop filter
            '-c:v', 'h264_videotoolbox',  # M2 hardware encoder
            '-b:v', '5M',  # Target bitrate
            '-q:v', '50',  # Quality setting
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)  # Output path
        ]
        
        # Fallback software command
        sw_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', str(video_path),  # Input video
            '-filter:v', f'crop={crop_width}:{crop_height}:{x1}:{y1}',  # Crop filter
            '-c:v', 'libx264',  # Software encoder
            '-preset', 'fast',  # Fast preset
            '-crf', '23',  # Good quality
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)  # Output path
        ]
        
        # Execute FFmpeg command with progress tracking
        if progress_callback:
            progress_callback(0.2, "✂️ Starting video crop (trying hardware acceleration)...")
            
        # Try hardware acceleration first
        cmd = hw_cmd
        use_hardware = True
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        # Monitor progress during cropping
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output and progress_callback:
                # Look for frame progress in FFmpeg output
                if 'frame=' in output:
                    try:
                        frame_info = output.split('frame=')[1].split()[0]
                        if frame_info.isdigit():
                            # Progress from 20% to 90%
                            progress = min(0.9, 0.2 + (int(frame_info) / 1000) * 0.7)
                            progress_callback(progress, f"✂️ Cropping... frame {frame_info}")
                    except:
                        pass
        
        result = process.poll()
        if result == 0:
            if progress_callback:
                hw_msg = " (hardware accelerated)" if use_hardware else " (software fallback)"
                progress_callback(1.0, f"✅ Video cropped successfully: {crop_width}x{crop_height}{hw_msg}")
            print(f"✅ Video cropped successfully: {crop_width}x{crop_height} from ({x1},{y1})")
            return output_path
        else:
            stderr_output = process.stderr.read()
            
            # If hardware acceleration failed, try software fallback
            if use_hardware and ("videotoolbox" in stderr_output.lower() or "hardware" in stderr_output.lower()):
                print("⚠️ Hardware acceleration failed, falling back to software encoding...")
                if progress_callback:
                    progress_callback(0.2, "⚠️ Hardware failed, trying software encoding...")
                
                # Try software fallback
                use_hardware = False
                process = subprocess.Popen(
                    sw_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True
                )
                
                # Monitor progress for fallback
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output and progress_callback:
                        if 'frame=' in output:
                            try:
                                frame_info = output.split('frame=')[1].split()[0]
                                if frame_info.isdigit():
                                    progress = min(0.9, 0.2 + (int(frame_info) / 1000) * 0.7)
                                    progress_callback(progress, f"✂️ Cropping (software)... frame {frame_info}")
                            except:
                                pass
                
                result = process.poll()
                if result == 0:
                    if progress_callback:
                        progress_callback(1.0, f"✅ Video cropped successfully: {crop_width}x{crop_height} (software fallback)")
                    print(f"✅ Video cropped successfully with software fallback: {crop_width}x{crop_height}")
                    return output_path
                else:
                    stderr_output = process.stderr.read()
                    print(f"❌ FFmpeg crop failed (software fallback): {stderr_output}")
                    return None
            else:
                print(f"❌ FFmpeg crop failed: {stderr_output}")
                return None
            
    except subprocess.TimeoutExpired:
        print("❌ Video cropping timed out (>5 minutes)")
        return None
    except Exception as e:
        print(f"❌ Video cropping error: {str(e)}")
        return None


def apply_video_stabilization_tracking_simple(video_path: Path, x1: int, y1: int, x2: int, y2: int, 
                                            output_width: int = 800, output_height: int = 600,
                                            fps: float = 3.0, progress_callback=None) -> Optional[Path]:
    """
    Simplified stabilization using basic FFmpeg filters (fallback if vidstab not available)
    """
    try:
        # Calculate center point of the tracking area
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        track_width = x2 - x1
        track_height = y2 - y1
        
        # Ensure even dimensions for video encoding compatibility
        if output_width % 2 != 0:
            output_width -= 1
        if output_height % 2 != 0:
            output_height -= 1
            
        # Create output path
        output_path = video_path.parent / f"{video_path.stem}_stabilized{video_path.suffix}"
        
        if progress_callback:
            progress_callback(0.1, "🎯 Starting simplified stabilization...")
        
        # Simple stabilization using deshake + crop + scale
        filter_chain = (
            # First downsample to target FPS to match AI processing
            f'fps={fps},'
            # Apply basic deshake stabilization
            f'deshake=x=-1:y=-1:w=-1:h=-1:rx=32:ry=32,'
            # Crop to focus on speedometer area with padding
            f'crop={track_width + 100}:{track_height + 100}:{center_x - 50}:{center_y - 50},'
            # Scale to desired output size
            f'scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,'
            # Add padding if needed to maintain exact dimensions
            f'pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black'
        )
        
        # Hardware accelerated command
        hw_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-hwaccel', 'videotoolbox',  # M2 hardware acceleration
            '-i', str(video_path),  # Input video
            '-vf', filter_chain,
            '-c:v', 'h264_videotoolbox',  # M2 hardware encoder
            '-b:v', '5M',  # Target bitrate for hardware encoder
            '-q:v', '50',  # Quality setting for VideoToolbox
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)  # Output path
        ]
        
        # Software fallback command
        sw_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file  
            '-i', str(video_path),  # Input video
            '-vf', filter_chain,
            '-c:v', 'libx264',  # Software H.264 codec
            '-preset', 'fast',  # Fast encoding preset
            '-crf', '23',  # Good quality
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)  # Output path
        ]
        
        # Try hardware acceleration first
        if progress_callback:
            progress_callback(0.3, "🎯 Applying simplified stabilization (trying hardware)...")
        
        cmd = hw_cmd
        use_hardware = True
        
        # Run stabilization with progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        # Monitor progress
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output and progress_callback:
                if 'frame=' in output:
                    try:
                        frame_info = output.split('frame=')[1].split()[0]
                        if frame_info.isdigit():
                            estimated_duration = 60
                            estimated_total_frames = int(estimated_duration * fps)
                            frame_progress = min(1.0, int(frame_info) / max(estimated_total_frames, 1))
                            progress = min(0.95, 0.3 + frame_progress * 0.65)
                            progress_callback(progress, f"🎯 Stabilizing... frame {frame_info} ({fps} FPS)")
                    except:
                        pass
        
        result = process.poll()
        if result == 0:
            if progress_callback:
                hw_msg = " (hardware accelerated)" if use_hardware else " (software fallback)"
                progress_callback(1.0, f"✅ Video stabilized successfully: {output_width}x{output_height}{hw_msg}")
            print(f"✅ Video stabilized successfully: {output_width}x{output_height}")
            return output_path
        else:
            stderr_output = process.stderr.read()
            
            # If hardware acceleration failed, try software fallback
            if use_hardware and ("videotoolbox" in stderr_output.lower() or "hardware" in stderr_output.lower()):
                print("⚠️ Hardware acceleration failed, falling back to software...")
                if progress_callback:
                    progress_callback(0.3, "⚠️ Hardware failed, trying software...")
                
                # Try software fallback
                use_hardware = False
                process = subprocess.Popen(
                    sw_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True
                )
                
                # Monitor progress for fallback
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output and progress_callback:
                        if 'frame=' in output:
                            try:
                                frame_info = output.split('frame=')[1].split()[0]
                                if frame_info.isdigit():
                                    estimated_duration = 60
                                    estimated_total_frames = int(estimated_duration * fps)
                                    frame_progress = min(1.0, int(frame_info) / max(estimated_total_frames, 1))
                                    progress = min(0.95, 0.3 + frame_progress * 0.65)
                                    progress_callback(progress, f"🎯 Stabilizing (software)... frame {frame_info} ({fps} FPS)")
                            except:
                                pass
                
                result = process.poll()
                if result == 0:
                    if progress_callback:
                        progress_callback(1.0, f"✅ Video stabilized successfully: {output_width}x{output_height} (software fallback)")
                    print(f"✅ Video stabilized with software fallback: {output_width}x{output_height}")
                    return output_path
                else:
                    stderr_output = process.stderr.read()
                    print(f"❌ Simplified stabilization failed (software fallback): {stderr_output}")
                    return None
            else:
                print(f"❌ Simplified stabilization failed: {stderr_output}")
                return None
                
    except Exception as e:
        print(f"❌ Error during simplified stabilization: {e}")
        return None


def apply_video_stabilization_tracking(video_path: Path, x1: int, y1: int, x2: int, y2: int, 
                                     output_width: int = 800, output_height: int = 600,
                                     fps: float = 3.0, progress_callback=None) -> Optional[Path]:
    """
    Advanced video stabilization that tracks and keeps the speedometer area centered using FFmpeg
    
    This function uses motion detection and tracking to keep the specified speedometer area
    stable and centered in the output video, compensating for camera shake and movement.
    Only processes frames at the specified FPS rate (same as will be sent to AI).
    
    Args:
        video_path: Path to the input video file
        x1, y1: Top-left coordinates of the area to track and keep centered
        x2, y2: Bottom-right coordinates of the area to track and keep centered  
        output_width: Width of the stabilized output video
        output_height: Height of the stabilized output video
        fps: Frames per second to process (should match AI analysis FPS)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the stabilized video file, or None if processing failed
    """
    try:
        import tempfile
        
        # Calculate center point of the tracking area
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        track_width = x2 - x1
        track_height = y2 - y1
        
        # Ensure even dimensions for video encoding compatibility
        if output_width % 2 != 0:
            output_width -= 1
        if output_height % 2 != 0:
            output_height -= 1
            
        # Create output path
        output_path = video_path.parent / f"{video_path.stem}_stabilized{video_path.suffix}"
        
        # Create temporary file for motion vectors
        with tempfile.NamedTemporaryFile(suffix='.trf', delete=False) as motion_file:
            motion_vectors_path = motion_file.name
        
        try:
            # Step 1: Detect motion vectors focusing on our tracking area
            if progress_callback:
                progress_callback(0.1, "🔍 Analyzing motion in speedometer area...")
            print("🔍 Analyzing motion in speedometer area...")
            
            detect_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-i', str(video_path),  # Input video
                '-vf', 
                # First downsample to target FPS, then analyze motion
                f'fps={fps},vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result={motion_vectors_path}:tripod=0',
                '-f', 'null',  # No output file for detection pass
                '-'
            ]
            
            # Run motion detection with progress tracking
            process = subprocess.Popen(
                detect_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Monitor progress during detection
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and progress_callback:
                    # Look for frame progress in FFmpeg output
                    if 'frame=' in output:
                        try:
                            frame_info = output.split('frame=')[1].split()[0]
                            if frame_info.isdigit():
                                # Calculate expected total frames based on FPS
                                # Get video duration first (rough estimate)
                                estimated_duration = 60  # Default assumption
                                estimated_total_frames = int(estimated_duration * fps)
                                
                                # Progress from 10% to 45% during detection
                                frame_progress = min(1.0, int(frame_info) / max(estimated_total_frames, 1))
                                progress = min(0.45, 0.1 + frame_progress * 0.35)
                                progress_callback(progress, f"🔍 Analyzing motion... frame {frame_info} ({fps} FPS)")
                        except:
                            pass
            
            detect_result = process.poll()
            if detect_result != 0:
                stderr_output = process.stderr.read()
                print(f"\n🔍 DEBUGGING Motion Detection Failure:")
                print(f"Exit code: {detect_result}")
                print(f"Full stderr output: {stderr_output}")
                print(f"Command that failed: {' '.join(detect_cmd)}")
                print(f"Video path: {video_path}")
                print(f"Motion vectors path: {motion_vectors_path}")
                
                if progress_callback:
                    progress_callback(0.3, f"❌ Motion detection failed (see console for details)")
                return None
                
            if progress_callback:
                progress_callback(0.5, "✅ Motion analysis complete")
            print("✅ Motion analysis complete")
            
            # Step 2: Apply stabilization with tracking to keep speedometer centered
            if progress_callback:
                progress_callback(0.6, "🎯 Stabilizing video to keep speedometer centered...")
            print("🎯 Stabilizing video to keep speedometer centered...")
            
            # Build complex filter chain for tracking and stabilization
            filter_chain = (
                # First downsample to target FPS to match AI processing
                f'fps={fps},'
                # Apply motion compensation/stabilization
                f'vidstabtransform=input={motion_vectors_path}:zoom=0:smoothing=30:crop=black:invert=0,'
                # Crop to focus on stabilized speedometer area with some padding
                f'crop={track_width + 200}:{track_height + 150}:{center_x - 100}:{center_y - 75},'
                # Scale to desired output size
                f'scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,'
                # Add padding if needed to maintain exact dimensions
                f'pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black'
            )
            
            # Hardware accelerated command
            hw_stabilize_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-hwaccel', 'videotoolbox',  # M2 hardware acceleration
                '-i', str(video_path),  # Input video
                '-vf', filter_chain,
                '-c:v', 'h264_videotoolbox',  # M2 hardware encoder
                '-b:v', '5M',  # Target bitrate for hardware encoder
                '-q:v', '50',  # Quality setting for VideoToolbox (lower = better)
                '-c:a', 'copy',  # Copy audio without re-encoding
                str(output_path)  # Output path
            ]
            
            # Software fallback command
            sw_stabilize_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file  
                '-i', str(video_path),  # Input video
                '-vf', filter_chain,
                '-c:v', 'libx264',  # Software H.264 codec
                '-preset', 'medium',  # Balanced encoding preset
                '-crf', '23',  # Good quality
                '-c:a', 'copy',  # Copy audio without re-encoding
                str(output_path)  # Output path
            ]
            
            # Try hardware acceleration first
            if progress_callback:
                progress_callback(0.6, "🎯 Stabilizing video (trying hardware acceleration)...")
            
            stabilize_cmd = hw_stabilize_cmd
            use_hardware = True
            
            # Run stabilization with progress tracking
            process = subprocess.Popen(
                stabilize_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Monitor progress during stabilization
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and progress_callback:
                    # Look for frame progress in FFmpeg output
                    if 'frame=' in output:
                        try:
                            frame_info = output.split('frame=')[1].split()[0]
                            if frame_info.isdigit():
                                # Calculate expected total frames based on FPS
                                estimated_duration = 60  # Default assumption
                                estimated_total_frames = int(estimated_duration * fps)
                                
                                # Progress from 60% to 95% during stabilization
                                frame_progress = min(1.0, int(frame_info) / max(estimated_total_frames, 1))
                                progress = min(0.95, 0.6 + frame_progress * 0.35)
                                progress_callback(progress, f"🎯 Stabilizing... frame {frame_info} ({fps} FPS)")
                        except:
                            pass
            
            stabilize_result = process.poll()
            if stabilize_result == 0:
                if progress_callback:
                    hw_msg = " (hardware accelerated)" if use_hardware else " (software fallback)"
                    progress_callback(1.0, f"✅ Video stabilized successfully: {output_width}x{output_height}{hw_msg}")
                print(f"✅ Video stabilized successfully: {output_width}x{output_height}")
                print(f"🎯 Speedometer area tracked and kept centered")
                return output_path
            else:
                stderr_output = process.stderr.read()
                
                # If hardware acceleration failed, try software fallback
                if use_hardware and ("videotoolbox" in stderr_output.lower() or "hardware" in stderr_output.lower()):
                    print("⚠️ Hardware acceleration failed for stabilization, falling back to software...")
                    if progress_callback:
                        progress_callback(0.6, "⚠️ Hardware failed, trying software stabilization...")
                    
                    # Try software fallback
                    use_hardware = False
                    process = subprocess.Popen(
                        sw_stabilize_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        universal_newlines=True
                    )
                    
                    # Monitor progress for fallback
                    while True:
                        output = process.stderr.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output and progress_callback:
                            if 'frame=' in output:
                                try:
                                    frame_info = output.split('frame=')[1].split()[0]
                                    if frame_info.isdigit():
                                        estimated_duration = 60
                                        estimated_total_frames = int(estimated_duration * fps)
                                        frame_progress = min(1.0, int(frame_info) / max(estimated_total_frames, 1))
                                        progress = min(0.95, 0.6 + frame_progress * 0.35)
                                        progress_callback(progress, f"🎯 Stabilizing (software)... frame {frame_info} ({fps} FPS)")
                                except:
                                    pass
                    
                    stabilize_result = process.poll()
                    if stabilize_result == 0:
                        if progress_callback:
                            progress_callback(1.0, f"✅ Video stabilized successfully: {output_width}x{output_height} (software fallback)")
                        print(f"✅ Video stabilized with software fallback: {output_width}x{output_height}")
                        return output_path
                    else:
                        stderr_output = process.stderr.read()
                        print(f"❌ Video stabilization failed (software fallback): {stderr_output}")
                        return None
                else:
                    print(f"❌ Advanced stabilization failed (exit code {stabilize_result}): {stderr_output}")
                    # Check if it's a vidstab filter issue
                    if "vidstab" in stderr_output.lower() or "unknown filter" in stderr_output.lower():
                        print("⚠️ Advanced stabilization not available, falling back to simplified stabilization...")
                        if progress_callback:
                            progress_callback(0.6, "⚠️ Falling back to simplified stabilization...")
                        return apply_video_stabilization_tracking_simple(video_path, x1, y1, x2, y2, output_width, output_height, fps, progress_callback)
                    else:
                        if progress_callback:
                            progress_callback(0.8, f"❌ Stabilization failed: {stderr_output[:100]}...")
                        return None
                
        finally:
            # Clean up temporary motion vectors file
            try:
                Path(motion_vectors_path).unlink()
            except:
                pass
                
    except Exception as e:
        print(f"❌ Error during video stabilization: {e}")
        return None


def apply_opencv_speedometer_tracking(video_path: Path, x1: int, y1: int, x2: int, y2: int,
                                    output_width: int = 800, output_height: int = 600,
                                    fps: float = 3.0, progress_callback=None) -> Optional[Path]:
    """
    OpenCV-based speedometer tracking that keeps the selected area centered and in view
    
    Uses advanced OpenCV tracking algorithms to follow the speedometer area throughout
    the video and maintains it in the center of the output frame.
    
    Args:
        video_path: Path to input video
        x1, y1, x2, y2: Initial bounding box coordinates for speedometer area
        output_width, output_height: Output video dimensions
        fps: Target FPS for processing
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to processed video or None if failed
    """
    try:
        import cv2
        import numpy as np
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = video_path.parent / f"opencv_tracked_{timestamp}.mp4"
        
        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("❌ Failed to open video file")
            return None
            
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎯 Starting OpenCV speedometer tracking...")
        print(f"📐 Initial tracking area: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"🎬 Video: {frame_width}x{frame_height} @ {original_fps:.1f} FPS, {total_frames} frames")
        
        # Calculate frame processing interval
        frame_interval = max(1, int(original_fps / fps))
        
        # Read first frame to extract template
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read first frame")
            return None
        
        # Set up video writer with browser-compatible codec
        # Try different codecs for maximum compatibility
        codecs_to_try = ['avc1', 'H264', 'XVID', 'mp4v']
        out = None
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
                if out.isOpened():
                    print(f"✅ Using codec: {codec}")
                    break
                else:
                    out.release()
                    out = None
            except:
                continue
                
        if out is None:
            print("❌ Failed to initialize video writer")
            return None
        
        # Use template matching instead of complex trackers for better compatibility
        print("🎯 Using template matching for speedometer tracking...")
        
        # Extract initial template from first frame
        template = frame[y1:y2, x1:x2]
        template_h, template_w = template.shape[:2]
        
        print("✅ Template matching initialized successfully")
        if progress_callback:
            progress_callback(5, "Template matching initialized")
        
        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process only every Nth frame based on target FPS
            if frame_count % frame_interval != 0:
                continue
                
            processed_frames += 1
            
            # Use template matching to find speedometer
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # If match confidence is good enough, use the matched location
            if max_val > 0.6:  # Good match threshold
                # Update tracking position
                x, y = max_loc
                w, h = template_w, template_h
                
                # Ensure bounding box is within frame bounds
                x = max(0, min(x, frame_width - w))
                y = max(0, min(y, frame_height - h))
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
                
                # Extract the tracked region
                tracked_region = frame[y:y+h, x:x+w]
                
                # Resize to output dimensions
                if tracked_region.size > 0:
                    resized_region = cv2.resize(tracked_region, (output_width, output_height))
                    out.write(resized_region)
                else:
                    # Fallback to original area
                    fallback_region = frame[y1:y2, x1:x2]
                    if fallback_region.size > 0:
                        resized_fallback = cv2.resize(fallback_region, (output_width, output_height))
                        out.write(resized_fallback)
                    else:
                        black_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                        out.write(black_frame)
                        
            else:
                # Low confidence match - use original crop area
                fallback_region = frame[y1:y2, x1:x2]
                if fallback_region.size > 0:
                    resized_fallback = cv2.resize(fallback_region, (output_width, output_height))
                    out.write(resized_fallback)
                else:
                    black_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    out.write(black_frame)
            
            # Update progress
            if progress_callback and processed_frames % 10 == 0:
                progress_pct = int((frame_count / total_frames) * 100)
                progress_callback(progress_pct, f"Tracking frame {frame_count}/{total_frames}")
        
        # Clean up
        cap.release()
        out.release()
        
        print(f"✅ OpenCV tracking completed! Processed {processed_frames} frames")
        if progress_callback:
            progress_callback(95, "Converting to browser-compatible format...")
        
        # Convert to browser-compatible format using FFmpeg
        final_output_path = output_path.parent / f"final_{output_path.name}"
        
        try:
            convert_cmd = [
                'ffmpeg', '-y',
                '-i', str(output_path),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Ensures compatibility
                '-movflags', '+faststart',  # Enables progressive download
                str(final_output_path)
            ]
            
            result = subprocess.run(convert_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Remove intermediate file and use final output
                output_path.unlink()
                output_path = final_output_path
                print("✅ Video converted to browser-compatible format")
            else:
                print("⚠️ Video conversion failed, using original format")
                
        except Exception as e:
            print(f"⚠️ Video conversion failed: {e}")
        
        if progress_callback:
            progress_callback(100, "OpenCV tracking complete!")
            
        return output_path
        
    except ImportError:
        print("❌ OpenCV not available. Install with: pip install opencv-python")
        return None
    except Exception as e:
        print(f"❌ Error during OpenCV tracking: {e}")
        return None


def apply_speed_overlay(video_path: str, results: List[Dict], corner_position: str, 
                       x_offset: int, y_offset: int, font_size: int, text_color: str,
                       show_background: bool, bg_opacity: float, progress_callback=None) -> Optional[Path]:
    """
    Apply speed data overlay to video using FFmpeg
    
    Args:
        video_path: Path to input video
        results: List of analysis results with speed data
        corner_position: Corner for overlay ("Top Left", "Top Right", "Bottom Left", "Bottom Right")
        x_offset: Horizontal offset from corner in pixels
        y_offset: Vertical offset from corner in pixels
        font_size: Size of the overlay text
        text_color: Color of the text
        show_background: Whether to show background behind text
        bg_opacity: Opacity of the background (0.0-1.0)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to output video with overlay or None if failed
    """
    try:
        if progress_callback:
            progress_callback(0.1, "🎬 Preparing video overlay...")
        
        video_path = Path(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = video_path.parent / f"speed_overlay_{timestamp}.mp4"
        
        # Create a subtitle file with speed data
        subtitle_path = video_path.parent / f"speed_data_{timestamp}.ass"
        
        if progress_callback:
            progress_callback(0.2, "📝 Creating speed data subtitle file...")
        
        # Generate ASS subtitle file with speed overlays
        create_speed_subtitle_file(results, subtitle_path, corner_position, 
                                  x_offset, y_offset, font_size, text_color, 
                                  show_background, bg_opacity)
        
        if progress_callback:
            progress_callback(0.4, "🎥 Applying overlay to video...")
        
        # FFmpeg command to apply subtitle overlay
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', str(video_path),  # Input video
            '-vf', f'ass={subtitle_path}',  # Apply ASS subtitles as overlay
            '-c:v', 'libx264',  # Video codec
            '-preset', 'medium',  # Encoding preset
            '-crf', '23',  # Quality
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)  # Output path
        ]
        
        # Try hardware acceleration if available
        hw_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-hwaccel', 'videotoolbox',  # Hardware acceleration
            '-i', str(video_path),  # Input video
            '-vf', f'ass={subtitle_path}',  # Apply ASS subtitles as overlay
            '-c:v', 'h264_videotoolbox',  # Hardware encoder
            '-b:v', '5M',  # Target bitrate
            '-c:a', 'copy',  # Copy audio without re-encoding
            str(output_path)  # Output path
        ]
        
        # Try hardware acceleration first
        process = subprocess.Popen(
            hw_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        # Monitor progress
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output and progress_callback:
                if 'frame=' in output:
                    try:
                        frame_info = output.split('frame=')[1].split()[0]
                        if frame_info.isdigit():
                            # Progress from 40% to 90%
                            progress = min(0.9, 0.4 + (int(frame_info) / 1000) * 0.5)
                            progress_callback(progress, f"🎬 Processing frame {frame_info}...")
                    except:
                        pass
        
        result = process.poll()
        if result != 0:
            # Hardware acceleration failed, try software fallback
            if progress_callback:
                progress_callback(0.5, "⚠️ Hardware failed, trying software encoding...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Monitor progress for software encoding
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and progress_callback:
                    if 'frame=' in output:
                        try:
                            frame_info = output.split('frame=')[1].split()[0]
                            if frame_info.isdigit():
                                progress = min(0.9, 0.5 + (int(frame_info) / 1000) * 0.4)
                                progress_callback(progress, f"🎬 Processing (software) frame {frame_info}...")
                        except:
                            pass
            
            result = process.poll()
        
        if result == 0:
            if progress_callback:
                progress_callback(1.0, "✅ Speed overlay applied successfully!")
            
            # Clean up subtitle file
            try:
                subtitle_path.unlink()
            except:
                pass
                
            return output_path
        else:
            stderr_output = process.stderr.read()
            print(f"❌ FFmpeg overlay failed: {stderr_output}")
            return None
            
    except Exception as e:
        print(f"❌ Error during speed overlay: {e}")
        return None


def create_speed_subtitle_file(results: List[Dict], subtitle_path: Path, corner_position: str,
                              x_offset: int, y_offset: int, font_size: int, text_color: str,
                              show_background: bool, bg_opacity: float):
    """
    Create an ASS subtitle file with speed data overlays
    """
    # Color mapping for ASS format (BGR format)
    color_map = {
        "White": "&HFFFFFF",
        "Yellow": "&H00FFFF", 
        "Green": "&H00FF00",
        "Red": "&H0000FF",
        "Blue": "&HFF0000",
        "Black": "&H000000"
    }
    
    text_color_code = color_map.get(text_color, "&H00FFFF")  # Default to yellow
    
    # Position mapping for ASS alignment
    if corner_position == "Top Left":
        alignment = 7  # Top left
        margin_x = x_offset
        margin_y = y_offset
    elif corner_position == "Top Right":
        alignment = 9  # Top right  
        margin_x = x_offset
        margin_y = y_offset
    elif corner_position == "Bottom Left":
        alignment = 1  # Bottom left
        margin_x = x_offset
        margin_y = y_offset
    else:  # Bottom Right
        alignment = 3  # Bottom right
        margin_x = x_offset
        margin_y = y_offset
    
    # Background settings
    if show_background:
        bg_alpha = hex(int((1 - bg_opacity) * 255))[2:].upper().zfill(2)
        outline_color = f"&H{bg_alpha}000000"  # Semi-transparent black background
        outline_width = 3
    else:
        outline_color = "&H00000000"  # Transparent
        outline_width = 0
    
    # Create ASS subtitle content
    ass_content = f"""[Script Info]
Title: Speed Overlay
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{text_color_code},&HFFFFFF,{outline_color},&H00000000,1,0,0,0,100,100,0,0,1,{outline_width},2,{alignment},{margin_x},{margin_x},{margin_y},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Add speed data events
    for i, result in enumerate(results):
        if result.get('success', False) and result.get('speed') is not None:
            # Convert timestamp to ASS time format (hours:minutes:seconds.centiseconds)
            start_time = result['timestamp']
            # Each frame shows for duration until next frame (or 1 second for last frame)
            if i < len(results) - 1:
                end_time = results[i + 1]['timestamp']
            else:
                end_time = start_time + 1.0
            
            start_ass = seconds_to_ass_time(start_time)
            end_ass = seconds_to_ass_time(end_time)
            
            speed_text = f"{int(result['speed'])} km/h"
            
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{speed_text}\n"
    
    # Write subtitle file
    with open(subtitle_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)


def seconds_to_ass_time(seconds: float) -> str:
    """
    Convert seconds to ASS time format (H:MM:SS.CC)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"