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


def apply_video_crop(video_path: Path, x1: int, y1: int, x2: int, y2: int) -> Optional[Path]:
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
        
        # Build FFmpeg command for cropping
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', str(video_path),  # Input video
            '-filter:v', f'crop={crop_width}:{crop_height}:{x1}:{y1}',  # Crop filter
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-preset', 'fast',  # Fast encoding preset
            str(output_path)  # Output path
        ]
        
        # Execute FFmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Video cropped successfully: {crop_width}x{crop_height} from ({x1},{y1})")
            return output_path
        else:
            print(f"❌ FFmpeg crop failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Video cropping timed out (>5 minutes)")
        return None
    except Exception as e:
        print(f"❌ Video cropping error: {str(e)}")
        return None