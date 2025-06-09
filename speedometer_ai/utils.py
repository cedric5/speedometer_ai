"""
Utility functions for video processing and data handling
"""

import cv2
import csv
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def extract_frames_from_video(video_path: Path, 
                            output_dir: Path, 
                            fps: float = 3.0) -> List[Path]:
    """
    Extract frames from video using FFmpeg
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Frames per second to extract
        
    Returns:
        List of extracted frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-y',  # Overwrite output files
        str(output_dir / 'frame_%03d.png')
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Get list of created frames
        frame_files = sorted(output_dir.glob('frame_*.png'))
        return frame_files
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")


def save_results_to_csv(results: List[Dict], output_path: Path) -> None:
    """
    Save analysis results to CSV
    
    Args:
        results: List of analysis results
        output_path: Path to save CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if not results:
            return
            
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'timestamp', 'speed', 'filename', 'success', 'response'
        ])
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'frame': result['frame'],
                'timestamp': round(result['timestamp'], 2),
                'speed': result['speed'] if result['speed'] else '',
                'filename': result['filename'],
                'success': result['success'],
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