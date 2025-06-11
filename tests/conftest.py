"""
Test configuration and fixtures
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_video_file(temp_dir):
    """Create a mock video file for testing"""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 3.0, (640, 480))
    
    # Create 10 frames with different content
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some content to make frames different
        cv2.putText(frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def mock_dashboard_image(temp_dir):
    """Create a mock dashboard image with speedometer"""
    img_path = temp_dir / "dashboard.png"
    
    # Create a simple dashboard image
    img = Image.new('RGB', (640, 480), color='black')
    
    # You could add more realistic dashboard elements here
    # For now, just save a simple black image
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    return "test_api_key_12345"


@pytest.fixture
def sample_analysis_results():
    """Sample analysis results for testing"""
    return [
        {
            'frame': 1,
            'timestamp': 0.0,
            'filename': 'frame_001.png',
            'speed': 157,
            'response': '157',
            'success': True
        },
        {
            'frame': 2,
            'timestamp': 0.33,
            'filename': 'frame_002.png',
            'speed': 145,
            'response': '145',
            'success': True
        },
        {
            'frame': 3,
            'timestamp': 0.67,
            'filename': 'frame_003.png',
            'speed': None,
            'response': 'UNCLEAR',
            'success': False
        },
        {
            'frame': 4,
            'timestamp': 1.0,
            'filename': 'frame_004.png',
            'speed': 132,
            'response': '132',
            'success': True
        }
    ]


@pytest.fixture
def sample_csv_file(temp_dir, sample_analysis_results):
    """Create a sample CSV file with analysis results"""
    import csv
    
    csv_path = temp_dir / "test_results.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'timestamp', 'speed', 'filename', 'success', 'response'
        ])
        writer.writeheader()
        for result in sample_analysis_results:
            writer.writerow(result)
    
    return csv_path


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    # Disable Streamlit telemetry for tests
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    # Set a test API key
    monkeypatch.setenv("GEMINI_API_KEY", "test_api_key_12345")