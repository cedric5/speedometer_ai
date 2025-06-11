"""
Tests for utility functions
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
import subprocess

from speedometer_ai.utils import (
    extract_frames_from_video,
    save_results_to_csv,
    load_results_from_csv,
    create_speed_chart,
    print_analysis_summary,
    validate_video_file,
    check_ffmpeg_available
)


class TestExtractFrames:
    
    def test_extract_frames_success(self, temp_dir, mock_video_file):
        """Test successful frame extraction"""
        frames_dir = temp_dir / "frames"
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            # Mock the glob to return some temp frame files, then final files
            temp_files = [
                frames_dir / "temp_frame_000001.png",
                frames_dir / "temp_frame_000002.png", 
                frames_dir / "temp_frame_000003.png"
            ]
            final_files = [
                frames_dir / "frame_t000.00s.png",
                frames_dir / "frame_t000.33s.png",
                frames_dir / "frame_t000.67s.png"
            ]
            
            with patch.object(Path, 'glob', return_value=temp_files), \
                 patch.object(Path, 'rename') as mock_rename:
                result = extract_frames_from_video(mock_video_file, frames_dir, fps=3.0)
                
                assert len(result) == 3
                assert all(isinstance(f, Path) for f in result)
                
                # Check FFmpeg command was called correctly
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert 'ffmpeg' in args
                assert '-vf' in args
                assert 'fps=3.0' in args
    
    def test_extract_frames_ffmpeg_error(self, temp_dir, mock_video_file):
        """Test FFmpeg error handling"""
        frames_dir = temp_dir / "frames"
        
        with patch('subprocess.run') as mock_run:
            error = subprocess.CalledProcessError(1, 'ffmpeg')
            error.stderr = "FFmpeg error message"
            mock_run.side_effect = error
            
            with pytest.raises(RuntimeError, match="FFmpeg failed"):
                extract_frames_from_video(mock_video_file, frames_dir, fps=3.0)
    
    def test_extract_frames_ffmpeg_not_found(self, temp_dir, mock_video_file):
        """Test FFmpeg not found error"""
        frames_dir = temp_dir / "frames"
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                extract_frames_from_video(mock_video_file, frames_dir, fps=3.0)


class TestCSVOperations:
    
    def test_save_results_to_csv(self, temp_dir, sample_analysis_results):
        """Test saving results to CSV"""
        csv_path = temp_dir / "test.csv"
        
        save_results_to_csv(sample_analysis_results, csv_path)
        
        assert csv_path.exists()
        
        # Read and verify content
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 4
            assert rows[0]['frame'] == '1'
            assert rows[0]['speed'] == '157'
            assert rows[0]['success'] == 'True'
            assert rows[2]['speed'] == ''  # None speed becomes empty string
            assert rows[2]['success'] == 'False'
    
    def test_save_empty_results(self, temp_dir):
        """Test saving empty results"""
        csv_path = temp_dir / "empty.csv"
        
        save_results_to_csv([], csv_path)
        
        # File should be created but empty
        assert csv_path.exists()
        assert csv_path.stat().st_size == 0
    
    def test_load_results_from_csv(self, sample_csv_file):
        """Test loading results from CSV"""
        df = load_results_from_csv(sample_csv_file)
        
        assert len(df) == 4
        assert 'frame' in df.columns
        assert 'speed' in df.columns
        assert 'success' in df.columns
        assert df.iloc[0]['speed'] == 157
        assert df.iloc[2]['success'] == False  # Use == instead of is


class TestChartCreation:
    
    def test_create_speed_chart_success(self, temp_dir, sample_analysis_results):
        """Test creating speed chart with successful results"""
        chart_path = temp_dir / "chart.png"
        
        # Mock the matplotlib import inside the function
        mock_plt = Mock()
        with patch.dict('sys.modules', {'matplotlib.pyplot': mock_plt}):
            create_speed_chart(sample_analysis_results, chart_path)
            
            # Check matplotlib functions were called
            mock_plt.figure.assert_called_once()
            mock_plt.plot.assert_called_once()
            mock_plt.title.assert_called_once()
            mock_plt.savefig.assert_called_once_with(chart_path, dpi=300, bbox_inches='tight')
            mock_plt.close.assert_called_once()
    
    def test_create_speed_chart_no_successful_results(self, temp_dir, capsys):
        """Test creating chart with no successful results"""
        results = [
            {'speed': None, 'success': False, 'timestamp': 0.0},
            {'speed': None, 'success': False, 'timestamp': 1.0},
        ]
        chart_path = temp_dir / "chart.png"
        
        create_speed_chart(results, chart_path)
        
        captured = capsys.readouterr()
        assert "No successful readings to plot" in captured.out
    
    def test_create_speed_chart_no_matplotlib(self, temp_dir, sample_analysis_results, capsys):
        """Test chart creation when matplotlib is not available"""
        chart_path = temp_dir / "chart.png"
        
        with patch('builtins.__import__', side_effect=lambda name, *args: 
                   ImportError() if name == 'matplotlib.pyplot' else __import__(name, *args)):
            create_speed_chart(sample_analysis_results, chart_path)
            
            captured = capsys.readouterr()
            assert "Matplotlib not available" in captured.out


class TestPrintSummary:
    
    def test_print_analysis_summary(self, sample_analysis_results, capsys):
        """Test printing analysis summary"""
        stats = {
            'total_frames': 4,
            'successful_readings': 3,
            'success_rate': 75.0,
            'duration': 1.0,
            'min_speed': 132,
            'max_speed': 157,
            'avg_speed': 144.7
        }
        
        print_analysis_summary(sample_analysis_results, stats)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "SPEEDOMETER ANALYSIS RESULTS" in output
        assert "Total frames: 4" in output
        assert "Successful readings: 3" in output
        assert "Success rate: 75.0%" in output
        assert "Duration: 1.0 seconds" in output
        assert "Speed range: 132-157 km/h" in output
        assert "Average speed: 144.7 km/h" in output
        assert "DETAILED TIMELINE:" in output


class TestValidation:
    
    def test_validate_video_file_success(self, mock_video_file):
        """Test successful video file validation"""
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap_class.return_value = mock_cap
            
            result = validate_video_file(mock_video_file)
            
            assert result is True
            mock_cap_class.assert_called_once_with(str(mock_video_file))
            mock_cap.release.assert_called_once()
    
    def test_validate_video_file_not_exists(self, temp_dir):
        """Test validation of non-existent file"""
        fake_path = temp_dir / "nonexistent.mp4"
        
        result = validate_video_file(fake_path)
        
        assert result is False
    
    def test_validate_video_file_opencv_error(self, mock_video_file):
        """Test validation with OpenCV error"""
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cap_class.return_value = mock_cap
            
            result = validate_video_file(mock_video_file)
            
            assert result is False
    
    def test_validate_video_file_exception(self, mock_video_file):
        """Test validation with exception"""
        with patch('cv2.VideoCapture', side_effect=Exception("OpenCV error")):
            result = validate_video_file(mock_video_file)
            
            assert result is False
    
    def test_check_ffmpeg_available_success(self):
        """Test FFmpeg availability check - success"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            result = check_ffmpeg_available()
            
            assert result is True
            mock_run.assert_called_once_with(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                check=True
            )
    
    def test_check_ffmpeg_available_not_found(self):
        """Test FFmpeg availability check - not found"""
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            result = check_ffmpeg_available()
            
            assert result is False
    
    def test_check_ffmpeg_available_error(self):
        """Test FFmpeg availability check - error"""
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffmpeg')):
            result = check_ffmpeg_available()
            
            assert result is False