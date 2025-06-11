"""
Tests for core SpeedometerAnalyzer functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from PIL import Image

from speedometer_ai.core import SpeedometerAnalyzer


class TestSpeedometerAnalyzer:
    
    def test_init(self, mock_api_key):
        """Test analyzer initialization"""
        with patch('speedometer_ai.core.genai.configure') as mock_configure, \
             patch('speedometer_ai.core.genai.GenerativeModel') as mock_model:
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            
            mock_configure.assert_called_once_with(api_key=mock_api_key)
            mock_model.assert_called_once_with("gemini-1.5-flash")
            assert analyzer.results == []
    
    def test_create_prompt(self, mock_api_key):
        """Test prompt creation"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel'):
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            prompt = analyzer.create_prompt()
            
            assert "Analyze this car dashboard image" in prompt
            assert "digital speedometer" in prompt
            assert "RESPONSE FORMAT" in prompt
            assert "Speed reading:" in prompt
    
    def test_parse_response_valid_speed(self, mock_api_key):
        """Test parsing valid speed responses"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel'):
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            
            # Test various valid responses
            assert analyzer._parse_response("157") == 157
            assert analyzer._parse_response("Speed reading: 125") == 125
            assert analyzer._parse_response("The speed is 89 km/h") == 89
            assert analyzer._parse_response("195") == 195
    
    def test_parse_response_invalid_speed(self, mock_api_key):
        """Test parsing invalid speed responses"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel'):
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            
            # Test unclear responses
            assert analyzer._parse_response("UNCLEAR") is None
            assert analyzer._parse_response("Cannot see") is None
            assert analyzer._parse_response("Unable to read") is None
            assert analyzer._parse_response("Not visible") is None
            
            # Test out of range speeds
            assert analyzer._parse_response("25") is None  # Too low
            assert analyzer._parse_response("450") is None  # Too high
            assert analyzer._parse_response("no numbers here") is None
    
    def test_analyze_frame_success(self, mock_api_key, mock_dashboard_image):
        """Test successful frame analysis"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel') as mock_model_class:
            
            # Mock the model and response
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "157"
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            speed, response = analyzer.analyze_frame(mock_dashboard_image)
            
            assert speed == 157
            assert response == "157"
            mock_model.generate_content.assert_called_once()
    
    def test_analyze_frame_error(self, mock_api_key, mock_dashboard_image):
        """Test frame analysis with error"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel') as mock_model_class:
            
            # Mock the model to raise an exception
            mock_model = Mock()
            mock_model.generate_content.side_effect = Exception("API Error")
            mock_model_class.return_value = mock_model
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            speed, response = analyzer.analyze_frame(mock_dashboard_image)
            
            assert speed is None
            assert "Error: API Error" in response
    
    def test_analyze_video_frames(self, mock_api_key, temp_dir):
        """Test analyzing multiple video frames"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel') as mock_model_class:
            
            # Create mock frame files
            frame_files = []
            for i in range(3):
                frame_path = temp_dir / f"frame_{i:03d}.png"
                img = Image.new('RGB', (100, 100), color='black')
                img.save(frame_path)
                frame_files.append(frame_path)
            
            # Mock the model responses
            mock_model = Mock()
            responses = [Mock() for _ in range(3)]
            responses[0].text = "157"
            responses[1].text = "145" 
            responses[2].text = "UNCLEAR"
            mock_model.generate_content.side_effect = responses
            mock_model_class.return_value = mock_model
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            
            # Track progress calls
            progress_calls = []
            def mock_progress(current, total, filename):
                progress_calls.append((current, total, filename))
            
            results = analyzer.analyze_video_frames(
                temp_dir, fps=1.0, delay_seconds=0, progress_callback=mock_progress
            )
            
            assert len(results) == 3
            assert results[0]['speed'] == 157
            assert results[0]['success'] is True
            assert results[1]['speed'] == 145
            assert results[1]['success'] is True
            assert results[2]['speed'] is None
            assert results[2]['success'] is False
            
            # Check progress was called
            assert len(progress_calls) == 3
            assert progress_calls[0] == (1, 3, "frame_000.png")
            assert progress_calls[1] == (2, 3, "frame_001.png")
            assert progress_calls[2] == (3, 3, "frame_002.png")
    
    def test_get_statistics(self, mock_api_key):
        """Test statistics calculation"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel'):
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            
            # Set mock results
            analyzer.results = [
                {'timestamp': 0.0, 'speed': 157, 'success': True},
                {'timestamp': 1.0, 'speed': 145, 'success': True},
                {'timestamp': 2.0, 'speed': None, 'success': False},
                {'timestamp': 3.0, 'speed': 132, 'success': True},
            ]
            
            stats = analyzer.get_statistics()
            
            assert stats['total_frames'] == 4
            assert stats['successful_readings'] == 3
            assert stats['success_rate'] == 75.0
            assert stats['duration'] == 3.0
            assert stats['min_speed'] == 132
            assert stats['max_speed'] == 157
            assert stats['avg_speed'] == (157 + 145 + 132) / 3
            assert stats['speed_range'] == 157 - 132
    
    def test_get_statistics_no_results(self, mock_api_key):
        """Test statistics with no results"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel'):
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            stats = analyzer.get_statistics()
            
            assert stats == {}
    
    def test_get_statistics_no_successful_readings(self, mock_api_key):
        """Test statistics with no successful readings"""
        with patch('speedometer_ai.core.genai.configure'), \
             patch('speedometer_ai.core.genai.GenerativeModel'):
            
            analyzer = SpeedometerAnalyzer(mock_api_key)
            analyzer.results = [
                {'timestamp': 0.0, 'speed': None, 'success': False},
                {'timestamp': 1.0, 'speed': None, 'success': False},
            ]
            
            stats = analyzer.get_statistics()
            
            assert stats['total_frames'] == 2
            assert stats['successful_readings'] == 0
            assert stats['success_rate'] == 0.0
            assert stats['duration'] == 1.0
            assert 'min_speed' not in stats
            assert 'max_speed' not in stats