"""
Tests for CLI functionality
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, Mock, call
from click.testing import CliRunner

from speedometer_ai.cli import cli, analyze, show, ui


class TestCLI:
    
    def test_cli_help(self):
        """Test CLI help message"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Speedometer AI - AI-powered speedometer reading" in result.output
        assert "analyze" in result.output
        assert "show" in result.output
        assert "ui" in result.output
    
    def test_analyze_help(self):
        """Test analyze command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', '--help'])
        
        assert result.exit_code == 0
        assert "Analyze speedometer readings from dashboard video" in result.output
        assert "--api-key" in result.output
        assert "--fps" in result.output
        assert "--delay" in result.output


class TestAnalyzeCommand:
    
    @patch('speedometer_ai.cli.SpeedometerAnalyzer')
    @patch('speedometer_ai.cli.extract_frames_from_video')
    @patch('speedometer_ai.cli.save_results_to_csv')
    @patch('speedometer_ai.cli.create_speed_chart')
    @patch('speedometer_ai.cli.validate_video_file')
    @patch('speedometer_ai.cli.check_ffmpeg_available')
    def test_analyze_success(self, mock_ffmpeg, mock_validate, mock_chart, 
                           mock_save_csv, mock_extract, mock_analyzer_class,
                           temp_dir, mock_video_file):
        """Test successful video analysis"""
        # Setup mocks
        mock_ffmpeg.return_value = True
        mock_validate.return_value = True
        mock_extract.return_value = [temp_dir / "frame_001.png"]
        
        results_data = [{'speed': 157, 'success': True, 'timestamp': 0.0}]
        mock_analyzer = Mock()
        mock_analyzer.analyze_video_frames.return_value = results_data
        mock_analyzer.get_statistics.return_value = {
            'success_rate': 100.0,
            'min_speed': 157,
            'max_speed': 157,
            'avg_speed': 157.0
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock print_analysis_summary to avoid the error
        with patch('speedometer_ai.cli.print_analysis_summary') as mock_print:
            runner = CliRunner()
            with runner.isolated_filesystem():
                # Create a temporary video file
                video_path = Path("test_video.mp4")
                video_path.touch()
                
                result = runner.invoke(cli, [
                    'analyze', str(video_path),
                    '--api-key', 'test_key',
                    '--fps', '3.0',
                    '--delay', '0.5',
                    '--verbose'
                ])
                
                assert result.exit_code == 0
                assert "Speedometer AI v1.0.0" in result.output
                assert "Extracting frames from video" in result.output
                assert "Analysis complete!" in result.output
                
                # Check mocks were called
                mock_extract.assert_called_once()
                mock_analyzer_class.assert_called_once_with('test_key')
                mock_save_csv.assert_called_once()
                mock_chart.assert_called_once()
    
    def test_analyze_invalid_video(self):
        """Test analyze with invalid video file"""
        with patch('speedometer_ai.cli.validate_video_file', return_value=False):
            runner = CliRunner()
            with runner.isolated_filesystem():
                video_path = Path("fake_video.mp4")
                video_path.touch()
                
                result = runner.invoke(cli, [
                    'analyze', str(video_path),
                    '--api-key', 'test_key'
                ])
                
                assert result.exit_code == 1
                assert "Invalid or unreadable video file" in result.output
    
    def test_analyze_no_ffmpeg(self):
        """Test analyze without FFmpeg"""
        with patch('speedometer_ai.cli.validate_video_file', return_value=True), \
             patch('speedometer_ai.cli.check_ffmpeg_available', return_value=False):
            
            runner = CliRunner()
            with runner.isolated_filesystem():
                video_path = Path("test_video.mp4")
                video_path.touch()
                
                result = runner.invoke(cli, [
                    'analyze', str(video_path),
                    '--api-key', 'test_key'
                ])
                
                assert result.exit_code == 1
                assert "FFmpeg not found" in result.output
    
    def test_analyze_no_api_key(self):
        """Test analyze without API key"""
        with patch('speedometer_ai.cli.validate_video_file', return_value=True), \
             patch('speedometer_ai.cli.check_ffmpeg_available', return_value=True), \
             patch.dict(os.environ, {}, clear=True):
            
            runner = CliRunner()
            with runner.isolated_filesystem():
                video_path = Path("test_video.mp4")
                video_path.touch()
                
                result = runner.invoke(cli, [
                    'analyze', str(video_path)
                ], input='\n')  # Empty input for API key prompt
                
                assert result.exit_code == 1
                assert "No API key provided" in result.output
    
    def test_analyze_with_env_api_key(self):
        """Test analyze with API key from environment"""
        with patch('speedometer_ai.cli.validate_video_file', return_value=True), \
             patch('speedometer_ai.cli.check_ffmpeg_available', return_value=True), \
             patch('speedometer_ai.cli.extract_frames_from_video'), \
             patch('speedometer_ai.cli.SpeedometerAnalyzer') as mock_analyzer_class, \
             patch('speedometer_ai.cli.save_results_to_csv'), \
             patch('speedometer_ai.cli.create_speed_chart'), \
             patch.dict(os.environ, {'GEMINI_API_KEY': 'env_test_key'}):
            
            mock_analyzer = Mock()
            mock_analyzer.analyze_video_frames.return_value = []
            mock_analyzer.get_statistics.return_value = {'success_rate': 0}
            mock_analyzer_class.return_value = mock_analyzer
            
            runner = CliRunner()
            with runner.isolated_filesystem():
                video_path = Path("test_video.mp4")
                video_path.touch()
                
                result = runner.invoke(cli, [
                    'analyze', str(video_path)
                ])
                
                assert result.exit_code == 0
                mock_analyzer_class.assert_called_once_with('env_test_key')
    
    @patch('speedometer_ai.cli.SpeedometerAnalyzer')
    @patch('speedometer_ai.cli.extract_frames_from_video')
    @patch('speedometer_ai.cli.validate_video_file')
    @patch('speedometer_ai.cli.check_ffmpeg_available')
    def test_analyze_exception(self, mock_ffmpeg, mock_validate, mock_extract, mock_analyzer_class):
        """Test analyze with exception during processing"""
        mock_ffmpeg.return_value = True
        mock_validate.return_value = True
        mock_extract.side_effect = Exception("Test error")
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            video_path = Path("test_video.mp4")
            video_path.touch()
            
            result = runner.invoke(cli, [
                'analyze', str(video_path),
                '--api-key', 'test_key'
            ])
            
            assert result.exit_code == 1
            assert "Error: Test error" in result.output


class TestShowCommand:
    
    def test_show_success(self, sample_csv_file):
        """Test successful show command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['show', str(sample_csv_file)])
        
        assert result.exit_code == 0
        assert "Speed Analysis Results" in result.output
        assert "Total frames: 4" in result.output
        assert "Successful readings: 3" in result.output
        assert "Success rate: 75.0%" in result.output
        assert "Speed range: 132-157 km/h" in result.output
        assert "Detailed timeline:" in result.output
    
    def test_show_nonexistent_file(self):
        """Test show command with non-existent file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['show', 'nonexistent.csv'])
        
        assert result.exit_code == 2  # Click file not found error
    
    def test_show_invalid_csv(self, temp_dir):
        """Test show command with invalid CSV"""
        invalid_csv = temp_dir / "invalid.csv"
        invalid_csv.write_text("not,valid,csv,content\n")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['show', str(invalid_csv)])
        
        assert result.exit_code == 0
        assert "Error reading results:" in result.output


class TestUICommand:
    
    @patch('speedometer_ai.cli.subprocess.run')
    @patch('speedometer_ai.cli.Path')
    def test_ui_success(self, mock_path_class, mock_subprocess):
        """Test successful UI launch"""
        mock_ui_script = Mock()
        mock_path_class.return_value.parent = Mock()
        mock_path_class.return_value.parent.__truediv__.return_value = mock_ui_script
        
        runner = CliRunner()
        result = runner.invoke(cli, ['ui'])
        
        assert result.exit_code == 0
        assert "Launching Speedometer AI Web UI" in result.output
        assert "Open http://localhost:8501" in result.output
        mock_subprocess.assert_called_once()
        
        # Check subprocess call
        args = mock_subprocess.call_args[0][0]
        assert "streamlit" in args
        assert "run" in args
    
    @patch('speedometer_ai.cli.subprocess.run', side_effect=ImportError())
    def test_ui_streamlit_not_installed(self, mock_subprocess):
        """Test UI command when Streamlit is not installed"""
        runner = CliRunner()
        result = runner.invoke(cli, ['ui'])
        
        assert result.exit_code == 0
        assert "Streamlit not installed" in result.output
    
    @patch('speedometer_ai.cli.subprocess.run', side_effect=Exception("Launch error"))
    @patch('speedometer_ai.cli.Path')
    def test_ui_launch_error(self, mock_path_class, mock_subprocess):
        """Test UI command with launch error"""
        mock_ui_script = Mock()
        mock_path_class.return_value.parent = Mock()
        mock_path_class.return_value.parent.__truediv__.return_value = mock_ui_script
        
        runner = CliRunner()
        result = runner.invoke(cli, ['ui'])
        
        assert result.exit_code == 0
        assert "Error launching UI: Launch error" in result.output