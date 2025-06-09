"""
Speedometer AI - AI-powered speedometer reading from dashboard video
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .core import SpeedometerAnalyzer
from .utils import extract_frames_from_video, save_results_to_csv