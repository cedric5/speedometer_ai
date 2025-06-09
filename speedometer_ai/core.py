"""
Core speedometer analysis functionality using Gemini AI
"""

import google.generativeai as genai
import time
import re
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple


class SpeedometerAnalyzer:
    """AI-powered speedometer analyzer using Google Gemini"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the analyzer
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.results = []
        
    def create_prompt(self) -> str:
        """Create the analysis prompt for Gemini"""
        return """
Analyze this car dashboard image and read the digital speedometer.

TASK: Find and read the digital speed display in the dashboard gauges.

INSTRUCTIONS:
1. Look for circular gauges in the center console area
2. Find the gauge with a red LED digital display showing speed
3. The display typically shows 2-3 digits (speed in km/h)
4. Focus on the clearest digital readout you can see

RESPONSE FORMAT:
Respond with ONLY the speed number you see.
Examples: "157", "125", "195"
If unclear, respond: "UNCLEAR"

Speed reading:"""

    def analyze_frame(self, image_path: Path) -> Tuple[Optional[int], str]:
        """
        Analyze a single frame for speed reading
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (speed_value, raw_response)
        """
        try:
            image = Image.open(image_path)
            prompt = self.create_prompt()
            
            response = self.model.generate_content([prompt, image])
            response_text = response.text.strip()
            
            speed = self._parse_response(response_text)
            return speed, response_text
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def _parse_response(self, response: str) -> Optional[int]:
        """Parse Gemini response to extract speed value"""
        cleaned = response.strip().upper()
        
        # Check for unclear responses
        if any(word in cleaned for word in ['UNCLEAR', 'CANNOT', 'UNABLE', 'NOT VISIBLE']):
            return None
        
        # Extract numbers
        numbers = re.findall(r'\b\d{2,3}\b', cleaned)
        
        # Filter for reasonable speeds
        for num_str in numbers:
            num = int(num_str)
            if 50 <= num <= 300:  # Reasonable speed range
                return num
        
        return None
    
    def analyze_video_frames(self, 
                           frames_dir: Path, 
                           fps: float = 3.0,
                           delay_seconds: float = 1.0,
                           progress_callback=None) -> List[Dict]:
        """
        Analyze all frames in a directory
        
        Args:
            frames_dir: Directory containing frame images
            fps: Frames per second of extraction
            delay_seconds: Delay between API calls
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of analysis results
        """
        frame_files = sorted(frames_dir.glob("*.png"))
        
        if not frame_files:
            raise ValueError(f"No PNG frames found in {frames_dir}")
        
        results = []
        
        for i, frame_path in enumerate(frame_files):
            timestamp = i / fps
            
            if progress_callback:
                progress_callback(i + 1, len(frame_files), frame_path.name)
            
            speed, response = self.analyze_frame(frame_path)
            
            result = {
                'frame': i + 1,
                'timestamp': timestamp,
                'filename': frame_path.name,
                'speed': speed,
                'response': response,
                'success': speed is not None
            }
            
            results.append(result)
            
            # Rate limiting
            if delay_seconds > 0 and i < len(frame_files) - 1:
                time.sleep(delay_seconds)
        
        self.results = results
        return results
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r['success']]
        speeds = [r['speed'] for r in successful]
        
        stats = {
            'total_frames': len(self.results),
            'successful_readings': len(successful),
            'success_rate': len(successful) / len(self.results) * 100,
            'duration': self.results[-1]['timestamp'] if self.results else 0
        }
        
        if speeds:
            stats.update({
                'min_speed': min(speeds),
                'max_speed': max(speeds),
                'avg_speed': sum(speeds) / len(speeds),
                'speed_range': max(speeds) - min(speeds)
            })
        
        return stats