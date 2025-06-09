"""
Core speedometer analysis functionality using Gemini AI
"""

import google.generativeai as genai
import time
import re
import asyncio
import concurrent.futures
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
        self.model_name = model_name
        self.results = []
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = asyncio.Lock()
        
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
            
            # Track API usage
            self.api_calls += 1
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            
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
                           progress_callback=None,
                           max_workers: int = 5) -> List[Dict]:
        """
        Analyze all frames in a directory with parallel processing
        
        Args:
            frames_dir: Directory containing frame images
            fps: Frames per second of extraction
            delay_seconds: Delay between API calls (ignored in parallel mode)
            progress_callback: Optional callback for progress updates
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of analysis results
        """
        frame_files = sorted(frames_dir.glob("*.png"))
        
        if not frame_files:
            raise ValueError(f"No PNG frames found in {frames_dir}")
        
        # Use parallel processing for better performance
        if max_workers > 1:
            return self._analyze_frames_parallel(frame_files, fps, progress_callback, max_workers)
        else:
            return self._analyze_frames_sequential(frame_files, fps, delay_seconds, progress_callback)
    
    def _analyze_frames_sequential(self, frame_files, fps, delay_seconds, progress_callback):
        """Sequential frame analysis (original method)"""
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
                'success': speed is not None,
                'interpolated': False,
                'smoothed': False,
                'anomaly_corrected': False
            }
            
            results.append(result)
            
            # Rate limiting
            if delay_seconds > 0 and i < len(frame_files) - 1:
                time.sleep(delay_seconds)
        
        self.results = results
        return results
    
    def _analyze_frames_parallel(self, frame_files, fps, progress_callback, max_workers):
        """Parallel frame analysis using ThreadPoolExecutor"""
        results = [None] * len(frame_files)
        completed_count = 0
        
        def analyze_single_frame(args):
            i, frame_path = args
            timestamp = i / fps
            
            speed, response = self.analyze_frame(frame_path)
            
            return i, {
                'frame': i + 1,
                'timestamp': timestamp,
                'filename': frame_path.name,
                'speed': speed,
                'response': response,
                'success': speed is not None,
                'interpolated': False,
                'smoothed': False,
                'anomaly_corrected': False
            }
        
        # Process frames in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(analyze_single_frame, (i, frame_path)): i 
                for i, frame_path in enumerate(frame_files)
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    i, result = future.result()
                    results[i] = result
                    completed_count += 1
                    
                    if progress_callback:
                        progress_callback(completed_count, len(frame_files), result['filename'])
                        
                except Exception as e:
                    # Handle individual frame failures
                    i = future_to_index[future]
                    results[i] = {
                        'frame': i + 1,
                        'timestamp': i / fps,
                        'filename': frame_files[i].name,
                        'speed': None,
                        'response': f"Error: {str(e)}",
                        'success': False,
                        'interpolated': False,
                        'smoothed': False,
                        'anomaly_corrected': False
                    }
                    completed_count += 1
                    
                    if progress_callback:
                        progress_callback(completed_count, len(frame_files), frame_files[i].name)
        
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
    
    def get_cost_info(self) -> Dict:
        """
        Calculate the cost of API usage based on Gemini pricing
        
        Returns:
            Dictionary with cost breakdown
        """
        # Gemini pricing (as of 2024-2025)
        # Prices per 1M tokens
        
        pricing = {
            'gemini-1.5-flash': {
                'input_per_1m': 0.075,
                'output_per_1m': 0.30
            },
            'gemini-1.5-pro': {
                'input_per_1m': 3.50,
                'output_per_1m': 10.50
            },
            'gemini-2.0-flash-exp': {
                'input_per_1m': 0.0375,  # Experimental pricing - may be free or very low cost
                'output_per_1m': 0.15
            }
        }
        
        model_pricing = pricing.get(self.model_name, pricing['gemini-1.5-flash'])
        
        input_cost = (self.total_input_tokens / 1_000_000) * model_pricing['input_per_1m']
        output_cost = (self.total_output_tokens / 1_000_000) * model_pricing['output_per_1m']
        total_cost = input_cost + output_cost
        
        return {
            'api_calls': self.api_calls,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'total_cost_usd': total_cost,
            'model': self.model_name
        }