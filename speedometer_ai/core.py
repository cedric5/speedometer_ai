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


class QuotaExceededError(Exception):
    """Custom exception for API quota exceeded errors"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


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
            error_str = str(e)
            
            # Check for quota exceeded errors (429)
            if "429" in error_str or "quota" in error_str.lower() or "exceeded" in error_str.lower():
                # Extract retry delay if available
                retry_after = None
                if "retry_delay" in error_str:
                    try:
                        import json
                        # Try to extract retry delay from error message
                        if "seconds:" in error_str:
                            parts = error_str.split("seconds:")
                            if len(parts) > 1:
                                retry_after = int(parts[1].split()[0])
                    except:
                        pass
                
                # Create user-friendly quota error message
                if "gemini-2.0-flash-exp" in error_str:
                    quota_msg = "❌ Quota exceeded for Gemini 2.0 Flash Experimental model. This model has very low rate limits. Please try:\n" \
                               "   • Switch to 'gemini-1.5-flash' model (much higher quota)\n" \
                               "   • Reduce parallel workers (--parallel 1 or 2)\n" \
                               "   • Wait a few minutes and try again"
                else:
                    quota_msg = f"❌ API quota exceeded for model '{self.model_name}'. Please try:\n" \
                               f"   • Reduce parallel workers (--parallel 1 or 2)\n" \
                               f"   • Wait a few minutes and try again\n" \
                               f"   • Check your API quota limits at https://ai.google.dev/gemini-api/docs/rate-limits"
                
                if retry_after:
                    quota_msg += f"\n   • Suggested retry delay: {retry_after} seconds"
                
                raise QuotaExceededError(quota_msg, retry_after)
            
            return None, f"Error: {error_str}"
    
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
                    # Check if this is a quota error that should stop everything
                    if isinstance(e, QuotaExceededError) or "429" in str(e) or "quota" in str(e).lower():
                        # Cancel all remaining futures
                        for remaining_future in future_to_index:
                            remaining_future.cancel()
                        raise e  # Re-raise quota error to stop analysis
                    
                    # Handle other individual frame failures
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
    
    def analyze_speed_data_with_ai(self, results: List[Dict], max_acceleration: float = 16.95) -> List[Dict]:
        """
        Use AI to analyze and correct speed data anomalies and fill gaps
        
        Args:
            results: List of analysis results with speed readings
            max_acceleration: Maximum realistic acceleration for the car (km/h/s)
            
        Returns:
            AI-corrected results with anomalies fixed and gaps filled
        """
        if not results or len(results) < 3:
            return results
            
        # Prepare speed data for AI analysis
        speed_data = []
        for i, result in enumerate(results):
            speed_data.append({
                'timestamp': result['timestamp'],
                'speed': result['speed'] if result['success'] else None,
                'filename': result['filename'],
                'original_response': result['response']
            })
        
        # Create AI prompt for speed data analysis
        prompt = self._create_speed_analysis_prompt(speed_data, max_acceleration)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Track API usage
            self.api_calls += 1
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            # Parse AI response and apply corrections
            corrected_results = self._parse_speed_corrections(results, response_text, max_acceleration)
            return corrected_results
            
        except Exception as e:
            # If AI analysis fails, return original results
            print(f"Warning: AI speed analysis failed: {e}")
            return results
    
    def _create_speed_analysis_prompt(self, speed_data: List[Dict], max_acceleration: float) -> str:
        """Create prompt for AI speed data analysis"""
        
        # Format speed data with detailed timing analysis
        data_text = "\nSpeed readings with timing analysis:\n"
        for i, item in enumerate(speed_data):
            speed_str = f"{item['speed']} km/h" if item['speed'] is not None else "MISSING"
            data_text += f"  {item['timestamp']:.2f}s: {speed_str}"
            
            # Add timing context for physics validation
            if i > 0:
                time_diff = item['timestamp'] - speed_data[i-1]['timestamp']
                prev_speed = speed_data[i-1]['speed']
                curr_speed = item['speed']
                
                if prev_speed is not None and curr_speed is not None:
                    speed_change = abs(curr_speed - prev_speed)
                    max_allowed_change = max_acceleration * time_diff
                    data_text += f" (Δt={time_diff:.2f}s, Δv={speed_change:.1f}, max_allowed={max_allowed_change:.1f})"
                    if speed_change > max_allowed_change:
                        data_text += " ⚠️VIOLATION"
            data_text += "\n"
        
        prompt = f"""
TASK: Analyze car speedometer readings for physics violations and missing data, then provide corrections.

CRITICAL PHYSICS CONSTRAINTS:
- Maximum acceleration: {max_acceleration} km/h per second
- For ANY two consecutive readings with time difference Δt seconds:
  |speed_new - speed_old| MUST BE ≤ {max_acceleration} × Δt
- Example: If readings are 0.33s apart, max speed change = {max_acceleration} × 0.33 = {max_acceleration * 0.33:.1f} km/h

SPEED DATA WITH PHYSICS ANALYSIS:{data_text}

STRICT ANALYSIS RULES:
1. PHYSICS VIOLATION DETECTION:
   - Any speed change marked with ⚠️VIOLATION is physically impossible
   - These readings MUST be corrected to respect acceleration limits
   - Calculate the maximum physically possible speed for each timestamp
   - Prefer corrections that maintain realistic driving patterns

2. OCR ERROR PATTERNS (common misreads):
   - 0 ↔ 6, 8, 9 (circular shapes confused)
   - 1 ↔ 7 (vertical lines)
   - 2 ↔ 5, 8 (similar curves)
   - 3 ↔ 8 (partial recognition)
   - 5 ↔ 6, 8 (curve confusion)
   - 6 ↔ 8, 9 (circular confusion)
   - 8 ↔ 9 (nearly identical)

3. GAP FILLING:
   - For MISSING readings, calculate interpolated value
   - Ensure interpolation respects acceleration limits at EVERY point
   - Use linear interpolation but cap changes to max_acceleration × time_interval

4. VALIDATION STEPS:
   - After each correction, verify it doesn't create new physics violations
   - Ensure 50-300 km/h range (reasonable highway speeds)
   - Maintain smooth driving behavior where possible

RESPONSE FORMAT:
Provide corrections as JSON array. Each correction MUST include physics justification:
{{
  "timestamp": 12.34,
  "action": "ANOMALY_CORRECTION" or "INTERPOLATION",
  "original_speed": 90,
  "corrected_speed": 60,
  "reason": "Physics violation: 90→120 in 0.33s requires 90.9 km/h/s (exceeds {max_acceleration}). OCR likely misread 6→9. Corrected to 60 maintains realistic progression.",
  "physics_check": "Previous: 55 km/h, Next: 65 km/h, Time gaps: 0.33s each, Max changes: ±{max_acceleration * 0.33:.1f} km/h"
}}

IMPORTANT: Every corrected speed MUST respect acceleration limits with BOTH adjacent readings.

If no corrections needed, respond with: []

CORRECTIONS:"""

        return prompt
    
    def _parse_speed_corrections(self, original_results: List[Dict], ai_response: str, max_acceleration: float) -> List[Dict]:
        """Parse AI response and apply speed corrections"""
        import json
        import re
        
        corrected_results = [result.copy() for result in original_results]
        
        # Initialize flags for all results
        for result in corrected_results:
            if 'interpolated' not in result:
                result['interpolated'] = False
            if 'anomaly_corrected' not in result:
                result['anomaly_corrected'] = False
            if 'ai_analyzed' not in result:
                result['ai_analyzed'] = True
        
        try:
            # Extract JSON from AI response
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if not json_match:
                # Try to find corrections in a different format
                if "[]" in ai_response or "no corrections" in ai_response.lower():
                    return corrected_results
                else:
                    raise ValueError("No valid JSON found in AI response")
            
            corrections = json.loads(json_match.group())
            
            # Apply corrections
            for correction in corrections:
                timestamp = correction.get('timestamp')
                action = correction.get('action')
                corrected_speed = correction.get('corrected_speed')
                reason = correction.get('reason', 'AI correction')
                
                # Find the result with matching timestamp
                for i, result in enumerate(corrected_results):
                    if abs(result['timestamp'] - timestamp) < 0.01:  # Small tolerance for floating point
                        original_speed = result['speed']
                        
                        # VALIDATE: Check if correction respects acceleration limits with adjacent readings
                        if self._validate_speed_correction(corrected_results, i, corrected_speed, max_acceleration):
                            corrected_results[i]['speed'] = corrected_speed
                            corrected_results[i]['success'] = True
                            
                            if action == "ANOMALY_CORRECTION":
                                corrected_results[i]['anomaly_corrected'] = True
                                corrected_results[i]['response'] = f"AI ANOMALY CORRECTION: Original {original_speed} km/h → {corrected_speed} km/h. {reason} | Original: {result['response']}"
                            elif action == "INTERPOLATION":
                                corrected_results[i]['interpolated'] = True
                                corrected_results[i]['response'] = f"AI INTERPOLATION: Filled missing value with {corrected_speed} km/h. {reason} | Original: {result['response']}"
                        else:
                            # AI suggestion violates physics - reject it
                            print(f"Warning: AI suggested correction violates acceleration limits at {timestamp:.2f}s: {original_speed} → {corrected_speed}")
                        break
            
        except Exception as e:
            print(f"Warning: Failed to parse AI corrections: {e}")
            # Return original results if parsing fails
            return corrected_results
        
        return corrected_results
    
    def _validate_speed_correction(self, results: List[Dict], index: int, proposed_speed: float, max_acceleration: float) -> bool:
        """
        Validate that a proposed speed correction respects acceleration limits with adjacent readings
        
        Args:
            results: List of current results
            index: Index of the result being corrected
            proposed_speed: The speed value being proposed
            max_acceleration: Maximum allowed acceleration in km/h/s
            
        Returns:
            True if correction is physically valid, False otherwise
        """
        # Check with previous reading
        if index > 0:
            prev_result = results[index - 1]
            if prev_result['success'] and prev_result['speed'] is not None:
                time_diff = results[index]['timestamp'] - prev_result['timestamp']
                speed_change = abs(proposed_speed - prev_result['speed'])
                max_allowed_change = max_acceleration * time_diff
                
                if speed_change > max_allowed_change:
                    return False
        
        # Check with next reading
        if index < len(results) - 1:
            next_result = results[index + 1]
            if next_result['success'] and next_result['speed'] is not None:
                time_diff = next_result['timestamp'] - results[index]['timestamp']
                speed_change = abs(next_result['speed'] - proposed_speed)
                max_allowed_change = max_acceleration * time_diff
                
                if speed_change > max_allowed_change:
                    return False
        
        return True
    
    @staticmethod
    def get_model_pricing(model_name: str) -> Dict[str, float]:
        """
        Get pricing information for a specific model
        
        Args:
            model_name: Name of the Gemini model
            
        Returns:
            Dictionary with input_per_1m and output_per_1m pricing
        """
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
        
        return pricing.get(model_name, pricing['gemini-1.5-flash'])
    
    @staticmethod
    def estimate_cost(estimated_frames: int, model_name: str, include_ai_analysis: bool = True) -> Dict[str, float]:
        """
        Estimate the cost of analysis for a given number of frames
        
        Args:
            estimated_frames: Number of frames to analyze
            model_name: Gemini model to use
            include_ai_analysis: Whether to include cost of AI data analysis
            
        Returns:
            Dictionary with cost breakdown
        """
        pricing = SpeedometerAnalyzer.get_model_pricing(model_name)
        
        # Estimate tokens for speedometer reading
        estimated_input_tokens = estimated_frames * 1000  # ~1000 tokens per image
        estimated_output_tokens = estimated_frames * 10   # ~10 tokens per response
        
        # Add tokens for AI data analysis if enabled
        if include_ai_analysis:
            # Additional tokens for analyzing the speed dataset
            analysis_input_tokens = estimated_frames * 50  # ~50 tokens per data point in analysis
            analysis_output_tokens = max(100, estimated_frames * 5)  # Base 100 + ~5 per potential correction
            
            estimated_input_tokens += analysis_input_tokens
            estimated_output_tokens += analysis_output_tokens
        
        input_cost = (estimated_input_tokens / 1_000_000) * pricing['input_per_1m']
        output_cost = (estimated_output_tokens / 1_000_000) * pricing['output_per_1m']
        total_cost = input_cost + output_cost
        
        return {
            'estimated_frames': estimated_frames,
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_output_tokens': estimated_output_tokens,
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'total_cost_usd': total_cost,
            'model': model_name
        }

    def get_cost_info(self) -> Dict:
        """
        Calculate the cost of API usage based on Gemini pricing
        
        Returns:
            Dictionary with cost breakdown
        """
        # Use centralized pricing information
        model_pricing = self.get_model_pricing(self.model_name)
        
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