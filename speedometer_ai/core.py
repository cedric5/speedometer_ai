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
        
        # Extract numbers (1-3 digits to include all valid speeds)
        numbers = re.findall(r'\b\d{1,3}\b', cleaned)
        
        # Filter for reasonable speeds
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num <= 400:  # Reasonable speed range (including low speeds and high-speed driving)
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
    
    def analyze_speed_data_with_rules(self, results: List[Dict], max_acceleration: float = 16.95) -> List[Dict]:
        """
        Use rule-based algorithms to correct speed data anomalies and fill gaps
        
        IMPORTANT: The first frame (index 0) is NEVER modified - it must always 
        be the original AI reading to establish the baseline speed.
        
        Args:
            results: List of analysis results with speed readings
            max_acceleration: Maximum realistic acceleration for the car (km/h/s)
            
        Returns:
            Rule-corrected results with anomalies fixed and gaps filled
        """
        if not results or len(results) < 3:
            return results
        
        # Count missing data before processing
        missing_count = sum(1 for r in results if not r['success'] or r['speed'] is None)
        total_count = len(results)
        
        print(f"\n🔧 Rule-based analyzing speed data for anomalies and gaps...")
        print(f"   • Total frames: {total_count}")
        print(f"   • Missing/unclear readings: {missing_count}")
        print(f"   • Success rate before processing: {((total_count - missing_count) / total_count * 100):.1f}%")
        print(f"   • Physics constraint: max {max_acceleration} km/h/s acceleration")
        
        # Import rule-based processing functions from utils
        from .utils import detect_and_correct_anomalies, interpolate_missing_speeds
        
        print(f"   • Detecting and correcting anomalies...")
        # Step 1: Detect and correct obvious anomalies (OCR misreads)
        corrected_results = detect_and_correct_anomalies(results, max_acceleration)
        
        print(f"   • Interpolating missing values...")
        # Step 2: Interpolate missing speed readings
        corrected_results = interpolate_missing_speeds(corrected_results, max_gap_size=5)
        
        print(f"   • Filling remaining gaps...")
        # Step 3: Fill any remaining gaps with estimates
        corrected_results = self._fill_remaining_gaps(corrected_results)
        
        print(f"   • Applying final smoothing...")
        # Step 4: Apply smoothing to ensure smooth speed line
        corrected_results = self._apply_smoothing(corrected_results)
        
        # Count and report the results
        anomaly_corrections = sum(1 for r in corrected_results if r.get('anomaly_corrected', False))
        interpolations = sum(1 for r in corrected_results if r.get('interpolated', False))
        smoothed = sum(1 for r in corrected_results if r.get('smoothed', False))
        final_missing = sum(1 for r in corrected_results if not r['success'] or r['speed'] is None)
        final_success_rate = ((len(corrected_results) - final_missing) / len(corrected_results) * 100)
        original_success_rate = ((len(corrected_results) - missing_count) / len(corrected_results) * 100)
        improvement = final_success_rate - original_success_rate
        
        print(f"\n✅ Rule-based analysis complete:")
        print(f"   • Anomaly corrections: {anomaly_corrections}")
        print(f"   • Gap interpolations: {interpolations}")
        print(f"   • Smoothed outliers: {smoothed}")
        print(f"   • Remaining missing: {final_missing}")
        print(f"   • Final success rate: {final_success_rate:.1f}% (improved by {improvement:.1f}%)")
        
        return corrected_results
    
    
    def _fill_remaining_gaps(self, results: List[Dict]) -> List[Dict]:
        """
        Fill any remaining gaps with interpolated or estimated values
        Ensures no empty cells in the final output
        """
        filled_results = [result.copy() for result in results]
        
        # Find all successful speed readings for reference
        valid_speeds = []
        valid_indices = []
        for i, result in enumerate(filled_results):
            if result['success'] and result['speed'] is not None:
                valid_speeds.append(result['speed'])
                valid_indices.append(i)
        
        if not valid_speeds:
            # If no valid speeds at all, use a default highway speed
            default_speed = 120
            for result in filled_results:
                if result['speed'] is None:
                    result['speed'] = default_speed
                    result['success'] = True
                    result['interpolated'] = True
                    result['response'] = f"FALLBACK ESTIMATE: No valid speeds available, used default {default_speed} km/h | Original: {result['response']}"
            return filled_results
        
        # Calculate average speed for fallback
        avg_speed = int(sum(valid_speeds) / len(valid_speeds))
        
        # Fill gaps by finding nearest valid readings (NEVER fill the first frame)
        for i, result in enumerate(filled_results):
            # Skip the first frame - it must always be the original AI reading
            if i == 0:
                continue
                
            if result['speed'] is None:
                # Find nearest valid speeds before and after
                prev_speed = None
                next_speed = None
                
                # Look backwards
                for j in range(i - 1, -1, -1):
                    if filled_results[j]['speed'] is not None:
                        prev_speed = filled_results[j]['speed']
                        break
                
                # Look forwards
                for j in range(i + 1, len(filled_results)):
                    if filled_results[j]['speed'] is not None:
                        next_speed = filled_results[j]['speed']
                        break
                
                # Estimate speed based on available data
                if prev_speed is not None and next_speed is not None:
                    # Interpolate between prev and next
                    estimated_speed = int((prev_speed + next_speed) / 2)
                elif prev_speed is not None:
                    # Use previous speed
                    estimated_speed = prev_speed
                elif next_speed is not None:
                    # Use next speed
                    estimated_speed = next_speed
                else:
                    # Use average speed
                    estimated_speed = avg_speed
                
                # Apply the estimate
                filled_results[i]['speed'] = estimated_speed
                filled_results[i]['success'] = True
                filled_results[i]['interpolated'] = True
                filled_results[i]['response'] = f"GAP FILLED: Estimated {estimated_speed} km/h based on surrounding data | Original: {result['response']}"
        
        return filled_results
    
    def _apply_smoothing(self, results: List[Dict], max_difference: float = 4.0, window_size: int = 5) -> List[Dict]:
        """
        Apply aggressive smoothing to ensure a very smooth speed line
        Uses a combination of outlier detection and moving average smoothing
        
        Args:
            results: List of analysis results
            max_difference: Maximum allowed difference between consecutive readings (km/h)
            window_size: Size of the smoothing window for moving average
            
        Returns:
            Aggressively smoothed results
        """
        smoothed_results = [result.copy() for result in results]
        smoothed_count = 0
        
        # Phase 1: Aggressive outlier detection and correction
        for pass_num in range(5):  # More passes for aggressive smoothing
            changes_made = False
            
            # Process frames from index 1 to n-2 (NEVER smooth the first frame)
            for i in range(1, len(smoothed_results) - 1):
                current = smoothed_results[i]
                prev_result = smoothed_results[i - 1] 
                next_result = smoothed_results[i + 1]
                
                # Only process if all three have valid speeds
                if (current['speed'] is not None and 
                    prev_result['speed'] is not None and 
                    next_result['speed'] is not None):
                    
                    current_speed = current['speed']
                    prev_speed = prev_result['speed']
                    next_speed = next_result['speed']
                    
                    # More aggressive: check difference with either neighbor (not both)
                    diff_prev = abs(current_speed - prev_speed)
                    diff_next = abs(current_speed - next_speed)
                    
                    # Apply smoothing if it differs significantly from either neighbor
                    if diff_prev > max_difference or diff_next > max_difference:
                        # Calculate expected speed based on trend
                        expected_speed = int((prev_speed + next_speed) / 2)
                        
                        # More aggressive threshold for smoothing
                        if abs(expected_speed - current_speed) > 2:  # Lower threshold
                            original_speed = current_speed
                            smoothed_results[i]['speed'] = expected_speed
                            smoothed_results[i]['smoothed'] = True
                            
                            # Update response to indicate smoothing
                            original_response = current['response']
                            smoothed_results[i]['response'] = f"SMOOTHED: Original {original_speed} km/h → {expected_speed} km/h to ensure smooth speed line | Original: {original_response}"
                            
                            smoothed_count += 1
                            changes_made = True
            
            # Stop if no changes were made in this pass
            if not changes_made:
                break
        
        # Phase 2: Moving average smoothing for extra smoothness
        moving_avg_count = 0
        if window_size >= 3:
            half_window = window_size // 2
            
            # Moving average from half_window to end-half_window (protects first frame)
            for i in range(max(1, half_window), len(smoothed_results) - half_window):
                current = smoothed_results[i]
                
                if current['speed'] is not None:
                    # Calculate moving average
                    window_speeds = []
                    for j in range(i - half_window, i + half_window + 1):
                        if (j < len(smoothed_results) and 
                            smoothed_results[j]['speed'] is not None):
                            window_speeds.append(smoothed_results[j]['speed'])
                    
                    if len(window_speeds) >= 3:  # Need at least 3 points
                        moving_avg = int(sum(window_speeds) / len(window_speeds))
                        
                        # Apply moving average if it smooths out variations
                        if abs(moving_avg - current['speed']) > 2:
                            original_speed = current['speed']
                            smoothed_results[i]['speed'] = moving_avg
                            smoothed_results[i]['smoothed'] = True
                            
                            # Update response
                            original_response = current['response']
                            smoothed_results[i]['response'] = f"SMOOTHED: Original {original_speed} km/h → {moving_avg} km/h (moving average) | Original: {original_response}"
                            
                            moving_avg_count += 1
        
        total_smoothed = smoothed_count + moving_avg_count
        if total_smoothed > 0:
            print(f"     • Aggressively smoothed {total_smoothed} readings ({smoothed_count} outliers + {moving_avg_count} moving avg)")
        
        return smoothed_results
    
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
                'input_per_1m': 0.075,    # $0.075 per 1M input tokens
                'output_per_1m': 0.30     # $0.30 per 1M output tokens
            },
            'gemini-1.5-pro': {
                'input_per_1m': 1.25,     # $1.25 per 1M input tokens
                'output_per_1m': 5.00     # $5.00 per 1M output tokens
            },
            'gemini-2.0-flash-exp': {
                'input_per_1m': 0.0,      # Free during experimental period
                'output_per_1m': 0.0      # Free during experimental period
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