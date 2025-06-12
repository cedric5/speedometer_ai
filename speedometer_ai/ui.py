"""
Streamlit web UI for Speedometer AI
"""

import streamlit as st
import streamlit.components.v1
import os
import tempfile
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from streamlit_cropper import st_cropper
from PIL import Image
from speedometer_ai.core import SpeedometerAnalyzer, QuotaExceededError
from speedometer_ai.utils import (
    extract_frames_from_video, 
    save_results_to_csv, 
    validate_video_file,
    check_ffmpeg_available,
    apply_video_crop,
    apply_opencv_speedometer_tracking,
    apply_speed_overlay
)



def load_secrets():
    """Load API keys from secrets.json file"""
    secrets_file = Path(__file__).parent.parent / "secrets.json"
    if secrets_file.exists():
        try:
            with open(secrets_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading secrets.json: {e}")
            return {}
    return {}


def get_api_key_from_sources():
    """Get API key from secrets file, environment variable, or session state"""
    # First try secrets file
    secrets = load_secrets()
    if secrets.get('gemini_api_key'):
        return secrets['gemini_api_key']
    
    # Then try environment variable
    env_key = os.getenv('GEMINI_API_KEY', '')
    if env_key:
        return env_key
    
    # Finally try session state
    return st.session_state.get('persistent_api_key', '')


def render_upload_section(api_key, fps, delay, model, parallel_workers, anomaly_detection, max_acceleration, interpolate_gaps, keep_frames, verbose):
    """Render the upload section with all upload and cropping functionality"""
    
    has_results = 'analysis_results' in st.session_state
    
    if has_results:
        # Show compact summary when in expander
        results_data = st.session_state.analysis_results
        if results_data:
            video_name = results_data.get('video_name', 'Unknown')
            st.caption(f"✅ Analyzed: {video_name}")
        else:
            st.caption("No analysis results available")
    
    uploaded_file = st.file_uploader(
        "Choose a dashboard video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video showing your car's dashboard",
        key="video_upload"
    )
    
    if uploaded_file is not None:
        # Check if this is a new video file and reset state if needed
        if 'current_video_name' not in st.session_state or st.session_state.current_video_name != uploaded_file.name:
            # Reset all session state when new video is uploaded
            st.session_state.current_video_name = uploaded_file.name
            st.session_state.processed_video_path = None
            st.session_state.analysis_results = None
            st.session_state.processing_complete = False
            st.session_state.analysis_complete = False
            # Reset crop values to defaults
            if 'crop_width' in st.session_state:
                del st.session_state.crop_width
            if 'crop_height' in st.session_state:
                del st.session_state.crop_height
            if 'crop_x_pos' in st.session_state:
                del st.session_state.crop_x_pos
            if 'crop_y_pos' in st.session_state:
                del st.session_state.crop_y_pos
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = Path(tmp_file.name)
            
        # Store original video path for overlay functionality
        st.session_state.original_video_path = str(temp_video_path)
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Single column layout for cropping controls with integrated preview
        # Add video processing functionality
        st.subheader("🎯 Video Processing (Optional)")
        st.caption("Process the video to focus on the speedometer area for better accuracy and faster processing")
        
        processing_mode = st.selectbox(
                "Processing Mode",
                ["None", "Basic Crop", "Ultra Tracking"],
                index=0,
                help="• None: Use original video\n• Basic Crop: Static crop to speedometer area\n• Ultra Tracking: Smart tracking that follows the speedometer area"
        )
        
        if processing_mode == "Ultra Tracking":
            st.info(f"ℹ️ **Smart Processing**: Will only process frames at {fps} FPS (same rate as AI analysis) for maximum efficiency")
        
        crop_video = processing_mode != "None"
        
        if crop_video:
            # Get video dimensions for cropping interface
            try:
                import cv2
                cap = cv2.VideoCapture(str(temp_video_path))
                ret, frame = cap.read()
                if ret:
                    original_height, original_width = frame.shape[:2]
                    cap.release()
                    
                    st.info(f"📐 Original video dimensions: {original_width} x {original_height} pixels")
                    
                    st.markdown("**🎯 Interactive Crop Area Selection**")
                    st.caption("Drag to select the speedometer area for cropping")
                    
                    # Create two columns for cropper and preview
                    cropper_col1, cropper_col2 = st.columns([1, 1])
                    
                    with cropper_col1:
                        st.markdown("**✂️ Crop Selection**")
                        # Convert frame to PIL Image for the cropper
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Use streamlit-cropper for interactive cropping
                        cropped_img = st_cropper(
                            pil_image,
                            realtime_update=True,
                            box_color='#FF0000',
                            aspect_ratio=None,  # Allow free aspect ratio
                            return_type='box'  # Return coordinates instead of cropped image
                        )
                    
                    # Extract coordinates from the cropper
                    if cropped_img is not None and 'left' in cropped_img:
                        crop_x1 = int(cropped_img['left'])
                        crop_y1 = int(cropped_img['top'])
                        crop_x2 = crop_x1 + int(cropped_img['width'])
                        crop_y2 = crop_y1 + int(cropped_img['height'])
                        
                        crop_width = crop_x2 - crop_x1
                        crop_height = crop_y2 - crop_y1
                    else:
                        # Default crop area if cropper hasn't been used yet
                        crop_width = min(400, original_width // 2)
                        crop_height = min(300, original_height // 2)
                        crop_x1 = original_width // 4
                        crop_y1 = original_height // 4
                        crop_x2 = crop_x1 + crop_width
                        crop_y2 = crop_y1 + crop_height
                    
                    # Show live crop preview in the second column
                    with cropper_col2:
                        st.markdown("**🔍 Live Crop Preview**")
                        
                        if crop_width > 0 and crop_height > 0:
                            # Extract the cropped area from the frame
                            try:
                                # Ensure coordinates are within bounds
                                crop_x1_safe = max(0, min(crop_x1, original_width - 1))
                                crop_y1_safe = max(0, min(crop_y1, original_height - 1))
                                crop_x2_safe = max(crop_x1_safe + 1, min(crop_x2, original_width))
                                crop_y2_safe = max(crop_y1_safe + 1, min(crop_y2, original_height))
                                
                                # Extract the cropped region
                                cropped_frame = frame[crop_y1_safe:crop_y2_safe, crop_x1_safe:crop_x2_safe]
                                
                                if cropped_frame.size > 0:
                                    # Save as temporary image for display
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_preview:
                                        cv2.imwrite(tmp_preview.name, cropped_frame)
                                        
                                        # Show the preview
                                        processing_caption = {
                                            "Basic Crop": "This area will be cropped from the video",
                                            "Ultra Tracking": "This area will be tracked and kept centered"
                                        }.get(processing_mode, "This area will be processed")
                                        
                                        st.image(tmp_preview.name, 
                                                caption=processing_caption, 
                                                use_container_width=True)
                                        
                                        # Show crop details
                                        st.caption(f"📏 {crop_width} × {crop_height} pixels")
                                else:
                                    st.warning("⚠️ Selected area is too small")
                            except Exception as e:
                                st.error(f"❌ Error creating preview: {str(e)}")
                        else:
                            st.info("👆 Select an area with the cropper to see preview")
                        
                    
                    # Display crop information below both columns
                    st.success(f"🎯 Selected crop area: {crop_width} x {crop_height} pixels at position ({crop_x1}, {crop_y1})")
                    
                    # Show crop details in columns
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.metric("Width", f"{crop_width}px")
                        st.metric("X Position", f"{crop_x1}px")
                    with info_col2:
                        st.metric("Height", f"{crop_height}px")
                        st.metric("Y Position", f"{crop_y1}px")
                    
                    # Apply processing based on selected mode
                    button_text = {
                        "Basic Crop": "✂️ Apply Basic Crop",
                        "Ultra Tracking": "🚀 Apply Ultra Tracking"
                    }.get(processing_mode, "✂️ Apply Processing")
                    
                    if st.button(button_text, type="primary"):
                            # Create progress indicators
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def progress_callback(progress: float, message: str):
                                progress_bar.progress(progress)
                                status_text.text(message)
                            
                            try:
                                if processing_mode == "Basic Crop":
                                    progress_callback(0.1, "✂️ Preparing video crop...")
                                    processed_video_path = apply_video_crop(
                                        temp_video_path, 
                                        crop_x1, crop_y1, crop_x2, crop_y2,
                                        progress_callback=progress_callback
                                    )
                                    success_msg = f"✅ Video cropped successfully! New dimensions: {crop_width} x {crop_height}"
                                        
                                elif processing_mode == "Ultra Tracking":
                                    progress_callback(0.05, f"🎯 Preparing OpenCV tracking at {fps} FPS...")
                                    processed_video_path = apply_opencv_speedometer_tracking(
                                        temp_video_path,
                                        crop_x1, crop_y1, crop_x2, crop_y2,
                                        output_width=800, output_height=600,
                                        fps=fps,
                                        progress_callback=progress_callback
                                    )
                                    success_msg = f"✅ OpenCV tracking complete at {fps} FPS! Speedometer tracked and centered: 800x600"
                            
                                if processed_video_path:
                                    temp_video_path = processed_video_path  # Use processed video for analysis
                                    st.success(success_msg)
                                    
                                    # Store processing info and video path in session state
                                    st.session_state.processing_complete = True
                                    st.session_state.processed_video_path = str(processed_video_path)
                                    st.session_state.processing_mode = processing_mode
                                    st.session_state.crop_info = {
                                        'original_size': (original_width, original_height),
                                        'crop_area': (crop_x1, crop_y1, crop_x2, crop_y2),
                                        'processing_mode': processing_mode,
                                        'processed_size': (800, 600) if processing_mode != "Basic Crop" else (crop_width, crop_height)
                                    }
                                    
                                    # Clean up progress indicators
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    st.rerun()  # Refresh to show processed video in preview
                                else:
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error(f"❌ Failed to process video with {processing_mode}")
                                    
                            except Exception as e:
                                progress_bar.empty()
                                status_text.empty()
                                st.error(f"❌ Error during {processing_mode}: {str(e)}")
                else:
                    st.error("❌ Could not read video for cropping")
                    cap.release()
            except Exception as e:
                st.error(f"❌ Error setting up video cropping: {str(e)}")
        
        
        # Video preview section
        st.markdown("---")
        st.subheader("📹 Video Preview")
        
        # Show processed video if available, otherwise show original
        if st.session_state.get('processing_complete', False) and 'processed_video_path' in st.session_state:
            processing_mode = st.session_state.get('processing_mode', 'Basic Crop')
            mode_captions = {
                'Basic Crop': '🎬 Showing cropped video',
                'Ultra Tracking': '🚀 Showing tracked video'
            }
            st.caption(mode_captions.get(processing_mode, '🎬 Showing processed video'))
            with open(st.session_state.processed_video_path, 'rb') as video_file:
                st.video(video_file.read(), start_time=0)
        else:
            st.caption("🎬 Showing original video")
            st.video(uploaded_file, start_time=0)
            
        # Add JavaScript to enable video time control
        st.components.v1.html("""
        <script>
        // Function to jump video to specific timestamp
        function jumpVideoToTime(timestamp) {
            // Find the video element in Streamlit
            const videos = document.querySelectorAll('video');
            if (videos.length > 0) {
                const video = videos[0]; // Get the first (main) video element
                if (video) {
                    video.currentTime = timestamp;
                    console.log('Video jumped to:', timestamp, 'seconds');
                    
                    // Optional: Show visual feedback
                    video.style.border = '3px solid #ff4444';
                    setTimeout(() => {
                        video.style.border = 'none';
                    }, 1000);
                }
            } else {
                console.log('No video element found');
            }
        }
        
        // Make function globally available
        window.jumpVideoToTime = jumpVideoToTime;
        
        // Also provide alternative names for compatibility
        window.jumpToTime = jumpVideoToTime;
        
        console.log('Video control functions loaded');
        </script>
        """, height=0)
        
        # Validation
        if not validate_video_file(temp_video_path):
            st.error("❌ Invalid video file")
            return
        
        if not check_ffmpeg_available():
            st.error("❌ FFmpeg not found. Please install FFmpeg.")
            return
        
        if not api_key:
            st.error("❌ Please enter your Gemini API key in the sidebar")
            return
        
        # Cost estimation
        if uploaded_file is not None:
            # Estimate video duration and frames using appropriate video source
            try:
                import cv2
                video_for_estimation = Path(st.session_state.processed_video_path) if st.session_state.get('processing_complete', False) and 'processed_video_path' in st.session_state else temp_video_path
                cap = cv2.VideoCapture(str(video_for_estimation))
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / video_fps if video_fps > 0 else 0
                cap.release()
                
                estimated_frames = int(duration * fps)
                # Use model-specific cost estimation
                cost_estimate = SpeedometerAnalyzer.estimate_cost(
                    estimated_frames, model, include_ai_analysis=False
                )
                
                cost_info = f"📊 **Estimated Analysis**: {estimated_frames} frames (~{duration:.1f}s video) | **Est. Cost**: ${cost_estimate['total_cost_usd']:.4f} USD ({model})"
                if anomaly_detection or interpolate_gaps:
                    cost_info += " | Plus rule-based post-processing (no additional cost)"
                st.info(cost_info)
            except:
                st.warning("⚠️ Could not estimate video duration")
        
        # Analysis button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            # Use cropped video for analysis if available
            video_to_analyze = Path(st.session_state.processed_video_path) if st.session_state.get('processing_complete', False) and 'processed_video_path' in st.session_state else temp_video_path
            analyze_video(video_to_analyze, api_key, model, fps, delay, parallel_workers, anomaly_detection, max_acceleration, interpolate_gaps, keep_frames, verbose)


def main():
    st.set_page_config(
        page_title="Speedometer AI",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Speedometer AI")
    st.markdown("AI-powered speedometer reading from dashboard video")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input with multiple sources
        # Initialize from multiple sources: secrets file > environment > session state
        if 'persistent_api_key' not in st.session_state:
            st.session_state.persistent_api_key = get_api_key_from_sources()
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            value=st.session_state.persistent_api_key,
            help="API key sources (in priority order):\n1. secrets.json file\n2. GEMINI_API_KEY environment variable\n3. Manual input (persists during session)",
            key="gemini_api_key"
        )
        
        # Update session state when API key changes
        if api_key != st.session_state.persistent_api_key:
            st.session_state.persistent_api_key = api_key
        
        # Show status about API key storage with source info
        secrets = load_secrets()
        env_key = os.getenv('GEMINI_API_KEY', '')
        
        if api_key:
            if secrets.get('gemini_api_key') and api_key == secrets['gemini_api_key']:
                st.success("🔑 API key loaded from secrets.json")
            elif env_key and api_key == env_key:
                st.success("🔑 API key loaded from environment variable")
            else:
                st.success("🔑 API key set manually (persists during session)")
        else:
            st.info("💡 API key sources:\n• Create secrets.json file (recommended)\n• Set GEMINI_API_KEY environment variable\n• Enter manually above")
        
        # Analysis settings
        st.subheader("Analysis Settings")
        fps = st.slider("Frames per second", 1.0, 10.0, 3.0, 0.5)
        delay = st.slider("API delay (seconds)", 0.5, 3.0, 1.0, 0.1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.subheader("AI Model")
            model = st.selectbox("Gemini Model", 
                               ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
                               index=0,
                               help="Choose the Gemini model:\n• Flash: Fastest & cheapest ($0.075/$0.30 per 1M tokens)\n• Pro: Most accurate but expensive ($1.25/$5.00 per 1M tokens)\n• 2.0 Experimental: Latest features, FREE during experimental period but very low quota limits")
            
            parallel_workers = st.slider("Parallel workers", 1, 20, 3, 1, 
                                        help="Number of parallel API calls (higher = faster but more load)")
            
            st.subheader("Rule-Based Data Processing")
            anomaly_detection = st.checkbox("Anomaly detection & correction", True,
                                           help="Use rule-based algorithms to detect and correct speed misreadings (e.g., OCR digit confusion)")
            max_acceleration = st.slider("Max car acceleration (km/h/s)", 5.0, 30.0, 16.95, 0.05,
                                       help="Maximum realistic acceleration for your car - used for physics validation")
            interpolate_gaps = st.checkbox("Gap filling & smoothing", True,
                                         help="Use rule-based interpolation and smoothing to fill missing readings and ensure smooth speed lines")
            
            st.subheader("Output Options")
            keep_frames = st.checkbox("Keep extracted frames", False)
            verbose = st.checkbox("Verbose output", False)
            
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🎥 Prepare", "🎬 Apply"])
    
    # Check for existing results
    has_results = 'analysis_results' in st.session_state and st.session_state.analysis_results is not None
    
    with tab1:
        st.header("🎥 Prepare Video Analysis")
        st.markdown("Upload and process your dashboard video, then analyze speed data.")
        
        # Upload section
        render_upload_section(api_key, fps, delay, model, parallel_workers, anomaly_detection, max_acceleration, interpolate_gaps, keep_frames, verbose)
        
        # Show new analysis option when results are available
        if has_results:
            st.markdown("---")
            if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
                # Clear results to show upload section again
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                st.rerun()
        
        # Results section in Prepare tab
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        if has_results:
            display_results(st.session_state.analysis_results)
        else:
            st.info("Upload a video and click 'Analyze Video' to see results here")
    
    with tab2:
        st.header("🎬 Apply Speed Overlay")
        st.markdown("Overlay the analyzed speed data onto your original video as a digital speedometer.")
        
        if has_results:
            render_overlay_section()
        else:
            st.info("🐈 Please analyze a video in the Prepare tab first to use overlay features.")


def analyze_video(video_path: Path, api_key: str, model: str, fps: float, delay: float, parallel_workers: int, anomaly_detection: bool, max_acceleration: float, interpolate_gaps: bool, keep_frames: bool, verbose: bool):
    """Analyze the uploaded video"""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    cost_display = st.empty()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            frames_dir = temp_dir_path / "frames"
            
            # Step 1: Extract frames
            status_text.text("📹 Extracting frames from video...")
            progress_bar.progress(0.1)
            
            frame_files = extract_frames_from_video(video_path, frames_dir, fps)
            st.success(f"✅ Extracted {len(frame_files)} frames")
            
            # Step 2: Initialize analyzer
            status_text.text(f"🤖 Initializing Gemini AI analyzer ({model})...")
            progress_bar.progress(0.2)
            
            analyzer = SpeedometerAnalyzer(api_key, model_name=model)
            
            # Step 3: Analyze frames
            if parallel_workers > 1:
                status_text.text(f"🔍 Analyzing frames with {parallel_workers} parallel workers...")
            else:
                status_text.text("🔍 Analyzing frames with AI...")
            
            def progress_callback(current, total, filename):
                progress = 0.2 + (current / total) * 0.6
                progress_bar.progress(progress)
                
                # Get current cost info and update displays
                cost_info = analyzer.get_cost_info()
                cost_str = f"${cost_info['total_cost_usd']:.4f}" if cost_info['total_cost_usd'] > 0 else "<$0.0001"
                
                if parallel_workers > 1:
                    status_text.text(f"🔍 Completed {current}/{total}: {filename}")
                else:
                    status_text.text(f"🔍 Analyzing frame {current}/{total}: {filename}")
                
                # Update cost display in real-time
                cost_display.info(f"💰 **Real-time Cost**: {cost_str} USD | **API Calls**: {cost_info['api_calls']} | **Model**: {cost_info['model']}")
            
            try:
                results = analyzer.analyze_video_frames(
                    frames_dir, fps, delay, progress_callback, max_workers=parallel_workers
                )
            except QuotaExceededError as e:
                st.error(f"🚫 **Quota Exceeded**\n\n{str(e)}")
                st.info("💡 **Quick Fix**: Change the model to 'gemini-1.5-flash' and reduce parallel workers to 1-2 in the sidebar settings.")
                return
            except Exception as e:
                error_str = str(e)
                st.error(f"❌ **Analysis Failed**: {error_str}")
                
                # Provide specific guidance
                if "429" in error_str or "quota" in error_str.lower():
                    st.warning("💡 **Rate Limit Issue**: Reduce parallel workers and try gemini-1.5-flash model")
                elif "api" in error_str.lower() or "key" in error_str.lower():
                    st.warning("💡 **API Key Issue**: Check your API key is valid and has access to the selected model")
                elif "permission" in error_str.lower():
                    st.warning(f"💡 **Permission Issue**: Your API key may not have access to the {model} model")
                else:
                    st.warning("💡 **Troubleshooting**: Try gemini-1.5-flash model with 1 parallel worker")
                return
            
            # Step 4: Post-process results
            status_text.text("🔧 Post-processing results...")
            progress_bar.progress(0.8)
            
            raw_success_rate = len([r for r in results if r['success']]) / len(results) * 100
            
            if anomaly_detection or interpolate_gaps:
                status_text.text("🔧 Rule-based analyzing speed data for anomalies and gaps...")
                try:
                    results = analyzer.analyze_speed_data_with_rules(results, max_acceleration)
                except Exception as e:
                    st.warning(f"⚠️ Rule-based analysis failed: {e}")
                    # Manual fallback if the integrated method fails
                    if anomaly_detection:
                        status_text.text("🔧 Detecting and correcting anomalies...")
                        from .utils import detect_and_correct_anomalies
                        results = detect_and_correct_anomalies(results, max_change_per_second=max_acceleration)
                    
                    if interpolate_gaps:
                        status_text.text("🔧 Filling gaps using interpolation...")
                        from .utils import interpolate_missing_speeds
                        results = interpolate_missing_speeds(results, max_gap_size=3)
            
            # Show improvement if processing was applied
            if anomaly_detection or interpolate_gaps:
                processed_success_rate = len([r for r in results if r['success']]) / len(results) * 100
                corrected_count = len([r for r in results if r.get('anomaly_corrected', False)])
                interpolated_count = len([r for r in results if r.get('interpolated', False)])
                smoothed_count = len([r for r in results if r.get('smoothed', False)])
                
                if processed_success_rate > raw_success_rate:
                    improvement = processed_success_rate - raw_success_rate
                    st.info(f"✨ Rule-based processing improved success rate by {improvement:.1f}% ({raw_success_rate:.1f}% → {processed_success_rate:.1f}%)")
                
                if corrected_count > 0:
                    st.success(f"⚠️ Corrected {corrected_count} anomalous readings")
                if interpolated_count > 0:
                    st.success(f"🔮 Interpolated {interpolated_count} missing values")
                if smoothed_count > 0:
                    st.success(f"📈 Smoothed {smoothed_count} outlier readings")
            
            # Step 5: Finalize results
            status_text.text("📊 Finalizing results...")
            progress_bar.progress(0.9)
            
            # Store results in session state
            st.session_state.analysis_results = {
                'results': results,
                'stats': analyzer.get_statistics(),
                'cost_info': analyzer.get_cost_info(),
                'video_name': video_path.name,
                'video_path': str(video_path)
            }
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Final cost display
            final_cost_info = analyzer.get_cost_info()
            final_cost_str = f"${final_cost_info['total_cost_usd']:.4f}" if final_cost_info['total_cost_usd'] > 0 else "<$0.0001"
            cost_display.success(f"💰 **Final Cost**: {final_cost_str} USD | **Total API Calls**: {final_cost_info['api_calls']}")
            
            st.success("🎉 Analysis completed successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        cost_display.empty()


def display_results(analysis_data):
    """Display analysis results"""
    
    results = analysis_data['results']
    stats = analysis_data['stats']
    cost_info = analysis_data.get('cost_info', {})
    video_name = analysis_data['video_name']
    video_path = analysis_data.get('video_path', None)
    
    # Summary metrics - displayed in a simple grid without nested columns
    st.subheader("📊 Summary Statistics")
    
    # Create metrics in rows instead of columns to avoid nesting issues
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", stats.get('total_frames', 0))
    
    with col2:
        st.metric("Duration", f"{stats.get('duration', 0):.1f}s")
    
    with col3:
        if 'min_speed' in stats:
            st.metric("Speed Range", f"{stats['min_speed']}-{stats['max_speed']} km/h")
        else:
            st.metric("Speed Range", "N/A")
    
    with col4:
        success_rate = stats.get('success_rate', 0)
        # Calculate breakdown
        ai_readings = len([r for r in results if r['success'] and not r.get('interpolated', False) and not r.get('anomaly_corrected', False)])
        interpolated_count = len([r for r in results if r.get('interpolated', False)])
        anomaly_corrected_count = len([r for r in results if r.get('anomaly_corrected', False)])
        
        st.metric("Success Rate", f"{success_rate:.1f}%", 
                 help=f"AI: {ai_readings}, Anomaly Corrected: {anomaly_corrected_count}, Interpolated: {interpolated_count}")
    
    # Cost information in a separate row
    if cost_info.get('total_cost_usd', 0) > 0:
        st.metric("Analysis Cost", f"${cost_info['total_cost_usd']:.4f}")
    
    # Create DataFrame for display
    df = pd.DataFrame(results)
    
    # Speed chart with timestamp selection
    if len(df[df['success'] == True]) > 0:
        st.subheader("📈 Interactive Speed Analysis")
        
        successful_df = df[df['success'] == True].copy()
        
        # Add timestamp selection functionality
        st.markdown("**⏰ Video Timestamp Navigation**")
        st.caption("Use the slider below to explore different timestamps. The chart will highlight the selected point with smooth, responsive updates.")
        
        max_time = successful_df['timestamp'].max() if len(successful_df) > 0 else 10.0
        
        # Timestamp selection
        selected_timestamp = st.slider(
            "Select timestamp to view:",
            min_value=0.0,
            max_value=float(max_time),
            value=0.0,
            step=0.01,  # Ultra-fine step for instant, smooth updates
            format="%.2fs",
            help="Drag the slider - graph updates instantly as you move it"
        )
        
        # Show current selection info
        if len(successful_df) > 0:
            closest_idx = (successful_df['timestamp'] - selected_timestamp).abs().idxmin()
            closest_speed = successful_df.loc[closest_idx, 'speed']
            st.info(f"**Selected Time**: {selected_timestamp:.1f}s | **Speed**: {closest_speed:.0f} km/h")
        
        # Video jumping functionality temporarily disabled for performance
        # Will be re-implemented later with a more efficient approach
        
        # Optimized Interactive chart with caching
        # Create base chart only once and cache it
        if 'base_chart_data' not in st.session_state or len(st.session_state.get('base_chart_data', {})) == 0:
            st.session_state.base_chart_data = {
                'timestamps': successful_df['timestamp'].tolist(),
                'speeds': successful_df['speed'].tolist(),
                'max_speed': successful_df['speed'].max(),
                'min_speed': successful_df['speed'].min()
            }
        
        chart_data = st.session_state.base_chart_data
        
        # Find closest point efficiently
        timestamps_array = chart_data['timestamps']
        closest_idx = min(range(len(timestamps_array)), 
                         key=lambda i: abs(timestamps_array[i] - selected_timestamp))
        selected_point_time = timestamps_array[closest_idx]
        selected_point_speed = chart_data['speeds'][closest_idx]
        
        # Create optimized figure with minimal traces
        fig = go.Figure()
        
        # Main speed line (single trace)
        fig.add_trace(go.Scatter(
            x=chart_data['timestamps'],
            y=chart_data['speeds'],
            mode='lines+markers',
            name='Speed',
            line=dict(width=2, color='#1f77b4'),
            marker=dict(size=4),
            hovertemplate='<b>%{x:.1f}s</b>: %{y:.0f} km/h<extra></extra>'
        ))
        
        # Selected point (single marker)
        fig.add_trace(go.Scatter(
            x=[selected_point_time],
            y=[selected_point_speed],
            mode='markers',
            name='Selected',
            marker=dict(size=12, color='red', symbol='circle-open', line=dict(width=2)),
            hovertemplate=f'<b>SELECTED</b><br>{selected_point_time:.1f}s: {selected_point_speed:.0f} km/h<extra></extra>'
        ))
        
        # Optimized layout
        fig.update_layout(
            title=dict(text="Speed Timeline", font=dict(size=16)),
            xaxis=dict(title="Time (s)", range=[0, max_time]),
            yaxis=dict(title="Speed (km/h)", range=[chart_data['min_speed']-5, chart_data['max_speed']+5]),
            hovermode='closest',
            height=350,
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=40),
            # Performance optimizations
            uirevision='chart_data',  # Prevents zoom reset
            dragmode='pan'  # Faster than zoom for timeline navigation
        )
        
        # Add vertical line as shape (faster than vline)
        fig.add_shape(
            type="line",
            x0=selected_timestamp, x1=selected_timestamp,
            y0=chart_data['min_speed']-5, y1=chart_data['max_speed']+5,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Render with maximum performance optimizations for instant updates
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            key="speed_chart",
            config={
                'displayModeBar': False,  # Hide toolbar for faster rendering
                'staticPlot': False,
                'responsive': True,
                'doubleClick': False,  # Disable double-click zoom for speed
                'showTips': False,  # Disable tooltips delay
                'displaylogo': False  # Remove plotly logo for cleaner UI
            }
        )
        
        # Show information about the selected timestamp
        if len(successful_df) > 0:
            closest_idx = (successful_df['timestamp'] - selected_timestamp).abs().idxmin()
            selected_data = successful_df.loc[closest_idx]
            
            st.info(
                f"🎯 **Selected Frame**: {selected_data['filename']} | "
                f"**Time**: {selected_data['timestamp']:.1f}s | "
                f"**Speed**: {selected_data['speed']:.0f} km/h"
            )
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Add legend
    st.caption("**Status Legend:** ✅ Rule-based Read = Direct reading | ⚠️ Corrected = Misread digit corrected | 🔮 Estimated = Value interpolated from surrounding data | ❌ Failed = No reading available")
    
    # Format the dataframe for display
    display_df = df.copy()
    
    def format_status(row):
        if row['success']:
            if row.get('interpolated', False):
                return '🔮 Estimated'
            elif row.get('anomaly_corrected', False):
                return '⚠️ Corrected'
            else:
                return '✅ Rule-based Read'
        else:
            return '❌ Failed'
    
    display_df['Status'] = display_df.apply(format_status, axis=1)
    display_df['Speed (km/h)'] = display_df['speed'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    display_df['Time (s)'] = display_df['timestamp'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_df[['Status', 'Time (s)', 'Speed (km/h)', 'filename']],
        use_container_width=True,
        hide_index=True
    )
    
    # Cost breakdown section
    if cost_info:
        st.subheader("💰 Cost Breakdown")
        
        # Display cost metrics in a simple layout
        st.text(f"Model: {cost_info.get('model', 'N/A')} | API Calls: {cost_info.get('api_calls', 0)}")
        st.text(f"Input Tokens: {cost_info.get('input_tokens', 0):,} | Output Tokens: {cost_info.get('output_tokens', 0):,}")
        st.text(f"Input Cost: ${cost_info.get('input_cost_usd', 0):.6f} | Output Cost: ${cost_info.get('output_cost_usd', 0):.6f}")
        
        if cost_info.get('total_cost_usd', 0) < 0.01:
            st.success("💡 Very affordable! Less than 1 cent.")
        elif cost_info.get('total_cost_usd', 0) < 0.10:
            st.info("💡 Very reasonable cost, less than 10 cents.")
        else:
            st.warning("⚠️ Consider optimizing frames/fps for cost reduction.")
    
    # Download section
    st.subheader("💾 Download Results")
    
    # CSV download
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📄 Download CSV",
        data=csv_data,
        file_name=f"{Path(video_name).stem}_speed_analysis.csv",
        mime="text/csv"
    )
    
    # Summary report
    report = generate_summary_report(results, stats, video_name, cost_info)
    st.download_button(
        label="📊 Download Report",
        data=report,
        file_name=f"{Path(video_name).stem}_analysis_report.txt",
        mime="text/plain"
    )


def generate_summary_report(results, stats, video_name, cost_info=None):
    """Generate a text summary report"""
    
    report = f"""SPEEDOMETER ANALYSIS REPORT
{'='*50}

Video: {video_name}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
• Total frames: {stats.get('total_frames', 0)}
• Successful readings: {stats.get('successful_readings', 0)}
• Success rate: {stats.get('success_rate', 0):.1f}%
• Duration: {stats.get('duration', 0):.1f} seconds
"""
    
    if 'min_speed' in stats:
        report += f"""• Speed range: {stats['min_speed']}-{stats['max_speed']} km/h
• Average speed: {stats['avg_speed']:.1f} km/h
"""
    
    if cost_info:
        report += f"""
COST BREAKDOWN:
• Model: {cost_info.get('model', 'N/A')}
• API calls: {cost_info.get('api_calls', 0)}
• Input tokens: {cost_info.get('input_tokens', 0):,}
• Output tokens: {cost_info.get('output_tokens', 0):,}
• Total cost: ${cost_info.get('total_cost_usd', 0):.6f} USD
"""
    
    report += f"""
DETAILED TIMELINE:
{'-'*30}
"""
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        speed_str = f"{result['speed']:3.0f}" if result['speed'] else "N/A"
        report += f"{status} {result['timestamp']:5.1f}s: {speed_str} km/h\n"
    
    report += f"""
{'-'*30}
Generated by Speedometer AI v1.0.0
"""
    
    return report


def render_overlay_section():
    """Render the overlay configuration and application section"""
    
    results_data = st.session_state.analysis_results
    # Use original video path for overlay, not the processed one
    original_video_path = st.session_state.get('original_video_path')
    results = results_data.get('results', [])
    
    if not original_video_path or not results:
        st.error("❌ No original video or analysis data available for overlay")
        return
    
    # Show info about which video will be used
    st.info(f"📹 **Overlay Target**: Original uploaded video (before any cropping/processing)")
    if st.session_state.get('processed_video_path'):
        st.caption("The speed data from the analyzed video will be overlaid onto your original video.")
    
    st.subheader("🎯 Overlay Configuration")
    
    # Two columns for overlay settings
    overlay_col1, overlay_col2 = st.columns([1, 1])
    
    with overlay_col1:
        st.markdown("**📍 Position Settings**")
        
        # Corner selection
        corner_position = st.selectbox(
            "Overlay Corner",
            ["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
            index=0,
            help="Choose which corner to place the speed overlay"
        )
        
        # Offset from corner
        x_offset = st.slider(
            "Horizontal Offset (px)",
            min_value=10,
            max_value=200,
            value=30,
            help="Distance from the edge of the video"
        )
        
        y_offset = st.slider(
            "Vertical Offset (px)",
            min_value=10,
            max_value=200,
            value=30,
            help="Distance from the edge of the video"
        )
    
    with overlay_col2:
        st.markdown("**🔤 Text Settings**")
        
        # Text size
        font_size = st.slider(
            "Font Size",
            min_value=20,
            max_value=120,
            value=60,
            help="Size of the speed text overlay"
        )
        
        # Text color
        text_color = st.selectbox(
            "Text Color",
            ["White", "Yellow", "Green", "Red", "Blue", "Black"],
            index=1,  # Default to Yellow
            help="Color of the speed text"
        )
        
        # Background
        show_background = st.checkbox(
            "Show Background",
            value=True,
            help="Add a semi-transparent background behind the text"
        )
        
        if show_background:
            bg_opacity = st.slider(
                "Background Opacity",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Transparency of the background"
            )
    
    # Preview section
    st.subheader("🔍 Preview")
    
    # Show a preview of how the overlay will look
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        preview_text = f"**{corner_position}**\n"
        preview_text += f"Font Size: {font_size}px\n"
        preview_text += f"Color: {text_color}\n"
        preview_text += f"Offset: {x_offset}px, {y_offset}px"
        
        st.info(f"👁️ Overlay Preview:\n\n{preview_text}")
        
        # Sample speed display
        if text_color == "White":
            color_code = "#FFFFFF"
        elif text_color == "Yellow":
            color_code = "#FFFF00"
        elif text_color == "Green":
            color_code = "#00FF00"
        elif text_color == "Red":
            color_code = "#FF0000"
        elif text_color == "Blue":
            color_code = "#0000FF"
        else:  # Black
            color_code = "#000000"
        
        # Show sample speed text
        sample_html = f"""
        <div style="
            font-size: {font_size//3}px; 
            color: {color_code}; 
            font-family: 'Courier New', monospace; 
            font-weight: bold;
            text-align: center;
            {'background-color: rgba(0,0,0,' + str(bg_opacity) + '); padding: 10px; border-radius: 5px;' if show_background else ''}
        ">
            85 km/h
        </div>
        """
        st.markdown(sample_html, unsafe_allow_html=True)
    
    # Apply overlay button
    st.markdown("---")
    
    if st.button("🎬 Apply Speed Overlay to Video", type="primary", use_container_width=True):
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def overlay_progress_callback(progress: float, message: str):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            overlay_progress_callback(0.1, "🎬 Starting video overlay process...")
            
            # Apply overlay using the configured settings
            overlay_video_path = apply_speed_overlay(
                video_path=original_video_path,
                results=results,
                corner_position=corner_position,
                x_offset=x_offset,
                y_offset=y_offset,
                font_size=font_size,
                text_color=text_color,
                show_background=show_background,
                bg_opacity=bg_opacity if show_background else 0,
                progress_callback=overlay_progress_callback
            )
            
            if overlay_video_path:
                st.success(f"✅ Speed overlay applied successfully!")
                
                # Store overlay video path in session state
                st.session_state.overlay_video_path = str(overlay_video_path)
                
                # Show download button
                with open(overlay_video_path, 'rb') as video_file:
                    st.download_button(
                        label="💾 Download Overlay Video",
                        data=video_file.read(),
                        file_name=f"speed_overlay_{Path(original_video_path).name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # Show video preview
                st.subheader("🎬 Overlay Video Preview")
                with open(overlay_video_path, 'rb') as video_file:
                    st.video(video_file.read())
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Failed to apply speed overlay")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during overlay: {str(e)}")


if __name__ == "__main__":
    main()