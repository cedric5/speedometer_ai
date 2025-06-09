"""
Streamlit web UI for Speedometer AI
"""

import streamlit as st
import streamlit.components.v1
import os
import tempfile
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from speedometer_ai.core import SpeedometerAnalyzer, QuotaExceededError
from speedometer_ai.utils import (
    extract_frames_from_video, 
    save_results_to_csv, 
    validate_video_file,
    check_ffmpeg_available
)


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
        
        # API Key input with persistent localStorage
        # Initialize API key from environment or localStorage simulation
        if 'api_key' not in st.session_state:
            st.session_state.api_key = os.getenv('GEMINI_API_KEY', '')
        
        # Create the API key input
        api_key_container = st.container()
        with api_key_container:
            api_key = st.text_input(
                "Gemini API Key", 
                type="password",
                value=st.session_state.api_key,
                help="Enter your Google Gemini API key (automatically saved in browser for future sessions)",
                key="gemini_api_key"
            )
        
        # Add localStorage JavaScript functionality
        st.components.v1.html(f"""
        <script>
        // Save API key to localStorage when it changes
        function saveApiKeyToStorage() {{
            const apiKeyInput = window.parent.document.querySelector('input[aria-label="Gemini API Key"]');
            if (apiKeyInput && apiKeyInput.value && apiKeyInput.value.trim()) {{
                localStorage.setItem('speedometer_ai_api_key', apiKeyInput.value.trim());
                console.log('API key saved to localStorage');
            }}
        }}
        
        // Load API key from localStorage on page load
        function loadApiKeyFromStorage() {{
            const storedKey = localStorage.getItem('speedometer_ai_api_key');
            if (storedKey && storedKey.trim()) {{
                const apiKeyInput = window.parent.document.querySelector('input[aria-label="Gemini API Key"]');
                if (apiKeyInput && !apiKeyInput.value) {{
                    apiKeyInput.value = storedKey;
                    apiKeyInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    console.log('API key loaded from localStorage');
                }}
            }}
        }}
        
        // Auto-save when user types (with debounce)
        let saveTimeout;
        function setupAutoSave() {{
            const apiKeyInput = window.parent.document.querySelector('input[aria-label="Gemini API Key"]');
            if (apiKeyInput) {{
                apiKeyInput.addEventListener('input', function() {{
                    clearTimeout(saveTimeout);
                    saveTimeout = setTimeout(saveApiKeyToStorage, 1000); // Save after 1 second of no typing
                }});
                
                apiKeyInput.addEventListener('blur', saveApiKeyToStorage); // Save when field loses focus
            }}
        }}
        
        // Initialize everything
        setTimeout(() => {{
            loadApiKeyFromStorage();
            setupAutoSave();
        }}, 500);
        
        // Also try to load after a longer delay in case Streamlit is slow
        setTimeout(loadApiKeyFromStorage, 2000);
        </script>
        """, height=0)
        
        # Update session state when API key changes
        if api_key and api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Show status about API key storage
        if api_key:
            st.success("🔑 API key is set and will be saved for future sessions")
        else:
            st.info("💡 Enter your API key above - it will be saved automatically")
        
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
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Video Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a dashboard video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video showing your car's dashboard"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = Path(tmp_file.name)
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Create container for video with ID
            video_container = st.container()
            with video_container:
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
                # Estimate video duration and frames
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(temp_video_path))
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
                analyze_video(temp_video_path, api_key, model, fps, delay, parallel_workers, anomaly_detection, max_acceleration, interpolate_gaps, keep_frames, verbose)
    
    with col2:
        st.header("📊 Results")
        
        # Check for existing results
        if 'analysis_results' in st.session_state:
            display_results(st.session_state.analysis_results)
        else:
            st.info("Upload a video and click 'Analyze Video' to see results here")


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
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Frames", stats.get('total_frames', 0))
    
    with col2:
        success_rate = stats.get('success_rate', 0)
        # Calculate breakdown
        ai_readings = len([r for r in results if r['success'] and not r.get('interpolated', False) and not r.get('anomaly_corrected', False)])
        interpolated_count = len([r for r in results if r.get('interpolated', False)])
        anomaly_corrected_count = len([r for r in results if r.get('anomaly_corrected', False)])
        
        st.metric("Success Rate", f"{success_rate:.1f}%", 
                 help=f"AI: {ai_readings}, Anomaly Corrected: {anomaly_corrected_count}, Interpolated: {interpolated_count}")
    
    with col3:
        if 'min_speed' in stats:
            st.metric("Speed Range", f"{stats['min_speed']}-{stats['max_speed']} km/h")
        else:
            st.metric("Speed Range", "N/A")
    
    with col4:
        st.metric("Duration", f"{stats.get('duration', 0):.1f}s")
    
    with col5:
        if cost_info.get('total_cost_usd', 0) > 0:
            st.metric("Cost", f"${cost_info['total_cost_usd']:.4f}")
        else:
            st.metric("Cost", "<$0.0001")
    
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
        
        # Create columns for slider and current selection display
        slider_col, info_col = st.columns([3, 1])
        
        with slider_col:
            selected_timestamp = st.slider(
                "Select timestamp to view:",
                min_value=0.0,
                max_value=float(max_time),
                value=0.0,
                step=0.05,  # Smaller step for smoother interaction
                format="%.1fs",
                help="Drag the slider to jump to different timestamps in the video analysis"
            )
        
        with info_col:
            # Find the closest data point to the selected timestamp
            if len(successful_df) > 0:
                closest_idx = (successful_df['timestamp'] - selected_timestamp).abs().idxmin()
                closest_speed = successful_df.loc[closest_idx, 'speed']
                st.metric("Speed at Time", f"{closest_speed:.0f} km/h", f"{selected_timestamp:.1f}s")
        
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
        
        # Render with performance optimizations
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            key="speed_chart",
            config={
                'displayModeBar': False,  # Hide toolbar for faster rendering
                'staticPlot': False,
                'responsive': True
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
        cost_col1, cost_col2, cost_col3 = st.columns(3)
        
        with cost_col1:
            st.metric("API Calls", cost_info.get('api_calls', 0))
            st.metric("Model", cost_info.get('model', 'N/A'))
        
        with cost_col2:
            st.metric("Input Tokens", f"{cost_info.get('input_tokens', 0):,}")
            st.metric("Output Tokens", f"{cost_info.get('output_tokens', 0):,}")
        
        with cost_col3:
            st.metric("Input Cost", f"${cost_info.get('input_cost_usd', 0):.6f}")
            st.metric("Output Cost", f"${cost_info.get('output_cost_usd', 0):.6f}")
        
        if cost_info.get('total_cost_usd', 0) < 0.01:
            st.success("💡 Very affordable! Less than 1 cent.")
        elif cost_info.get('total_cost_usd', 0) < 0.10:
            st.info("💡 Very reasonable cost, less than 10 cents.")
        else:
            st.warning("⚠️ Consider optimizing frames/fps for cost reduction.")
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"{Path(video_name).stem}_speed_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
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


if __name__ == "__main__":
    main()