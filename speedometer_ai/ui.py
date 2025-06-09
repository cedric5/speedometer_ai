"""
Streamlit web UI for Speedometer AI
"""

import streamlit as st
import os
import tempfile
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from .core import SpeedometerAnalyzer
from .utils import (
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
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            value=os.getenv('GEMINI_API_KEY', ''),
            help="Enter your Google Gemini API key"
        )
        
        # Analysis settings
        st.subheader("Analysis Settings")
        fps = st.slider("Frames per second", 1.0, 10.0, 3.0, 0.5)
        delay = st.slider("API delay (seconds)", 0.5, 3.0, 1.0, 0.1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            parallel_workers = st.slider("Parallel workers", 1, 10, 5, 1, 
                                        help="Number of parallel API calls (higher = faster but more load)")
            
            st.subheader("Data Processing")
            anomaly_detection = st.checkbox("Detect and correct anomalies", True,
                                           help="Detect and correct AI misreadings (e.g., 90 read as 60)")
            max_acceleration = st.slider("Max car acceleration (km/h/s)", 5.0, 30.0, 16.95, 0.05,
                                       help="Maximum realistic acceleration for your car in km/h per second")
            interpolate_gaps = st.checkbox("Fill gaps using interpolation", True,
                                         help="Automatically fill missing speed readings using surrounding data")
            
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
            st.video(uploaded_file)
            
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
                    # Rough estimate: ~1000 input tokens per frame, ~10 output tokens
                    estimated_input_tokens = estimated_frames * 1000
                    estimated_output_tokens = estimated_frames * 10
                    estimated_cost = (estimated_input_tokens / 1_000_000) * 0.075 + (estimated_output_tokens / 1_000_000) * 0.30
                    
                    st.info(f"📊 **Estimated Analysis**: {estimated_frames} frames (~{duration:.1f}s video) | **Est. Cost**: ${estimated_cost:.4f} USD")
                except:
                    st.warning("⚠️ Could not estimate video duration")
            
            # Analysis button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                analyze_video(temp_video_path, api_key, fps, delay, parallel_workers, anomaly_detection, max_acceleration, interpolate_gaps, keep_frames, verbose)
    
    with col2:
        st.header("📊 Results")
        
        # Check for existing results
        if 'analysis_results' in st.session_state:
            display_results(st.session_state.analysis_results)
        else:
            st.info("Upload a video and click 'Analyze Video' to see results here")


def analyze_video(video_path: Path, api_key: str, fps: float, delay: float, parallel_workers: int, anomaly_detection: bool, max_acceleration: float, interpolate_gaps: bool, keep_frames: bool, verbose: bool):
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
            status_text.text("🤖 Initializing Gemini AI analyzer...")
            progress_bar.progress(0.2)
            
            analyzer = SpeedometerAnalyzer(api_key)
            
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
            
            results = analyzer.analyze_video_frames(
                frames_dir, fps, delay, progress_callback, max_workers=parallel_workers
            )
            
            # Step 4: Post-process results
            status_text.text("🔧 Post-processing results...")
            progress_bar.progress(0.8)
            
            raw_success_rate = len([r for r in results if r['success']]) / len(results) * 100
            
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
                
                if processed_success_rate > raw_success_rate:
                    improvement = processed_success_rate - raw_success_rate
                    st.info(f"✨ Data processing improved success rate by {improvement:.1f}% ({raw_success_rate:.1f}% → {processed_success_rate:.1f}%)")
                
                if corrected_count > 0:
                    st.success(f"⚠️ Corrected {corrected_count} anomalous AI readings")
                if interpolated_count > 0:
                    st.success(f"🔮 Interpolated {interpolated_count} missing values")
            
            # Step 5: Finalize results
            status_text.text("📊 Finalizing results...")
            progress_bar.progress(0.9)
            
            # Store results in session state
            st.session_state.analysis_results = {
                'results': results,
                'stats': analyzer.get_statistics(),
                'cost_info': analyzer.get_cost_info(),
                'video_name': video_path.name
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
    
    # Speed chart
    if len(df[df['success'] == True]) > 0:
        st.subheader("📈 Speed Progression")
        
        successful_df = df[df['success'] == True].copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=successful_df['timestamp'],
            y=successful_df['speed'],
            mode='lines+markers',
            name='Speed',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Speed vs Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Speed (km/h)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Add legend
    st.caption("**Status Legend:** ✅ AI Read = Direct AI reading | ⚠️ Corrected = AI misread digit, corrected based on context | 🔮 Estimated = AI couldn't read, value interpolated from surrounding data | ❌ Failed = No reading available")
    
    # Format the dataframe for display
    display_df = df.copy()
    
    def format_status(row):
        if row['success']:
            if row.get('interpolated', False):
                return '🔮 Estimated'
            elif row.get('anomaly_corrected', False):
                return '⚠️ Corrected'
            else:
                return '✅ AI Read'
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