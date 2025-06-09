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
            
            # Analysis button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                analyze_video(temp_video_path, api_key, fps, delay, keep_frames, verbose)
    
    with col2:
        st.header("📊 Results")
        
        # Check for existing results
        if 'analysis_results' in st.session_state:
            display_results(st.session_state.analysis_results)
        else:
            st.info("Upload a video and click 'Analyze Video' to see results here")


def analyze_video(video_path: Path, api_key: str, fps: float, delay: float, keep_frames: bool, verbose: bool):
    """Analyze the uploaded video"""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
            status_text.text("🔍 Analyzing frames with AI...")
            
            def progress_callback(current, total, filename):
                progress = 0.2 + (current / total) * 0.6
                progress_bar.progress(progress)
                status_text.text(f"🔍 Analyzing frame {current}/{total}: {filename}")
            
            results = analyzer.analyze_video_frames(
                frames_dir, fps, delay, progress_callback
            )
            
            # Step 4: Process results
            status_text.text("📊 Processing results...")
            progress_bar.progress(0.9)
            
            # Store results in session state
            st.session_state.analysis_results = {
                'results': results,
                'stats': analyzer.get_statistics(),
                'video_name': video_path.name
            }
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            st.success("🎉 Analysis completed successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def display_results(analysis_data):
    """Display analysis results"""
    
    results = analysis_data['results']
    stats = analysis_data['stats']
    video_name = analysis_data['video_name']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", stats.get('total_frames', 0))
    
    with col2:
        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
    
    with col3:
        if 'min_speed' in stats:
            st.metric("Speed Range", f"{stats['min_speed']}-{stats['max_speed']} km/h")
        else:
            st.metric("Speed Range", "N/A")
    
    with col4:
        st.metric("Duration", f"{stats.get('duration', 0):.1f}s")
    
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
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df['Status'] = display_df['success'].apply(lambda x: '✅' if x else '❌')
    display_df['Speed (km/h)'] = display_df['speed'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    display_df['Time (s)'] = display_df['timestamp'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_df[['Status', 'Time (s)', 'Speed (km/h)', 'filename']],
        use_container_width=True,
        hide_index=True
    )
    
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
        report = generate_summary_report(results, stats, video_name)
        st.download_button(
            label="📊 Download Report",
            data=report,
            file_name=f"{Path(video_name).stem}_analysis_report.txt",
            mime="text/plain"
        )


def generate_summary_report(results, stats, video_name):
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