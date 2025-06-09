"""
Command-line interface for Speedometer AI
"""

import click
import os
import sys
from pathlib import Path
from .core import SpeedometerAnalyzer
from .utils import (
    extract_frames_from_video, 
    save_results_to_csv, 
    create_speed_chart,
    print_analysis_summary,
    validate_video_file,
    check_ffmpeg_available
)


@click.command()
@click.argument('video_path', type=click.Path(exists=True, path_type=Path))
@click.option('--api-key', '-k', 
              help='Gemini API key (or set GEMINI_API_KEY env var)')
@click.option('--output', '-o', 
              type=click.Path(path_type=Path),
              help='Output directory (default: video_name_analysis)')
@click.option('--fps', '-f', 
              default=3.0, 
              help='Frames per second to extract (default: 3.0)')
@click.option('--delay', '-d', 
              default=1.0, 
              help='Delay between API calls in seconds (default: 1.0)')
@click.option('--chart/--no-chart', 
              default=True, 
              help='Generate speed chart (default: True)')
@click.option('--keep-frames/--no-keep-frames', 
              default=False, 
              help='Keep extracted frames (default: False)')
@click.option('--verbose', '-v', 
              is_flag=True, 
              help='Verbose output')
def analyze(video_path, api_key, output, fps, delay, chart, keep_frames, verbose):
    """
    Analyze speedometer readings from dashboard video using AI
    
    VIDEO_PATH: Path to the dashboard video file
    """
    
    # Header
    click.echo(click.style("🚗 Speedometer AI v1.0.0", fg='blue', bold=True))
    click.echo("AI-powered speedometer reading from dashboard video\n")
    
    # Validate inputs
    if not validate_video_file(video_path):
        click.echo(click.style("❌ Error: Invalid or unreadable video file", fg='red'))
        sys.exit(1)
    
    if not check_ffmpeg_available():
        click.echo(click.style("❌ Error: FFmpeg not found. Please install FFmpeg.", fg='red'))
        sys.exit(1)
    
    # Get API key
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            api_key = click.prompt('Enter your Gemini API key', hide_input=True)
    
    if not api_key:
        click.echo(click.style("❌ Error: No API key provided", fg='red'))
        sys.exit(1)
    
    # Setup output directory
    if not output:
        output = video_path.parent / f"{video_path.stem}_analysis"
    
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    
    try:
        # Step 1: Extract frames
        click.echo(f"📹 Extracting frames from video at {fps} fps...")
        
        if verbose:
            click.echo(f"   Input: {video_path}")
            click.echo(f"   Output: {frames_dir}")
        
        frame_files = extract_frames_from_video(video_path, frames_dir, fps)
        click.echo(f"   ✓ Extracted {len(frame_files)} frames")
        
        # Step 2: Initialize analyzer
        click.echo(f"🤖 Initializing Gemini AI analyzer...")
        analyzer = SpeedometerAnalyzer(api_key)
        
        # Step 3: Analyze frames
        click.echo(f"🔍 Analyzing frames (delay: {delay}s between calls)...")
        
        def progress_callback(current, total, filename):
            if verbose:
                click.echo(f"   Analyzing {current}/{total}: {filename}")
            else:
                # Simple progress indicator
                click.echo(f"   Progress: {current}/{total}", nl=False)
                if current < total:
                    click.echo("\r", nl=False)
                else:
                    click.echo()
        
        results = analyzer.analyze_video_frames(
            frames_dir, fps, delay, progress_callback
        )
        
        # Step 4: Save results
        csv_path = output / "speed_results.csv"
        save_results_to_csv(results, csv_path)
        click.echo(f"   ✓ Results saved to {csv_path}")
        
        # Step 5: Generate chart
        if chart:
            chart_path = output / "speed_chart.png"
            create_speed_chart(results, chart_path)
            click.echo(f"   ✓ Chart saved to {chart_path}")
        
        # Step 6: Show summary
        stats = analyzer.get_statistics()
        
        if verbose:
            print_analysis_summary(results, stats)
        else:
            click.echo(f"\n📊 Analysis complete!")
            click.echo(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
            if 'min_speed' in stats:
                click.echo(f"   Speed range: {stats['min_speed']}-{stats['max_speed']} km/h")
        
        # Step 7: Cleanup frames if requested
        if not keep_frames:
            import shutil
            shutil.rmtree(frames_dir)
            if verbose:
                click.echo("   ✓ Cleaned up temporary frames")
        
        click.echo(f"\n🎉 Analysis complete! Results in: {output}")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'))
        sys.exit(1)


@click.group()
def cli():
    """Speedometer AI - AI-powered speedometer reading"""
    pass


@cli.command()
@click.argument('csv_path', type=click.Path(exists=True, path_type=Path))
def show(csv_path):
    """Show analysis results from CSV file"""
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        click.echo(click.style("📊 Speed Analysis Results", fg='blue', bold=True))
        click.echo(f"File: {csv_path}\n")
        
        # Basic stats
        successful = df[df['success'] == True]
        
        click.echo(f"Total frames: {len(df)}")
        click.echo(f"Successful readings: {len(successful)}")
        click.echo(f"Success rate: {len(successful)/len(df)*100:.1f}%")
        
        if len(successful) > 0:
            speeds = successful['speed'].dropna()
            click.echo(f"Speed range: {speeds.min():.0f}-{speeds.max():.0f} km/h")
            click.echo(f"Average speed: {speeds.mean():.1f} km/h")
        
        click.echo(f"\nDetailed timeline:")
        for _, row in df.iterrows():
            status = "✓" if row['success'] else "✗"
            speed = f"{row['speed']:.0f}" if pd.notna(row['speed']) else "N/A"
            click.echo(f"  {status} {row['timestamp']:5.1f}s: {speed:>3} km/h")
            
    except Exception as e:
        click.echo(click.style(f"Error reading results: {e}", fg='red'))


cli.add_command(analyze)


@cli.command()
def ui():
    """Launch the web UI"""
    try:
        import subprocess
        import sys
        
        # Get the path to the UI script
        ui_script = Path(__file__).parent / "ui.py"
        
        click.echo("🌐 Launching Speedometer AI Web UI...")
        click.echo("   Open http://localhost:8501 in your browser")
        
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_script)])
        
    except ImportError:
        click.echo(click.style("❌ Streamlit not installed. Install with: pip install streamlit", fg='red'))
    except Exception as e:
        click.echo(click.style(f"❌ Error launching UI: {e}", fg='red'))


if __name__ == '__main__':
    cli()