"""
Command-line interface for Speedometer AI
"""

import click
import os
import sys
from pathlib import Path
from .core import SpeedometerAnalyzer, QuotaExceededError
from .utils import (
    extract_frames_from_video, 
    save_results_to_csv, 
    create_speed_chart,
    print_analysis_summary,
    print_cost_summary,
    validate_video_file,
    check_ffmpeg_available,
    interpolate_missing_speeds,
    detect_and_correct_anomalies
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
              help='Delay between API calls in seconds (default: 1.0, ignored if parallel)')
@click.option('--parallel', '-p', 
              default=10, 
              help='Number of parallel workers (default: 10, use 1 for sequential)')
@click.option('--interpolate/--no-interpolate', 
              default=True, 
              help='AI-powered gap filling in speed data (default: True)')
@click.option('--anomaly-detection/--no-anomaly-detection', 
              default=True, 
              help='AI-powered anomaly detection and correction (default: True)')
@click.option('--max-acceleration', 
              default=16.95, 
              help='Maximum car acceleration in km/h/s for AI analysis (default: 16.95 km/h/s)')
@click.option('--model', '-m',
              default='gemini-1.5-flash',
              type=click.Choice(['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp'], case_sensitive=False),
              help='Gemini model to use (default: gemini-1.5-flash)')
@click.option('--chart/--no-chart', 
              default=True, 
              help='Generate speed chart (default: True)')
@click.option('--keep-frames/--no-keep-frames', 
              default=False, 
              help='Keep extracted frames (default: False)')
@click.option('--verbose', '-v', 
              is_flag=True, 
              help='Verbose output')
def analyze(video_path, api_key, output, fps, delay, parallel, interpolate, anomaly_detection, max_acceleration, model, chart, keep_frames, verbose):
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
        
        # Cost estimation
        estimated_frames = len(frame_files)
        estimated_input_tokens = estimated_frames * 1000  # Rough estimate
        estimated_output_tokens = estimated_frames * 10
        estimated_cost = (estimated_input_tokens / 1_000_000) * 0.075 + (estimated_output_tokens / 1_000_000) * 0.30
        click.echo(f"   💰 Estimated cost: ${estimated_cost:.4f} USD for {estimated_frames} frames")
        
        # Step 2: Initialize analyzer
        click.echo(f"🤖 Initializing Gemini AI analyzer ({model})...")
        analyzer = SpeedometerAnalyzer(api_key, model_name=model)
        
        # Step 3: Analyze frames
        if parallel > 1:
            click.echo(f"🔍 Analyzing frames with {parallel} parallel workers...")
        else:
            click.echo(f"🔍 Analyzing frames sequentially (delay: {delay}s between calls)...")
        
        def progress_callback(current, total, filename):
            # Get current cost info
            cost_info = analyzer.get_cost_info()
            cost_str = f"${cost_info['total_cost_usd']:.4f}" if cost_info['total_cost_usd'] > 0 else "<$0.0001"
            
            if verbose:
                if parallel > 1:
                    click.echo(f"   Completed {current}/{total}: {filename} | Cost so far: {cost_str}")
                else:
                    click.echo(f"   Analyzing {current}/{total}: {filename} | Cost so far: {cost_str}")
            else:
                # Simple progress indicator with cost
                click.echo(f"   Progress: {current}/{total} | Cost: {cost_str}", nl=False)
                if current < total:
                    click.echo("\r", nl=False)
                else:
                    click.echo()
        
        try:
            results = analyzer.analyze_video_frames(
                frames_dir, fps, delay, progress_callback, max_workers=parallel
            )
        except QuotaExceededError as e:
            click.echo(f"\n{str(e)}")
            click.echo(f"\n💡 Tip: For Gemini 2.0 models, try: speedometer-ai analyze --model gemini-1.5-flash --parallel 1")
            sys.exit(1)
        except Exception as e:
            error_str = str(e)
            click.echo(f"\n❌ Analysis failed: {error_str}")
            
            # Provide specific guidance based on error type
            if "429" in error_str or "quota" in error_str.lower():
                click.echo(f"\n💡 This appears to be a quota/rate limit error. Try:")
                click.echo(f"   • Reduce parallel workers: --parallel 1")
                click.echo(f"   • Switch to gemini-1.5-flash: --model gemini-1.5-flash")
                click.echo(f"   • Wait a few minutes before retrying")
            elif "api" in error_str.lower() or "key" in error_str.lower():
                click.echo(f"\n💡 This appears to be an API key issue. Check:")
                click.echo(f"   • Your API key is valid")
                click.echo(f"   • You have access to the selected model")
                click.echo(f"   • Your Gemini API account is active")
            elif "permission" in error_str.lower() or "access" in error_str.lower():
                click.echo(f"\n💡 This appears to be a permissions issue. Check:")
                click.echo(f"   • Your API key has access to the {model} model")
                click.echo(f"   • The model name is correct")
            else:
                click.echo(f"\n💡 General troubleshooting:")
                click.echo(f"   • Check your internet connection")
                click.echo(f"   • Verify your API key is correct")
                click.echo(f"   • Try with --model gemini-1.5-flash --parallel 1")
            
            sys.exit(1)
        
        # Step 4: Post-process results with AI
        raw_success_rate = len([r for r in results if r['success']]) / len(results) * 100
        
        if anomaly_detection or interpolate:
            click.echo(f"🤖 AI analyzing speed data for anomalies and gaps...")
            try:
                results = analyzer.analyze_speed_data_with_ai(results, max_acceleration)
            except Exception as e:
                click.echo(f"   ⚠️ AI analysis failed, falling back to rule-based methods: {e}")
                # Fallback to original methods if AI fails
                if anomaly_detection:
                    click.echo(f"🔧 Detecting and correcting anomalies (rule-based)...")
                    results = detect_and_correct_anomalies(results, max_change_per_second=max_acceleration)
                    
                if interpolate:
                    click.echo(f"🔧 Filling gaps using interpolation (rule-based)...")
                    results = interpolate_missing_speeds(results, max_gap_size=3)
        
        # Show improvement if processing was applied
        if anomaly_detection or interpolate:
            processed_success_rate = len([r for r in results if r['success']]) / len(results) * 100
            corrected_count = len([r for r in results if r.get('anomaly_corrected', False)])
            interpolated_count = len([r for r in results if r.get('interpolated', False)])
            
            if processed_success_rate > raw_success_rate:
                improvement = processed_success_rate - raw_success_rate
                click.echo(f"   ✓ Improved success rate by {improvement:.1f}% ({raw_success_rate:.1f}% → {processed_success_rate:.1f}%)")
            
            if corrected_count > 0:
                click.echo(f"   ✓ Corrected {corrected_count} anomalous readings")
            if interpolated_count > 0:
                click.echo(f"   ✓ Interpolated {interpolated_count} missing values")
        
        # Step 5: Save results
        csv_path = output / "speed_results.csv"
        save_results_to_csv(results, csv_path)
        click.echo(f"   ✓ Results saved to {csv_path}")
        
        # Step 6: Generate chart
        if chart:
            chart_path = output / "speed_chart.png"
            create_speed_chart(results, chart_path)
            click.echo(f"   ✓ Chart saved to {chart_path}")
        
        # Step 7: Show summary
        stats = analyzer.get_statistics()
        cost_info = analyzer.get_cost_info()
        
        # Save results with cost info
        save_results_to_csv(results, csv_path, cost_info)
        
        if verbose:
            print_analysis_summary(results, stats)
            print_cost_summary(cost_info)
        else:
            click.echo(f"\n📊 Analysis complete!")
            click.echo(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
            if 'min_speed' in stats:
                click.echo(f"   Speed range: {stats['min_speed']}-{stats['max_speed']} km/h")
            
            # Always show cost
            click.echo(f"\n💰 Cost:")
            click.echo(f"   API calls: {cost_info['api_calls']}")
            if cost_info['total_cost_usd'] > 0:
                click.echo(f"   Total cost: ${cost_info['total_cost_usd']:.4f} USD")
            else:
                click.echo(f"   Total cost: <$0.0001 USD (very low)")
            click.echo(f"   Model: {cost_info['model']}")
        
        # Step 8: Cleanup frames if requested
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
        ai_readings = df[(df['success'] == True) & (df.get('interpolated', False) == False) & 
                        (df.get('anomaly_corrected', False) == False)]
        interpolated = df[df.get('interpolated', False) == True]
        anomaly_corrected = df[df.get('anomaly_corrected', False) == True]
        
        click.echo(f"Total frames: {len(df)}")
        click.echo(f"Successful readings: {len(successful)}")
        click.echo(f"Success rate: {len(successful)/len(df)*100:.1f}%")
        click.echo(f"  • Direct AI readings: {len(ai_readings)}")
        click.echo(f"  • Anomaly corrected: {len(anomaly_corrected)}")
        click.echo(f"  • Interpolated values: {len(interpolated)}")
        
        if len(successful) > 0:
            speeds = successful['speed'].dropna()
            click.echo(f"Speed range: {speeds.min():.0f}-{speeds.max():.0f} km/h")
            click.echo(f"Average speed: {speeds.mean():.1f} km/h")
        
        click.echo(f"\nDetailed timeline:")
        click.echo(f"  Legend: ✓=AI Read, ⚠=Anomaly Corrected, ~=Interpolated, ✗=Failed")
        for _, row in df.iterrows():
            if row['success']:
                if row.get('interpolated', False):
                    status = "~"
                elif row.get('anomaly_corrected', False):
                    status = "⚠"
                else:
                    status = "✓"
            else:
                status = "✗"
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