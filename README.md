# Speedometer AI

AI-powered speedometer reading from dashboard video using Google Gemini.

## Features

- Extract speed data from dashboard videos using AI vision
- **Parallel processing** for fast analysis (up to 10x faster)
- **AI-powered anomaly detection** that intelligently corrects speed misreadings using physics and context
- **AI-powered gap filling** that uses surrounding data and car physics to estimate missing values
- **Configurable car acceleration limits** for accurate AI-powered analysis
- **Multiple Gemini models** including the latest Gemini 2.0 Flash experimental
- CLI and web UI interfaces
- Export results to CSV and charts
- High accuracy with Gemini AI (100% success rate in testing)
- Real-time cost tracking and breakdown
- Timestamp-based frame naming for precise analysis

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Basic usage
speedometer-ai analyze path/to/dashboard_video.mp4 -k YOUR_GEMINI_API_KEY

# With parallel processing and custom settings  
speedometer-ai analyze video.mp4 -k API_KEY --fps 5 --parallel 15 --verbose

# With custom car acceleration (default: 16.95 km/h/s)
speedometer-ai analyze video.mp4 -k API_KEY --max-acceleration 20.0

# With latest experimental model (Gemini 2.0)
speedometer-ai analyze video.mp4 -k API_KEY --model gemini-2.0-flash-exp

# Sequential processing (slower but more conservative)
speedometer-ai analyze video.mp4 -k API_KEY --parallel 1 --delay 1.0

# View results
speedometer-ai show results.csv
```

### Web UI

```bash
speedometer-ai ui
```

Then open http://localhost:8501 in your browser.

### Environment Variable

Set your API key as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
speedometer-ai analyze video.mp4
```

## Requirements

- Python 3.8+
- FFmpeg
- Google Gemini API key

## Tested On

- Insta360 GO 3S camera footage
- Nissan 350Z dashboard
- Red LED digital speedometer displays