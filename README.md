# Speedometer AI

AI-powered speedometer reading from dashboard video using Google Gemini.

## Features

- Extract speed data from dashboard videos using AI vision
- CLI and web UI interfaces
- Export results to CSV and charts
- High accuracy with Gemini AI (100% success rate in testing)

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Basic usage
speedometer-ai path/to/dashboard_video.mp4 -k YOUR_GEMINI_API_KEY

# With custom settings
speedometer-ai video.mp4 -k API_KEY --fps 5 --delay 0.5 --verbose

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
speedometer-ai video.mp4
```

## Requirements

- Python 3.8+
- FFmpeg
- Google Gemini API key

## Tested On

- Insta360 GO 3S camera footage
- Nissan 350Z dashboard
- Red LED digital speedometer displays