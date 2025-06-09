# Speedometer AI

AI-powered speedometer reading from dashboard video using Google Gemini with rule-based post-processing.

## Features

- **AI vision analysis** extracts speed data from dashboard videos using Google Gemini
- **Parallel processing** for fast analysis (up to 10x faster)
- **Rule-based anomaly detection** corrects OCR misreadings using physics constraints and digit confusion patterns
- **Intelligent gap filling** estimates missing values using linear interpolation with physics validation
- **Aggressive smoothing** ensures ultra-smooth speed lines by correcting outliers and applying moving averages
- **Complete data guarantee** - no empty cells in final CSV output
- **Configurable acceleration limits** for realistic physics validation (default: 16.95 km/h/s)
- **Multiple Gemini models** including Gemini 1.5 Flash, Pro, and 2.0 Flash experimental
- **Deterministic results** - same input always produces same output with rule-based processing
- CLI and web UI interfaces
- Export results to CSV with detailed processing flags and interactive charts
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

# With custom car acceleration limits for physics validation
speedometer-ai analyze video.mp4 -k API_KEY --max-acceleration 20.0

# Disable rule-based post-processing (AI vision only)
speedometer-ai analyze video.mp4 -k API_KEY --no-anomaly-detection --no-interpolate

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

## Processing Pipeline

The system uses a hybrid approach combining AI vision with rule-based post-processing:

1. **Frame Extraction**: Video is split into frames using FFmpeg
2. **AI Vision Analysis**: Google Gemini reads speedometer values from each frame
3. **Rule-Based Post-Processing** (applied automatically):
   - **Anomaly Detection**: Corrects OCR misreads using digit confusion patterns (0↔6, 8↔9, etc.)
   - **Physics Validation**: Removes readings that violate acceleration limits
   - **Gap Interpolation**: Fills missing values using linear interpolation
   - **Aggressive Smoothing**: Creates ultra-smooth speed lines using outlier detection + moving averages
   - **Complete Data**: Ensures every CSV row has a speed value

**Result**: Clean, smooth, complete speed data with detailed processing flags in CSV output.

## Requirements

- Python 3.8+
- FFmpeg
- Google Gemini API key

## Tested On

- Insta360 GO 3S camera footage
- Nissan 350Z dashboard
- Red LED digital speedometer displays

## CSV Output

The generated CSV includes these columns:
- `frame`: Frame number
- `timestamp`: Time in video (seconds)
- `speed`: Speed reading (km/h) - never empty
- `filename`: Source frame filename
- `success`: Whether AI reading was successful
- `interpolated`: Whether value was interpolated
- `smoothed`: Whether value was smoothed
- `anomaly_corrected`: Whether anomaly was corrected
- `response`: Detailed processing history