# Speedometer AI

AI-powered speedometer reading from dashboard video using Google Gemini with rule-based post-processing.

## Features

- **AI vision analysis** extracts speed data from dashboard videos using Google Gemini
- **Parallel processing** for fast analysis (up to 10x faster)
- **Rule-based anomaly detection** corrects OCR misreadings using physics constraints and digit confusion patterns
- **Intelligent gap filling** estimates missing values using linear interpolation with physics validation
- **Data smoothing** to ensure a smooth speed line by correcting outliers and applying moving averages
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

## API Key Setup

**First, configure your Gemini API key using one of these methods:**

### Option 1: Secrets File (Recommended)
```bash
cp secrets.json.example secrets.json
# Edit secrets.json with your API key
```

### Option 2: Environment Variable
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Option 3: Command Line
Use the `-k` flag to specify the API key directly.

## Usage

### CLI

```bash
# Basic usage (uses secrets.json or environment variable)
speedometer-ai analyze path/to/dashboard_video.mp4

# With parallel processing and custom settings  
speedometer-ai analyze video.mp4 --fps 5 --parallel 15 --verbose

# With custom car acceleration limits for physics validation
speedometer-ai analyze video.mp4 --max-acceleration 20.0

# Disable rule-based post-processing (AI vision only)
speedometer-ai analyze video.mp4 --no-anomaly-detection --no-interpolate

# With latest experimental model (Gemini 2.0)
speedometer-ai analyze video.mp4 --model gemini-2.0-flash-exp

# Sequential processing (slower but more conservative)
speedometer-ai analyze video.mp4 --parallel 1 --delay 1.0

# Using API key directly (if not using secrets.json or environment)
speedometer-ai analyze video.mp4 -k YOUR_GEMINI_API_KEY

# View results
speedometer-ai show results.csv
```

### Web UI

```bash
speedometer-ai ui
```

Then open http://localhost:8501 in your browser. The UI will automatically load your API key from secrets.json or environment variables.

## API Key Details

The system loads API keys in this priority order:
1. **secrets.json file** (recommended for local development)
2. **GEMINI_API_KEY environment variable** (good for servers/containers)  
3. **Manual input** (CLI `-k` flag or web UI input)

**secrets.json format:**
```json
{
  "gemini_api_key": "your_gemini_api_key_here"
}
```

**Note**: The `secrets.json` file is automatically ignored by git for security.

## Processing Pipeline

The system uses a hybrid approach combining AI vision with rule-based post-processing:

1. **Frame Extraction**: Video is split into frames using FFmpeg
2. **AI Vision Analysis**: Google Gemini reads speedometer values from each frame
3. **Rule-Based Post-Processing** (applied automatically):
   - **Anomaly Detection**: Corrects OCR misreads using digit confusion patterns (0↔6, 8↔9, etc.)
   - **Physics Validation**: Removes readings that violate acceleration limits
   - **Gap Interpolation**: Fills missing values using linear interpolation
   - **Data Smoothing**: Creates smooth speed lines using outlier detection + moving averages

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