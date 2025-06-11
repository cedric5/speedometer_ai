#!/usr/bin/env python3
"""
Startup script for Speedometer AI that removes file upload size limits
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Set environment variable to remove file size limit
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '10000'  # 10GB in MB
    
    # Also try the alternative environment variable name
    os.environ['STREAMLIT_SERVER_MAXUPLOADSIZE'] = '10000'
    
    # Get the path to the UI module
    current_dir = Path(__file__).parent
    ui_path = current_dir / "speedometer_ai" / "ui.py"
    
    # Run streamlit with the configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(ui_path),
        "--server.maxUploadSize=10000",
        "--server.fileWatcherType=auto"
    ]
    
    print("🚗 Starting Speedometer AI with no file size limits...")
    print(f"📁 UI path: {ui_path}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Speedometer AI stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()