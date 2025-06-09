from setuptools import setup, find_packages

setup(
    name="speedometer-ai",
    version="1.0.0",
    description="AI-powered speedometer reading from dashboard video",
    author="Claude Code",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.8.0",
        "Pillow>=9.0.0",
        "click>=8.0.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "streamlit>=1.28.0",
        "plotly>=5.0.0",
        "pytest>=7.0.0",
        "pytest-streamlit>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "speedometer-ai=speedometer_ai.cli:cli",
        ],
    },
    python_requires=">=3.8",
)