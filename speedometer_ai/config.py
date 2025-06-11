"""
Configuration management for Speedometer AI
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for Speedometer AI"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config_data = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find the config file in various locations"""
        possible_locations = [
            # Current working directory
            "config.json",
            # Project root
            Path(__file__).parent.parent / "config.json",
            # User home directory
            Path.home() / ".speedometer_ai_config.json",
            # System config directory
            "/etc/speedometer_ai/config.json"
        ]
        
        for location in possible_locations:
            if Path(location).exists():
                return str(location)
        
        # Return default location if none found
        return str(Path(__file__).parent.parent / "config.json")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        # Return default configuration
        return {
            "gemini_api_key": "",
            "default_settings": {
                "fps": 3.0,
                "delay": 1.0,
                "model": "gemini-1.5-flash",
                "parallel_workers": 3,
                "anomaly_detection": True,
                "max_acceleration": 16.95,
                "interpolate_gaps": True
            }
        }
    
    def get_api_key(self) -> str:
        """Get Gemini API key from config or environment"""
        # Priority: config file > environment variable > empty string
        api_key = self.config_data.get("gemini_api_key", "")
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "")
        return api_key
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default application settings"""
        return self.config_data.get("default_settings", {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config_data.get(key, default)
    
    def save_config(self, config_data: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            # Ensure directory exists
            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.config_data = config_data
            return True
        except (PermissionError, OSError) as e:
            print(f"Error: Could not save config file {self.config_file}: {e}")
            return False
    
    def update_api_key(self, api_key: str) -> bool:
        """Update API key in configuration"""
        config_copy = self.config_data.copy()
        config_copy["gemini_api_key"] = api_key
        return self.save_config(config_copy)


# Global config instance
config = Config()