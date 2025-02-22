import os
import tomli
from pathlib import Path
from typing import Optional, Dict

DEFAULT_CONFIG = {
    "obsidian": {
        "vault_path": "~/Documents/Obsidian",
        "template": """# {title}

## Metadata
- **Podcast**: {podcast_title}
- **Published**: {published_at}
- **URL**: {audio_url}

## Summary
{summary}

## Transcript
{transcript}
"""
    },
    "openrouter": {
        "api_key": "",  # Set via PODSIDIAN_OPENROUTER_API_KEY env var
        "model": "anthropic/claude-2",
        "prompt": """You are a helpful podcast summarizer. 
Given the following podcast transcript, provide:
1. A concise 2-3 paragraph summary of the key points
2. A bullet list of the most important takeaways
3. Any notable quotes, properly attributed

Transcript:
{transcript}
"""
    }
}

class Config:
    def __init__(self):
        self.config_path = os.path.expanduser("~/.config/podsidian/config.toml")
        self.config = DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "rb") as f:
                user_config = tomli.load(f)
                # Deep merge user config with defaults
                self._merge_configs(self.config, user_config)
        
        # Override with environment variables if set
        if api_key := os.getenv("PODSIDIAN_OPENROUTER_API_KEY"):
            self.config["openrouter"]["api_key"] = api_key
            
        if vault_path := os.getenv("PODSIDIAN_VAULT_PATH"):
            self.config["obsidian"]["vault_path"] = vault_path
    
    def _merge_configs(self, base: Dict, override: Dict):
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    @property
    def vault_path(self) -> Path:
        """Get the configured Obsidian vault path."""
        return Path(os.path.expanduser(self.config["obsidian"]["vault_path"]))
    
    @property
    def note_template(self) -> str:
        """Get the configured note template."""
        return self.config["obsidian"]["template"]
    
    @property
    def openrouter_api_key(self) -> Optional[str]:
        """Get the OpenRouter API key."""
        return self.config["openrouter"]["api_key"]
    
    @property
    def openrouter_model(self) -> str:
        """Get the configured OpenRouter model."""
        return self.config["openrouter"]["model"]
    
    @property
    def openrouter_prompt(self) -> str:
        """Get the configured prompt template."""
        return self.config["openrouter"]["prompt"]

# Global config instance
config = Config()
