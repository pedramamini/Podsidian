import os
import tomli
from pathlib import Path
from typing import Optional, Dict, List

# Available Whisper models
WHISPER_MODELS = {
    'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
    'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3',
    'large', 'large-v3-turbo'
}

DEFAULT_CONFIG = {
    "whisper": {
        "model": "large-v3",
        "language": "",
        "cpu_only": False,
        "threads": 4
    },
    "obsidian": {
        "vault_path": "~/Documents/Obsidian",
        "template": """# {title}

## Metadata
- **Podcast**: {podcast_title}
- **Published**: {published_at}
- **URL**: {audio_url}

## Summary
{summary}

{value_analysis}
## Transcript
{transcript}
"""
    },
    "openrouter": {
        "api_key": "",  # Set via PODSIDIAN_OPENROUTER_API_KEY env var
        "model": "openai/gpt-4",
        "processing_model": "openai/gpt-4",  # Model for topic detection and transcript correction
        "topic_sample_size": 4000,  # Sample size for topic detection
        "prompt": """You are a helpful podcast summarizer. 
Given the following podcast transcript, provide:
1. A concise 2-3 paragraph summary of the key points
2. A bullet list of the most important takeaways
3. Any notable quotes, properly attributed

Transcript:
{transcript}
""",
        "value_prompt_enabled": False,  # Whether to include value analysis
        "value_prompt": """IDENTITY and PURPOSE

You are an expert parser and rater of value in content. Your goal is to determine how much value a reader/listener is
being provided in a given piece of content as measured by a new metric called Value Per Minute (VPM).

Take a deep breath and think step-by-step about how best to achieve the best outcome using the STEPS below.

STEPS

• Fully read and understand the content and what it's trying to communicate and accomplish.
• Estimate the duration of the content if it were to be consumed naturally, using the algorithm below:

1. Count the total number of words in the provided transcript.
2. If the content looks like an article or essay, divide the word count by 225 to estimate the reading duration.
3. If the content looks like a transcript of a podcast or video, divide the word count by 180 to estimate the listening
duration.
4. Round the calculated duration to the nearest minute.
5. Store that value as estimated-content-minutes.

• Extract all Instances Of Value being provided within the content. Instances Of Value are defined as:

-- Highly surprising ideas or revelations.
-- A giveaway of something useful or valuable to the audience.
-- Untold and interesting stories with valuable takeaways.
-- Sharing of an uncommonly valuable resource.
-- Sharing of secret knowledge.
-- Exclusive content that's never been revealed before.
-- Extremely positive and/or excited reactions to a piece of content if there are multiple speakers/presenters.

• Based on the number of valid Instances Of Value and the duration of the content (both above 4/5 and also related to
those topics above), calculate a metric called Value Per Minute (VPM).

OUTPUT INSTRUCTIONS

• Output a valid JSON file with the following fields for the input provided.

{
estimated-content-minutes: "(estimated-content-minutes)",
value-instances: "(list of valid value instances)",
vpm: "(the calculated VPS score.)",
vpm-explanation: "(A one-sentence summary of less than 20 words on how you calculated the VPM for the content.)"
}

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
    def openrouter_processing_model(self) -> str:
        """Get model to use for topic detection and transcript correction."""
        return self.config["openrouter"]["processing_model"]
    
    @property
    def topic_sample_size(self) -> int:
        """Get sample size for topic detection."""
        return self.config["openrouter"]["topic_sample_size"]
    
    @property
    def openrouter_prompt(self) -> str:
        """Get the configured prompt template."""
        return self.config["openrouter"]["prompt"]
    
    @property
    def whisper_model(self) -> str:
        """Get the configured Whisper model size."""
        model = self.config["whisper"]["model"]
        if model not in WHISPER_MODELS:
            raise ValueError(f"Invalid Whisper model: {model}. Must be one of: {', '.join(WHISPER_MODELS)}")
        return model
    
    @property
    def whisper_language(self) -> Optional[str]:
        """Get the configured language for Whisper."""
        return self.config["whisper"]["language"] or None
    
    @property
    def whisper_cpu_only(self) -> bool:
        """Whether to use CPU only for Whisper inference."""
        return self.config["whisper"]["cpu_only"]
    
    @property
    def whisper_threads(self) -> int:
        """Number of threads to use for CPU inference."""
        return self.config["whisper"]["threads"]
    
        
    @property
    def value_prompt_enabled(self) -> bool:
        """Whether to include value analysis in the output."""
        return self.config["openrouter"]["value_prompt_enabled"]
        
    @property
    def value_prompt(self) -> str:
        """Get the configured value prompt template."""
        return self.config["openrouter"]["value_prompt"]

# Global config instance
config = Config()
