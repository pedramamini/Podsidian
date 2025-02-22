# Podsidian

Podsidian is a powerful tool that bridges your Apple Podcast subscriptions with Obsidian, creating an automated pipeline for podcast content analysis and knowledge management.

## Features

- **Apple Podcast Integration**: Automatically extracts and processes your Apple Podcast subscriptions
- **RSS Feed Processing**: Retrieves and parses podcast RSS feeds to discover new episodes
- **Smart Storage**:
  - SQLite3 database for episode metadata and full transcripts
  - Vector embeddings for efficient semantic search
  - Configurable Obsidian markdown export
- **Efficient Processing**: Downloads and transcribes episodes, then discards audio to save space
- **Transcription**: Leverages local Whisper installation for high-quality audio transcription
- **AI-Powered Analysis**: Uses OpenRouter to generate customized summaries and insights
- **Semantic Search**: Command-line interface for searching through podcast history using natural language
- **Obsidian Integration**: Generates markdown notes with customizable templates
- **AI Agent Integration**: Exposes an MCP (Message Control Program) service for AI agents

## Installation

```bash
# Clone the repository
git clone https://github.com/pedramamini/podsidian.git
cd podsidian

# Install using uv (recommended)
uv pip install hatch
uv pip install -e .

# Or using pip
pip install hatch
pip install -e .
```

Note: We use `hatch` as our build system. The `-e` flag installs the package in editable mode, which is recommended for development.

## Configuration

1. Initialize configuration:
```bash
podsidian init
```
This creates a config file at `~/.config/podsidian/config.toml`

2. Configure settings:
```toml
[obsidian]
# Path to your Obsidian vault
vault_path = "~/Documents/Obsidian"

# Customize note template
template = """
# {title}

## Metadata
- **Podcast**: {podcast_title}
- **Published**: {published_at}
- **URL**: {audio_url}

## Summary
{summary}

## Transcript
{transcript}
"""

[openrouter]
# Set via PODSIDIAN_OPENROUTER_API_KEY env var or here
api_key = ""

# Choose AI model
model = "anthropic/claude-2"

# Customize summary prompt
prompt = """Your custom prompt template here.
Available variable: {transcript}"""
```

## Usage

```bash
# Initialize configuration
podsidian init

# Process new episodes
podsidian ingest

# Export a specific episode transcript
podsidian export --transcript <episode_id>

# Search through podcast history
podsidian search "blockchain security"

# Start the MCP service
podsidian mcp --port 8080
```

## How It Works

1. **Podcast Discovery**:
   - Reads your Apple Podcast subscriptions
   - Fetches RSS feeds for each podcast
   - Identifies new episodes

2. **Content Processing**:
   - Downloads episodes temporarily
   - Transcribes using Whisper AI
   - Generates vector embeddings
   - Stores in SQLite database

3. **AI Processing**:
   - Generates summaries via OpenRouter
   - Uses customizable prompts
   - Creates semantic embeddings

4. **Knowledge Integration**:
   - Writes to Obsidian using templates
   - Organizes by podcast/episode
   - Enables semantic search

## MCP Service API

RESTful API for AI agent integration:

```bash
# Base URL
http://localhost:8080/api/v1

# Endpoints
GET  /search          # Semantic search across transcripts
GET  /episodes        # List all processed episodes
GET  /episodes/:id    # Get episode details and transcript
```

## Requirements

- Python 3.9+
- Local Whisper installation
- OpenRouter API access
- Apple Podcasts subscriptions
- Obsidian vault (optional)

## Development

```bash
# Setup development environment
./scripts/setup_dev.sh

# Activate environment
source .venv/bin/activate
```

Detailed configuration instructions and environment setup will be provided in the documentation.

## License

This project is open source and available under the MIT License.
