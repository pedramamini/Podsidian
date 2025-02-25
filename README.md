# Podsidian

Podsidian is a powerful tool that bridges your Apple Podcast subscriptions with Obsidian, creating an automated pipeline for podcast content analysis and knowledge management.

## Features

- **Apple Podcast Integration**: 
  - Automatically extracts and processes your Apple Podcast subscriptions
  - Smart episode filtering with configurable lookback period
  - Easy subscription management and episode listing
- **RSS Feed Processing**: 
  - Retrieves and parses podcast RSS feeds to discover new episodes
  - Defaults to processing only recent episodes (last 7 days)
  - Configurable lookback period for older episodes
- **Smart Storage**:
  - SQLite3 database for episode metadata and full transcripts
  - Annoy vector index for fast semantic search (inspired by Spotify)
  - Vector embeddings for efficient content discovery
  - Configurable Obsidian markdown export
- **Efficient Processing**: Downloads and transcribes episodes, then discards audio to save space
- **Smart Transcription Pipeline**:
  - Initial transcription using OpenAI's Whisper
  - Automatic domain detection (e.g., Brazilian Jiu-Jitsu, Quantum Physics)
  - Domain-aware transcript correction for technical terms and jargon
  - High-quality output optimized for each podcast's subject matter
- **AI-Powered Analysis**: Uses OpenRouter to generate customized summaries and insights
- **Natural Language Search**: 
  - Fast semantic search powered by Spotify's Annoy library
  - Intelligent search that understands the meaning of your queries
  - Finds relevant content even when exact words don't match
  - Configurable relevance threshold for fine-tuning results
  - Results grouped by podcast with relevant excerpts
- **Obsidian Integration**: Generates markdown notes with customizable templates
- **AI Agent Integration**: Exposes an MCP (Message Control Program) service for AI agents

## Installation

```bash
# Clone the repository
git clone https://github.com/pedramamini/podsidian.git
cd podsidian

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install hatch
uv pip install -e .

# Or if you prefer using regular pip
python -m venv .venv
source .venv/bin/activate
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

[annoy]
# Path to vector index file
index_path = "~/.config/podsidian/annoy.idx"

# Number of trees (more = better accuracy but slower build)
n_trees = 10

# Distance metric (angular = cosine similarity)
metric = "angular"
```

## Usage

```bash
# Initialize configuration
podsidian init

# Show configuration and system status
podsidian show-config    # Displays config, vector index status, and episode stats

# Manage podcast subscriptions
podsidian subscriptions list    # List all subscriptions and their mute state
podsidian subscriptions mute "Podcast Title"    # Mute a podcast (skip during ingestion)
podsidian subscriptions unmute "Podcast Title"  # Unmute a podcast

# List all downloaded episodes
podsidian episodes

# Process new episodes (last 7 days by default)
podsidian ingest

# Process episodes from last 30 days with debug output
podsidian ingest --lookback 30 --debug

# Process episodes with detailed debug information
podsidian ingest --debug

# Export a specific episode transcript
podsidian export <episode_id>

# Search through podcast content using natural language (default 30% relevance)
podsidian search "impact of blockchain on cybersecurity"

# Search with custom relevance threshold (0-100)
podsidian search "meditation techniques for beginners" --relevance 75

# Force refresh of search index before searching
podsidian search "blockchain" --refresh

# Start the MCP service
podsidian mcp --port 8080

# Manage database backups
podsidian backup create           # Create a new backup with timestamp
podsidian backup list            # List all available backups
podsidian backup restore 2025-02-24  # Restore from a specific date
```

## Database Backup

Podsidian includes a robust backup system to help you safeguard your podcast database:

- **Automatic Timestamping**: Backups are automatically named with YYYY-MM-DD format
- **Multiple Daily Backups**: System automatically handles multiple backups on the same day by adding an index
- **Safe Restore Process**: Creates temporary backup before restore in case of failures
- **Backup Location**: All backups are stored in `~/.local/share/podsidian/backups`

### Commands

```bash
# Create a new backup
podsidian backup create

# List all backups with sizes and dates
podsidian backup list

# Restore from a specific date
podsidian backup restore 2025-02-24
```

When restoring a backup, Podsidian will:
1. Show size difference between current and backup database
2. Display time difference between current and backup
3. Require explicit confirmation before proceeding
4. Create a temporary backup of your current database as a safety measure

## System Status

Use the `show-config` command to view the current state of your Podsidian installation:

```bash
podsidian show-config
```

This will display:
- Vector index location and size
- Number of total episodes
- Number of episodes with embeddings
- Other configuration settings

## How It Works

1. **Podcast Discovery**:
   - Reads your Apple Podcast subscriptions
   - Fetches RSS feeds for each podcast
   - Identifies new episodes

2. **Content Processing**:
   - Downloads episodes temporarily
   - Transcribes using Whisper AI
   - Generates vector embeddings
   - Updates Annoy vector index
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
GET  /search                            # Natural language search across transcripts
GET  /episodes                           # List all processed episodes
GET  /episodes/:id                       # Get episode details and transcript
GET  /subscriptions                      # List all subscriptions with mute state
POST /subscriptions/:title/mute          # Mute a podcast subscription
POST /subscriptions/:title/unmute        # Unmute a podcast subscription
```

## Requirements

- Python 3.9+
- OpenRouter API access
- Apple Podcasts subscriptions
- Obsidian vault (optional)

### Installing Whisper

Whisper requires FFmpeg for audio processing. Install it first:

```bash
# On macOS using Homebrew
brew install ffmpeg

# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg
```

Whisper also requires PyTorch. For optimal performance with GPU support:

```bash
# For CUDA (NVIDIA GPU)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only or M1/M2 Macs
uv pip install torch torchvision torchaudio
```

The main Whisper package will be installed automatically as a dependency of Podsidian. The first time you run transcription, it will download the model files (size varies by model choice).

### Configuring Whisper

Whisper can be configured in your `config.toml`:

```toml
[whisper]
# Choose model size based on your needs
model = "large-v3"  # Options: tiny, base, small, medium, large, large-v3

# Optionally specify language (auto-detected if not set)
language = "en"  # Use language codes like "en", "es", "fr", etc.

# Performance settings
cpu_only = false  # Set to true to force CPU usage
threads = 4      # Number of CPU threads when using CPU
```

Model size trade-offs:
- **tiny**: 1GB VRAM, fastest, least accurate
- **base**: 1GB VRAM, good balance for most uses
- **small**: 2GB VRAM, better accuracy
- **medium**: 5GB VRAM, high accuracy
- **large**: 10GB VRAM, very high accuracy
- **large-v3**: 10GB VRAM, highest accuracy, improved performance (default)

## Smart Transcript Processing

Podsidian uses a sophisticated pipeline to ensure high-quality transcripts:

1. **Initial Transcription**: Uses Whisper to convert audio to text
2. **Domain Detection**: Analyzes a sample of the transcript to identify the podcast's domain (e.g., Brazilian Jiu-Jitsu, Quantum Physics, Constitutional Law)
3. **Expert Correction**: Uses domain expertise to fix technical terms, jargon, and specialized vocabulary
4. **Final Processing**: The corrected transcript is then summarized and stored

This is particularly useful for:
- Technical podcasts with specialized terminology
- Academic discussions with field-specific jargon
- Sports content with unique moves and techniques
- Medical or scientific podcasts with complex terminology

For example, in a Brazilian Jiu-Jitsu podcast, it will correctly handle terms like:
- Gi, Omoplata, De La Riva, Berimbolo
- Practitioner and technique names
- Portuguese terminology

Configure the processing in your `config.toml`:
```toml
[openrouter]
# API key (required)
api_key = "your-api-key"  # Or set PODSIDIAN_OPENROUTER_API_KEY env var

# Model settings
model = "openai/gpt-4"             # Model for summarization
processing_model = "openai/gpt-4"  # Model for domain detection and corrections
topic_sample_size = 16000          # Characters to analyze for domain detection

[search]
# Default relevance threshold for semantic search (0-100)
default_relevance = 60

# Override relevance thresholds for specific queries
relevance_overrides = [
  { query = "technical details", threshold = 75 },
  { query = "general discussion", threshold = 40 }
]
```

## Performance Tips
1. Use GPU if available (default behavior)
2. If using CPU, adjust `threads` based on your system
3. Choose model size based on your available memory and accuracy needs
4. Specify language if known for better accuracy

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
