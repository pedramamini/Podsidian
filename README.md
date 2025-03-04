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
- **Efficient Processing**:
  - Downloads and transcribes episodes, then discards audio to save space
- **Smart Transcription Pipeline**:
  - Automatic detection and use of external transcripts when available
  - Fallback to OpenAI's Whisper for transcription when needed
  - Automatic domain detection (e.g., Brazilian Jiu-Jitsu, Quantum Physics)
  - Domain-aware transcript correction for technical terms and jargon
  - High-quality output optimized for each podcast's subject matter
- **AI-Powered Analysis**:
  - Uses OpenRouter to generate customized summaries and insights
  - Monitor and report expenses for all AI API calls with detailed breakdowns by model and operation
- **Natural Language Search**:
  - Fast semantic search powered by Spotify's Annoy library
  - Intelligent search that understands the meaning of your queries
  - Finds relevant content even when exact words don't match
  - Configurable relevance threshold for fine-tuning results
  - Results grouped by podcast with relevant excerpts
- **Obsidian Integration**:
  - Generates markdown notes with customizable templates
- **AI Agent Integration**:
  - Exposes an MCP (Message Control Program) service for AI agents

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

### OSX XCode Issues

If you have build issues, try:

```
sudo xcode-select --reset
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

sudo xcodebuild -license

export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
export CFLAGS="-isysroot $SDKROOT -I$SDKROOT/usr/include"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-L$SDKROOT/usr/lib"
export PATH="$SDKROOT/usr/bin:$PATH"

rm -rf ~/.cache/uv/builds-v0
uv pip install -e .
```

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

# Template for generated notes
# Available variables: {title}, {podcast_title}, {published_at}, {audio_url}, {podcasts_app_url}, {summary}, {value_analysis}, {transcript}, {episode_id}, {episode_wordcount}, {podcast_guid}
template = """
{title}

# Metadata
- **Podcast**: {podcast_title}
- **Published**: {published_at}
- **URL**: {audio_url}
- **Open in Podcasts App**: {podcasts_app_url}
- **Podcast GUID**: {podcast_guid}

# Summary
{summary}

# Value Analysis
{value_analysis}

# Transcript
{transcript}
"""

[whisper]
# Model size to use for transcription
# Options: tiny, base, small, medium, large, large-v3
# Larger models are more accurate but slower and use more memory
# Model sizes and VRAM requirements:
# - tiny: 1GB VRAM, fastest, least accurate
# - base: 1GB VRAM, good balance for most uses
# - small: 2GB VRAM, better accuracy
# - medium: 5GB VRAM, high accuracy
# - large: 10GB VRAM, very high accuracy
# - large-v3: 10GB VRAM, highest accuracy, improved performance
model = "medium.en"

# Language to use for transcription (optional)
# If not specified, Whisper will auto-detect the language
# Example: "en" for English, "es" for Spanish, etc.
language = ""

# Use CPU instead of GPU for inference
# Set to true if you don't have a GPU or encounter GPU memory issues
cpu_only = false

# Number of threads to use for CPU inference
# Default is 4, increase for faster CPU processing if available
threads = 4

[openrouter]
# OpenRouter API configuration
# API key can also be set via PODSIDIAN_OPENROUTER_API_KEY environment variable
api_key = ""

# Model to use for topic detection and transcript correction
processing_model = "openai/gpt-4o"

# Sample size in characters for topic detection
topic_sample_size = 4096

# Model to use for summarization
# See https://openrouter.ai/docs for available models
model = "openai/gpt-4o"

# Enable cost tracking for AI API calls
# When enabled, displays cost summary after operations that use AI
cost_tracking_enabled = true

# Prompt template for processing transcripts
# Available variables: {transcript}
prompt = """You are a helpful podcast summarizer.
Given the following podcast transcript, provide:
1. A concise 2-3 paragraph summary of the key points
2. A bullet list of the most important takeaways
3. Any notable quotes, properly attributed

Transcript:
{transcript}
"""

# Enable value analysis in output
# When enabled, each episode will include a Value Per Minute (VPM) analysis
value_prompt_enabled = true

# Value analysis prompt template
# This prompt analyzes the transcript to determine its value density
# This prompt is from Daniel Miessler's Fabric (https://github.com/danielmiessler/fabric)
# Available variables: {transcript}

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
podsidian subscriptions list              # List all subscriptions (sorted alphabetically)
podsidian subscriptions list --sort=episodes  # List all subscriptions (sorted by episode count)
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

# Start the MCP service (HTTP mode)
podsidian mcp --port 8080

# Start the MCP service in STDIO mode for AI agent integration
podsidian mcp --stdio

# Manage database backups
podsidian backup create           # Create a new backup with timestamp
podsidian backup list            # List all available backups
podsidian backup restore 2025-02-24  # Restore from a specific date
```

## Database Migration

When upgrading to a new version of Podsidian that includes database schema changes, you can use the included migration script:

```bash
# Run the database migration script
python -m podsidian.migrate_db

# Specify a custom database path if needed
python -m podsidian.migrate_db --db-path /path/to/your/database.db
```

The migration script will safely add new columns to the database without affecting existing data.

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
- AI cost tracking status
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

## MCP Service

Podsidian provides an MCP (Message Control Program) service for AI agent integration with two operating modes:

### HTTP Mode

RESTful API accessible via HTTP:

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

### STDIO Mode

STDIO mode enables direct integration with AI agents like Claude Desktop through standard input/output:

```bash
# Start in STDIO mode
podsidian mcp --stdio

# With configuration
podsidian mcp --stdio --config '{"vaultPath":"/path/to/vault"}'
```

This mode can be used with tools like Smithery CLI:

```bash
npx -y @smithery/cli run podsidian-mcp --config '{"vaultPath":"/path/to/vault"}'
```

The STDIO server exposes the same functionality as the HTTP API but through a JSON-based message protocol over stdin/stdout.

### Claude Desktop Integration

To set up Claude Desktop with Podsidian support:

1. Install Claude Desktop from [Anthropic's website](https://claude.ai/desktop)

2. Open Claude Desktop and go to Settings (gear icon) > Advanced > Custom Tools

3. Click "Add Tool" and configure as follows:

   ```json
   {
     "podsidian": {
       "command": "/Users/pedram/Projects/Podsidian/.venv/bin/podsidian",
       "args": [
         "mcp",
         "--stdio"
       ]
     }
   }
   ```

4. Click "Save"

5. You can now ask Claude to search your podcast content with queries like:
   - "Search my podcasts for discussions about artificial intelligence"
   - "Find podcast episodes about climate change"
   - "Get the transcript of episode 42"

### Other AI Agent Integrations

Podsidian's STDIO mode can be integrated with any AI agent that supports the STDIO protocol for tools:

#### Using with Smithery CLI

```bash
npx -y @smithery/cli run podsidian
```

#### Using with LangChain

```python
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import subprocess
import json

def search_podcasts(query, limit=10, relevance=25):
    """Search podcast transcripts using natural language"""
    cmd = ["podsidian", "mcp", "--stdio"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Send tool call message
    message = {
        "type": "tool_call",
        "data": {
            "name": "search-semantic",
            "parameters": {
                "query": query,
                "limit": limit,
                "relevance": relevance
            },
            "id": "search-1"
        }
    }

    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()

    # Read response
    response = json.loads(proc.stdout.readline())
    proc.terminate()

    return response["data"]["result"]

# Create LangChain tool
podsidian_tool = StructuredTool.from_function(
    func=search_podcasts,
    name="search_podcasts",
    description="Search through podcast transcripts using natural language"
)

# Create agent with the tool
llm = ChatOpenAI(model="gpt-4")
tools = [podsidian_tool]
prompt = ChatPromptTemplate.from_messages([...])
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
agent_executor.invoke({"input": "Find podcast discussions about climate change"})
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

# Length of excerpt to show in search results (in characters)
excerpt_length = 300

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
