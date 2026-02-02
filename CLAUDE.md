# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Podsidian is a sophisticated podcast content management system that bridges Apple Podcasts with Obsidian knowledge management. It processes podcast episodes through AI-powered transcription, semantic search, and exports structured markdown files to Obsidian vaults.

## Tech Stack

- **Language**: Python 3.9+
- **Build System**: Hatch with UV package manager
- **Database**: SQLite + SQLAlchemy ORM
- **AI/ML**: OpenAI Whisper, Sentence Transformers, Annoy vector search
- **Web APIs**: FastAPI + Uvicorn
- **CLI**: Click framework with Rich progress indicators
- **External Services**: OpenRouter for LLM operations

## Common Development Commands

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv pip install hatch && uv pip install -e .
# Alternative: ./scripts/setup_dev.sh

# Core operations
podsidian init                    # Initialize configuration
podsidian show-config            # Display current config and status
podsidian ingest --debug         # Process new episodes with debug output
podsidian search "query"         # Perform semantic search
podsidian briefing               # Generate personalized news briefing
podsidian mcp --stdio           # Start MCP server for AI agents

# Database operations
podsidian backup create         # Create database backup
podsidian backup list          # List available backups
podsidian backup restore       # Restore from backup

# Linting and formatting
ruff check .                   # Check code style (100-char line limit)
ruff format .                  # Format code
```

## High-Level Architecture

### Core Processing Pipeline
1. **Apple Podcasts Integration** → Fetch episode metadata and audio
2. **Audio Processing** → Whisper transcription with domain detection
3. **AI Correction** → LLM-powered transcript refinement using detected domain context
4. **Embedding Generation** → Sentence transformers for semantic vectors
5. **Vector Storage** → Annoy index for fast similarity search
6. **Obsidian Export** → Structured markdown with metadata and tags

### Key Components

- **PodcastProcessor** (`core.py`): Main processing engine that orchestrates the entire pipeline
- **CLI Interface** (`cli.py`): Click-based command system with rich progress indicators
- **Database Models** (`models.py`): SQLAlchemy schema for Podcasts and Episodes
- **MCP Server** (`stdio_server.py`): Model Context Protocol implementation for AI agent integration
- **Configuration System** (`config.py`): TOML-based config management with validation

### Important Entry Points

- **Main CLI**: `podsidian/cli.py:cli()` - Primary command interface
- **Core Engine**: `podsidian/core.py:PodcastProcessor` - Main processing class
- **MCP Server**: `podsidian/stdio_server.py` - AI agent integration endpoint
- **API Server**: `podsidian/api.py` - HTTP API for web integrations

## Configuration and Data

- **Config File**: `~/.config/podsidian/config.toml` (use `config.toml.example` as template)
- **Database**: `~/.local/share/podsidian/podsidian.db` (SQLite)
- **Vector Index**: Stored alongside database for semantic search
- **Audio Cache**: Configurable location for downloaded episode audio

## Development Notes

### Database Management
- Uses SQLAlchemy ORM with manual migration system
- Run `podsidian/migrate_db.py` for schema updates
- Backup system includes safety checks and metadata validation

### AI Integration
- Domain detection improves transcription accuracy by providing context to correction LLM
- Cost tracking monitors OpenRouter API usage across all operations
- MCP protocol enables integration with Claude Desktop and other AI agents

### Testing
- MCP server tests: `test_mcp_server.py` and `testing/test_mcp_client.py`
- Use `--debug` flag for detailed operation logging
- Database operations include transaction rollback on errors

### Code Style
- Configured for 100-character line limit
- Use Ruff for linting and formatting
- Python 3.9+ features expected
- Type hints preferred for new code