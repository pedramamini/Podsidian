# Changelog

## [1.1.0] - 2026-02-01

### New Features

- **Briefing Command**: Added `podsidian briefing` for generating personalized news briefings from recent podcast episodes. Supports configurable interest categories (cybersecurity, AI, startups, investing, science, health), relevance thresholds, lookback windows, and LLM synthesis prompts. Also exposed as an MCP tool.

- **WhisperKit-CLI Integration**: Added support for WhisperKit-CLI as the preferred transcription engine on Apple Silicon. Transcription now follows a priority order: RSS transcript > WhisperKit-CLI > Python Whisper. Includes automatic detection, configurable language/debug modes, and local model path support.

- **Reingest Command**: Added `podsidian reingest <episode_id> [...]` to re-process episodes from scratch — re-downloads audio, re-transcribes (using WhisperKit-CLI if available), regenerates embeddings, and re-exports to Obsidian. Useful for fixing truncated transcripts and testing configuration changes.

### Improvements

- **Configurable Transcript Correction**: Added chunked transcript correction with configurable settings.
- **Briefing Configuration**: New `[briefing]` config section with customizable categories, labels, relevance thresholds, lookback days, and LLM synthesis prompt.
- **Code Quality**: Applied ruff formatting across core modules (cli.py, config.py, core.py, stdio_server.py).
- **Documentation**: Updated README with WhisperKit-CLI installation instructions, performance benchmarks, reingest command documentation, and briefing command usage.

### Bug Fixes

- Fixed a processing bug in core.py.

## [1.0] - 2025-07-12

Initial tagged release with core functionality:
- Apple Podcasts integration and episode metadata fetching
- Whisper-based audio transcription with domain detection
- LLM-powered transcript correction via OpenRouter
- Semantic search with sentence transformers and Annoy vector index
- Obsidian markdown export with metadata and tags
- MCP server for AI agent integration
- FastAPI HTTP API
- SQLite database with SQLAlchemy ORM
- Click CLI with rich progress indicators
- Database backup and restore system
- API cost tracking
