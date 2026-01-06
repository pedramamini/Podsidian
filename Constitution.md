# Podsidian Constitution

This document defines the architectural principles, design patterns, and coding conventions that govern the Podsidian codebase. All contributions should adhere to these guidelines to maintain consistency and quality.

---

## Core Philosophy

**Podsidian bridges Apple Podcasts with Obsidian knowledge management through AI-powered transcription and semantic search.**

### Guiding Principles

1. **Graceful Degradation**: When one method fails, fall back to alternatives rather than failing completely
2. **Lazy Loading**: Load heavy resources (ML models, connections) only when actually needed
3. **User Visibility**: Provide real-time progress feedback for long-running operations
4. **Data Portability**: Store data in open formats (SQLite, Markdown, JSON) that users can access independently
5. **Configuration Over Code**: Prefer TOML configuration for behavior customization

---

## Architecture Overview

### Processing Pipeline

```
Apple Podcasts DB → RSS Feed Parsing → Audio Download →
Transcription (WhisperKit/Whisper) → Domain Detection →
Transcript Correction → Embedding Generation → Vector Index →
Summary/Analysis → Obsidian Export
```

### Module Responsibilities

| Module | Responsibility | Entry Point |
|--------|---------------|-------------|
| `cli.py` | User interface, command dispatch | `cli()` |
| `core.py` | Processing engine, orchestration | `PodcastProcessor` |
| `config.py` | Configuration loading, validation | `Config` |
| `models.py` | Database schema, ORM models | `Podcast`, `Episode` |
| `api.py` | HTTP API endpoints | `create_api()` |
| `stdio_server.py` | MCP STDIO server | `MCPServer` |
| `apple_podcasts.py` | Apple Podcasts database access | `get_subscriptions()` |
| `cost_tracker.py` | API cost monitoring | `track_api_call()` |
| `markdown.py` | Obsidian file generation | Markdown utilities |
| `backup.py` | Database backup/restore | Backup utilities |
| `migrate_db.py` | Schema migrations | Schema updates |

---

## Design Patterns

### 1. Lazy Loading Pattern

Heavy resources are loaded on first use, not at initialization.

```python
# GOOD: Load only when needed
def _load_whisper(self):
    if self.whisper_model is None:
        self.whisper_model = whisper.load_model(...)
    return self.whisper_model

# BAD: Load at initialization
def __init__(self):
    self.whisper_model = whisper.load_model(...)  # Always loads, even if unused
```

**Applied to**: Whisper model, embedding model, Annoy index

### 2. Strategy Pattern with Fallback

Multiple strategies for the same operation, with automatic fallback.

```python
# Transcription strategy chain:
# 1. Try external transcript from RSS feed
# 2. Try WhisperKit-CLI (fast on Apple Silicon)
# 3. Fall back to Python Whisper (universal)

try:
    result = self._transcribe_with_whisperkit(audio_path)
except Exception:
    # Notify user of fallback
    progress_callback({'stage': 'warning', 'message': 'Falling back to Python Whisper'})
    result = self._transcribe_audio(audio_path)
```

### 3. Callback-Based Progress Reporting

Long operations report progress via callbacks, decoupling UI from logic.

```python
# In core.py
def ingest_subscriptions(self, progress_callback=None):
    if progress_callback:
        progress_callback({'stage': 'downloading', 'progress': 0.5})

# In cli.py
def show_progress(info):
    if info['stage'] == 'downloading':
        click.echo("→ Downloading audio...")

processor.ingest_subscriptions(progress_callback=show_progress)
```

### 4. Dependency Injection

Sessions and dependencies are injected, not created internally.

```python
# GOOD: Accept session as parameter
class PodcastProcessor:
    def __init__(self, db_session):
        self.db = db_session

# BAD: Create session internally
class PodcastProcessor:
    def __init__(self):
        self.db = create_session()  # Harder to test, less flexible
```

### 5. Property-Based Configuration Access

Configuration values accessed via properties with type hints.

```python
class Config:
    @property
    def vault_path(self) -> Path:
        return Path(os.path.expanduser(self.config["obsidian"]["vault_path"]))

    @property
    def openrouter_api_key(self) -> Optional[str]:
        return self.config["openrouter"]["api_key"]
```

---

## Code Conventions

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `PodcastProcessor`, `MCPServer` |
| Functions/Methods | snake_case | `ingest_subscriptions`, `get_summary` |
| Private methods | Leading underscore | `_load_whisper`, `_generate_embedding` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_DB_PATH`, `WHISPER_MODELS` |
| Variables | snake_case | `audio_path`, `episode_title` |

### File Organization

```
podsidian/
├── __init__.py          # Package exports, version
├── cli.py               # Click commands (UI layer)
├── core.py              # PodcastProcessor (business logic)
├── config.py            # Configuration management
├── models.py            # SQLAlchemy ORM models
├── api.py               # FastAPI HTTP endpoints
├── stdio_server.py      # MCP STDIO server
├── apple_podcasts.py    # Apple Podcasts integration
├── cost_tracker.py      # API cost tracking
├── markdown.py          # Obsidian export utilities
├── backup.py            # Database backup
└── migrate_db.py        # Schema migrations
```

### Import Style

```python
# Standard library first
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Third-party packages
import click
import requests
from sqlalchemy import Column, String

# Relative imports for package modules
from .config import config
from .models import Podcast, Episode
```

**Circular Import Prevention**: Use deferred imports within functions when needed.

```python
# When circular import would occur
def some_function():
    from .cost_tracker import track_api_call  # Import inside function
    track_api_call(...)
```

### Type Hints

- Use type hints for function signatures on public methods
- Use `Optional[T]` for nullable parameters
- Use `List`, `Dict`, `Tuple` from `typing` module for Python 3.9 compatibility

```python
def semantic_search(
    self,
    query: str,
    limit: int = 10,
    relevance: Optional[float] = None
) -> List[Dict[str, Any]]:
```

### Code Style

- **Line Length**: 100 characters maximum
- **Formatter**: Ruff
- **Linter**: Ruff

```bash
ruff check .    # Lint
ruff format .   # Format
```

---

## Error Handling

### Exception Strategy

1. **Catch specific exceptions** where recovery is possible
2. **Let exceptions propagate** when caller should handle them
3. **Report via callbacks** during long operations
4. **Log errors** for debugging

```python
# Recovery with fallback
try:
    result = primary_method()
except SpecificException as e:
    logger.warning(f"Primary failed: {e}")
    result = fallback_method()

# Propagate when caller should decide
def load_config(path: str) -> dict:
    with open(path, 'rb') as f:
        return tomli.load(f)  # Let FileNotFoundError propagate
```

### Database Safety

```python
# Use no_autoflush for read operations that might trigger unwanted flushes
with self.db.no_autoflush:
    existing = self.db.query(Episode).filter_by(guid=guid).first()

# Commit after successful operations
self.db.add(episode)
self.db.commit()
```

---

## Configuration System

### Priority Order (highest to lowest)

1. Environment variables (`PODSIDIAN_OPENROUTER_API_KEY`)
2. TOML configuration file (`~/.config/podsidian/config.toml`)
3. Default values (`DEFAULT_CONFIG` in config.py)

### Configuration Sections

```toml
[whisper]
model = "base"           # Whisper model size
language = "en"          # Target language
cpu_only = false         # Force CPU mode
threads = 4              # Thread count

[openrouter]
api_key = ""             # Required for LLM features
model = "anthropic/claude-3.5-sonnet"
processing_model = "google/gemini-2.0-flash-001"

[transcript_correction]
enabled = true           # Enable AI correction
chunk_size = 4000        # Characters per chunk

[obsidian]
vault_path = "~/Documents/Obsidian/Podsidian"
template = "..."         # Markdown template

[annoy]
n_trees = 10             # Index trees (more = better accuracy)
metric = "angular"       # Distance metric
```

---

## Database Patterns

### Models

**Podcast**: Container for episodes, contains feed metadata and mute state

**Episode**: Individual podcast episode with transcript, embedding, and analysis

```python
class Podcast(Base):
    __tablename__ = 'podcasts'
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    feed_url = Column(String(512), unique=True, nullable=False)
    muted = Column(Boolean, default=False, nullable=False)
    episodes = relationship("Episode", back_populates="podcast")

class Episode(Base):
    __tablename__ = 'episodes'
    id = Column(Integer, primary_key=True)
    podcast_id = Column(Integer, ForeignKey('podcasts.id'))
    guid = Column(String(512), unique=True, nullable=False)
    transcript = Column(Text)
    vector_embedding = Column(Text)  # JSON-encoded
    rating = Column(String(10))      # S/A/B/C/D
    quality_score = Column(Integer)  # 1-100
```

### Migrations

Add columns without affecting existing data:

```python
# migrate_db.py pattern
inspector = inspect(engine)
existing_columns = [col['name'] for col in inspector.get_columns('episodes')]

if 'new_column' not in existing_columns:
    with engine.connect() as conn:
        conn.execute(text('ALTER TABLE episodes ADD COLUMN new_column TEXT'))
        conn.commit()
```

---

## CLI Patterns

### Click Command Structure

```python
@click.group()
def cli():
    """Podsidian - Apple Podcasts to Obsidian Bridge"""
    pass

@cli.command()
@click.option('--debug', is_flag=True, help='Enable debug output')
def ingest(debug):
    """Process new podcast episodes."""
    ...

@cli.group()
def subscriptions():
    """Manage subscriptions."""
    pass

@subscriptions.command(name='list')
def list_subscriptions():
    """List all subscriptions."""
    ...
```

### Progress Feedback

```python
# Colored status indicators
click.echo(click.style('✓', fg='green') + " Complete")
click.echo(click.style('✗', fg='red') + " Failed")
click.echo(click.style('→', fg='yellow') + " Processing...")

# Styled text
click.echo(click.style('Title:', bold=True) + " Episode Name")
click.echo(click.style('Info:', fg='bright_black') + " Details")
```

---

## AI/ML Integration

### Transcription Strategy Chain

1. **External Transcript**: Check RSS feed for provided transcript (JSON, WebVTT, SRT, text)
2. **WhisperKit-CLI**: Use if available (5-10x faster on Apple Silicon)
3. **Python Whisper**: Universal fallback

### Embedding Generation

- **Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Normalization**: L2 normalized for cosine similarity
- **Storage**: JSON-encoded in SQLite, Annoy index for fast search

### LLM Operations

- **Provider**: OpenRouter (supports multiple models)
- **Operations**: Summary, value analysis, domain detection, transcript correction
- **Cost Tracking**: Thread-local accumulation of token usage

```python
# Cost tracking pattern
from .cost_tracker import track_api_call

response = requests.post(api_url, json=payload)
track_api_call(
    model=model_name,
    prompt_tokens=response_data['usage']['prompt_tokens'],
    completion_tokens=response_data['usage']['completion_tokens']
)
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── test_core.py         # PodcastProcessor tests
├── test_config.py       # Configuration tests
├── test_models.py       # ORM model tests
└── test_api.py          # API endpoint tests

testing/
└── test_mcp_client.py   # Integration test client
```

### Test Patterns

- Use dependency injection to provide mock sessions
- Test fallback chains by simulating failures
- Verify database operations with rollback

---

## External Services

### OpenRouter API

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://github.com/pedramamini/podsidian",
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json={
        "model": model_name,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}]
    }
)
```

### Apple Podcasts Database

- Read-only access to SQLite database
- Location varies by macOS version
- Query `ZMTPODCAST` table for subscriptions

---

## File Paths

| Purpose | Path |
|---------|------|
| Configuration | `~/.config/podsidian/config.toml` |
| Database | `~/.local/share/podsidian/podsidian.db` |
| Backups | `~/.local/share/podsidian/backups/` |
| Vector Index | `~/.local/share/podsidian/` (Annoy files) |
| Obsidian Export | Configured `vault_path` |

---

## Contributing Checklist

- [ ] Code follows naming conventions
- [ ] Type hints on public functions
- [ ] No hardcoded paths (use config or constants)
- [ ] Progress callbacks for operations > 1 second
- [ ] Graceful fallback for optional features
- [ ] `ruff check .` passes
- [ ] `ruff format .` applied
- [ ] Database changes include migration
- [ ] New config options have defaults
