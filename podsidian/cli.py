import os
import click
import uvicorn
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import init_db
from .core import PodcastProcessor
from .api import create_api

import shutil
from .config import config

# Get default paths
DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/podsidian/podsidian.db")

def get_db_session():
    """Initialize database and return session."""
    # Ensure directory exists
    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    
    # Initialize database
    engine = init_db(DEFAULT_DB_PATH)
    Session = sessionmaker(bind=engine)
    return Session()

@click.group()
def cli():
    """Podsidian - Apple Podcasts to Obsidian Bridge"""
    pass

@cli.command()
def init():
    """Initialize Podsidian configuration."""
    config_dir = os.path.dirname(config.config_path)
    os.makedirs(config_dir, exist_ok=True)
    
    if os.path.exists(config.config_path):
        click.confirm("Configuration file already exists. Overwrite?", abort=True)
    
    # Copy example config
    example_config = os.path.join(os.path.dirname(__file__), '..', 'config.toml.example')
    shutil.copy2(example_config, config.config_path)
    click.echo(f"Created configuration file at: {config.config_path}")
    click.echo("Please edit this file to configure your OpenRouter API key and preferences.")

@cli.command()
def ingest():
    """Process new episodes from Apple Podcasts subscriptions."""
    session = get_db_session()
    processor = PodcastProcessor(session)
    processor.ingest_subscriptions()
    click.echo("Ingestion complete!")

@cli.command()
@click.argument('query')
def search(query):
    """Search through podcast history."""
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    results = processor.search(query)
    for result in results:
        click.echo(f"\n{result['podcast']} - {result['episode']}")
        click.echo(f"Published: {result['published_at']}")
        click.echo(f"Relevance: {result['similarity']:.2f}")
        if result['transcript']:
            click.echo("\nRelevant transcript excerpt:")
            # Show first 200 characters of transcript
            click.echo(result['transcript'][:200] + "...")

@cli.command()
@click.option('--transcript', help='Episode ID to export transcript')
def export(transcript):
    """Export episode transcript to stdout."""
    if not transcript:
        click.echo("Please specify an episode ID")
        return
        
    session = get_db_session()
    episode = session.query(Episode).filter_by(id=int(transcript)).first()
    
    if not episode:
        click.echo("Episode not found")
        return
        
    if not episode.transcript:
        click.echo("No transcript available for this episode")
        return
        
    click.echo(episode.transcript)

@cli.command()
@click.option('--port', default=8080, help='Port to run the MCP service on')
def mcp(port):
    """Start the MCP service for AI agent integration."""
    session = get_db_session()
    app = create_api(session)
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    cli()
