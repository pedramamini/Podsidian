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

@cli.group()
def subscriptions():
    """Manage Apple Podcasts subscriptions."""
    pass

@subscriptions.command(name='list')
def list_subscriptions():
    """List all Apple Podcasts subscriptions."""
    from .apple_podcasts import get_subscriptions
    from .models import Podcast
    session = get_db_session()
    
    # Get subscriptions from Apple Podcasts
    subs = get_subscriptions()
    if not subs:
        click.echo("No Apple Podcasts subscriptions found.")
        return
    
    # Ensure all podcasts exist in database
    for sub in subs:
        podcast = session.query(Podcast).filter_by(feed_url=sub['feed_url']).first()
        if not podcast:
            podcast = Podcast(
                title=sub['title'],
                author=sub['author'],
                feed_url=sub['feed_url'],
                muted=False
            )
            session.add(podcast)
    session.commit()
    
    # Get mute states from database
    muted_feeds = {p.feed_url: p.muted for p in session.query(Podcast).all()}
    
    # Split into muted and unmuted lists
    muted_subs = []
    unmuted_subs = []
    for sub in sorted(subs, key=lambda x: x['title']):
        if muted_feeds.get(sub['feed_url'], False):
            muted_subs.append(sub)
        else:
            unmuted_subs.append(sub)
    
    # Show active subscriptions
    click.echo("\nActive Subscriptions:")
    click.echo("-" * 30)
    if unmuted_subs:
        for sub in unmuted_subs:
            click.echo(f"• {sub['title']}")
    else:
        click.echo("No active subscriptions")
    
    # Show muted subscriptions
    click.echo("\nMuted Subscriptions:")
    click.echo("-" * 30)
    if muted_subs:
        for sub in muted_subs:
            click.echo(f"• {sub['title']}")
    else:
        click.echo("No muted subscriptions")
    click.echo()

@subscriptions.command()
@click.argument('title')
def mute(title):
    """Mute a podcast subscription by title.
    
    The podcast will not be ingested until unmuted.
    """
    session = get_db_session()
    podcast = session.query(Podcast).filter(Podcast.title.ilike(f"%{title}%")).first()
    
    if not podcast:
        click.echo(f"No podcast found matching title: {title}")
        return
    
    podcast.muted = True
    session.commit()
    click.echo(f"Muted podcast: {podcast.title}")

@subscriptions.command()
@click.argument('title')
def unmute(title):
    """Unmute a podcast subscription by title.
    
    The podcast will be included in future ingests.
    """
    session = get_db_session()
    podcast = session.query(Podcast).filter(Podcast.title.ilike(f"%{title}%")).first()
    
    if not podcast:
        click.echo(f"No podcast found matching title: {title}")
        return
    
    podcast.muted = False
    session.commit()
    click.echo(f"Unmuted podcast: {podcast.title}")

@cli.command()
def episodes():
    """List all downloaded episodes."""
    session = get_db_session()
    from .models import Episode, Podcast
    
    episodes = session.query(Episode).join(Podcast).order_by(Podcast.title, Episode.published_at.desc()).all()
    
    if not episodes:
        click.echo("No episodes found in database.")
        return
    
    current_podcast = None
    for episode in episodes:
        if episode.podcast.title != current_podcast:
            current_podcast = episode.podcast.title
            click.echo(f"\n{current_podcast}:")
            click.echo("-" * len(current_podcast) + "-" * 1)
        
        date_str = episode.published_at.strftime("%Y-%m-%d") if episode.published_at else "No date"
        status = "✓" if episode.transcript else " "
        # Add episode ID in a visually distinct way
        click.echo(f"[{status}] {click.style(f'#{episode.id:04d}', fg='bright_blue')} {date_str} - {episode.title}")
    
    click.echo("\nUse 'podsidian export <episode_id>' to export a transcript")
    click.echo()

@cli.command()
@click.option('--lookback', type=int, default=7, help='Number of days to look back for episodes (default: 7)')
@click.option('--debug', is_flag=True, help='Enable debug output')
def ingest(lookback, debug):
    """Process new episodes from Apple Podcasts subscriptions.
    
    By default, only processes episodes published in the last 7 days.
    Use --lookback to override this (e.g. --lookback 30 for last 30 days).
    """
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    if lookback <= 0:
        click.echo("Error: Lookback days must be greater than 0")
        return
    
    if lookback > 7:
        click.confirm(f"Warning: Looking back {lookback} days may take a while. Continue?", abort=True)
    
    click.echo(f"Ingesting episodes from the last {lookback} days...")
    
    # Track current podcast and episode progress
    current_podcast = None
    total_podcasts = 0
    
    def show_progress(info):
        nonlocal current_podcast, total_podcasts
        
        stage = info['stage']
        
        if stage == 'init':
            total_podcasts = info['total_podcasts']
            click.echo(f"Found {total_podcasts} podcast subscriptions")
            
        elif stage == 'podcast':
            podcast = info['podcast']
            current = info['current']
            total = info['total']
            
            current_podcast = podcast['title']
            click.echo(f"\n[{current}/{total}] Processing podcast: {click.style(current_podcast, fg='blue', bold=True)}")
            
        elif stage == 'episodes_found':
            podcast = info['podcast']
            total = info['total']
            click.echo(f"Found {total} recent episodes")
            
        elif stage == 'episode_start':
            episode = info['episode']
            current = info['current']
            total = info['total']
            published = episode['published_at'].strftime('%Y-%m-%d') if episode.get('published_at') else 'Unknown date'
            
            click.echo(f"\n  Episode [{current}/{total}] {published}")
            click.echo(f"  {click.style('Title:', fg='bright_black')} {episode['title']}")
            
        elif stage == 'downloading':
            click.echo(f"  {click.style('→', fg='yellow')} Downloading audio...")
            
        elif stage == 'transcribing':
            click.echo(f"  {click.style('→', fg='yellow')} Transcribing audio...")
            
        elif stage == 'transcribing_progress':
            progress = info['progress']
            width = 30
            filled = int(width * progress)
            bar = '=' * filled + '-' * (width - filled)
            percentage = int(progress * 100)
            # Use carriage return to update in place
            click.echo(f"\r  {click.style('→', fg='yellow')} Transcribing: [{bar}] {percentage}%", nl=False)
            
        elif stage == 'embedding':
            click.echo(f"  {click.style('→', fg='yellow')} Generating embeddings...")
            
        elif stage == 'exporting':
            click.echo(f"  {click.style('→', fg='yellow')} Exporting to Obsidian...")
            
        elif stage == 'episode_complete':
            click.echo(f"  {click.style('✓', fg='green')} Processing complete")
            
        elif stage == 'debug':
            click.echo(f"  {click.style('🔍', fg='bright_black')} {info['message']}")
            
        elif stage == 'error':
            # Make sure we're on a new line
            click.echo()
            click.echo(f"  {click.style('✗', fg='red')} {info['error']}")
    
    processor.ingest_subscriptions(lookback_days=lookback, progress_callback=show_progress, debug=debug)
    click.echo("\n\nIngestion complete!")

@cli.group()
def search():
    """Search through podcast history."""
    pass

@search.command('semantic')
@click.argument('query')
@click.option('--relevance', type=int, default=25, help='Minimum relevance score (0-100) for results')
def semantic_search(query, relevance):
    """Search using semantic similarity (embeddings)."""
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    # Convert relevance to 0-1 scale
    relevance_float = relevance / 100.0
    results = processor.search(query, relevance_threshold=relevance_float)
    
    # Sort by relevance score
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    if not results:
        click.echo("No results found matching your query with the current relevance threshold.")
        click.echo(f"Try lowering the threshold (current: {relevance}%)")
        return
        
    for result in results:
        click.echo(f"\n{result['podcast']} - {result['episode']}")
        click.echo(f"Published: {result['published_at']}")
        # Convert similarity back to percentage
        similarity_pct = int(result['similarity'] * 100)
        click.echo(f"Relevance: {similarity_pct}%")
        if result['transcript']:
            click.echo("\nRelevant transcript excerpt:")
            # Show first 200 characters of transcript
            click.echo(result['transcript'][:200] + "...")

@search.command('keyword')
@click.argument('keyword')
def keyword_search(keyword):
    """Search for exact keyword matches in transcripts."""
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    results = processor.keyword_search(keyword)
    
    if not results:
        click.echo("No exact matches found for your keyword.")
        return
        
    for result in results:
        click.echo(f"\n{result['podcast']} - {result['episode']}")
        click.echo(f"Published: {result['published_at']}")
        click.echo(f"\nMatching excerpt:")
        click.echo(result['excerpt'])

@cli.command()
@click.argument('episode_id', type=int)
def export(episode_id):
    """Export episode transcript to stdout.
    
    EPISODE_ID is the numeric ID shown in the episodes list (e.g. 42)
    """
    session = get_db_session()
    from .models import Episode, Podcast
    
    episode = session.query(Episode).join(Podcast).filter(Episode.id == episode_id).first()
    
    if not episode:
        click.echo(f"Error: Episode #{episode_id:04d} not found")
        return
        
    if not episode.transcript:
        click.echo(f"Error: No transcript available for episode #{episode_id:04d}")
        return
    
    # Print episode info header
    click.echo(click.style(f"\n{episode.podcast.title}", fg='blue', bold=True))
    click.echo(click.style(f"{episode.title}", fg='bright_black'))
    if episode.published_at:
        click.echo(click.style(f"Published: {episode.published_at.strftime('%Y-%m-%d')}", fg='bright_black'))
    click.echo("\n" + "=" * 40 + "\n")
    
    # Print transcript
    click.echo(episode.transcript)

@cli.command()
@click.option('--port', default=8080, help='Port to run the MCP service on')
def mcp(port):
    """Start the MCP service for AI agent integration."""
    session = get_db_session()
    app = create_api(session)
    uvicorn.run(app, host="0.0.0.0", port=port)

@cli.command(name='show-config')
def show_config():
    """Display current configuration settings."""
    def format_section(title, items, indent=0):
        click.echo("\n" + " " * indent + click.style(f"[{title}]", fg='green', bold=True))
        for key, value in items:
            # Skip the template and prompt as they're too long
            if key in ['template', 'prompt']:
                value = '<configured>' if value else '<not configured>'
            # Mask API key
            elif key == 'api_key':
                value = '***' + value[-4:] if value else '<not set>'
            click.echo(" " * (indent + 2) + click.style(f"{key}: ", fg='bright_black') + str(value))
    
    click.echo(click.style("\nPodsidian Configuration", fg='blue', bold=True))
    click.echo(click.style("=" * 21, fg='blue'))
    
    # Obsidian Settings
    obsidian_items = [
        ('vault_path', config.vault_path),
        ('template', config.note_template)
    ]
    format_section('Obsidian', obsidian_items)
    
    # Whisper Settings
    whisper_items = [
        ('model', config.whisper_model),
        ('language', config.whisper_language or '<auto>'),
        ('cpu_only', config.whisper_cpu_only),
        ('threads', config.whisper_threads)
    ]
    format_section('Whisper', whisper_items)
    
    # OpenRouter Settings
    openrouter_items = [
        ('api_key', config.openrouter_api_key),
        ('model', config.openrouter_model),
        ('processing_model', config.openrouter_processing_model),
        ('topic_sample_size', config.topic_sample_size),
        ('prompt', config.openrouter_prompt)
    ]
    format_section('OpenRouter', openrouter_items)
    
    # Database Settings
    db_items = [
        ('path', DEFAULT_DB_PATH),
    ]
    format_section('Database', db_items)
    
    click.echo("\n" + click.style("Config File: ", fg='bright_black') + config.config_path)
    click.echo()

if __name__ == '__main__':
    cli()
