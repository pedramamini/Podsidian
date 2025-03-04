import os
import click
from tqdm import tqdm
import shutil
import uvicorn
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from .models import init_db, Podcast
from .config import config
from .backup import create_backup, list_backups, find_backup_by_date, restore_backup
from .api import create_api

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

@cli.command(name='show-config')
def show_config():
    """Show current configuration and status."""
    # Check if config file exists
    if not os.path.exists(config.config_path):
        click.echo(click.style("\nWarning: No configuration file found!", fg='red', bold=True))
        click.echo(click.style("Run 'podsidian init' to create a default configuration file,", fg='yellow'))
        click.echo(click.style("then adjust the settings in " + config.config_path + " as needed.\n", fg='yellow'))
        return
        
    session = get_db_session()
    
    def print_section(title, items, indent=0):
        click.echo("\n" + " " * indent + click.style(f"[{title}]", fg='green', bold=True))
        for key, value in items:
            # Skip the template and prompt as they're too long
            if key in ['template', 'prompt', 'value_prompt']:
                value = '<configured>' if value else '<not configured>'
            # Mask API key
            elif key == 'api_key':
                value = '***' + value[-4:] if value else '<not set>'
            # Format lists nicely
            elif isinstance(value, list):
                value = '\n' + '\n'.join([' ' * (indent + 4) + '• ' + str(item) for item in value]) if value else '<none>'
            click.echo(" " * (indent + 2) + click.style(f"{key}: ", fg='bright_black') + str(value))
    
    # Get Annoy index info
    annoy_path = config.annoy_index_path
    annoy_exists = os.path.exists(annoy_path)
    annoy_size = os.path.getsize(annoy_path) if annoy_exists else 0
    
    # Get episode stats
    from .models import Episode, Podcast
    total_episodes = session.query(Episode).count()
    episodes_with_embeddings = session.query(Episode).filter(Episode.vector_embedding.isnot(None)).count()
    
    click.echo("\nAnnoy Vector Index:")
    click.echo(f"  Path: {click.style(annoy_path, fg='blue')}")
    click.echo(f"  Exists: {click.style('Yes', fg='green') if annoy_exists else click.style('No', fg='red')}")
    if annoy_exists:
        click.echo(f"  Size: {click.style(str(round(annoy_size / 1024 / 1024, 2)), fg='blue')} MB")
    
    # Get podcast stats
    total_podcasts = session.query(Podcast).count()
    active_podcasts = session.query(Podcast).filter(Podcast.muted == False).count()
    muted_podcasts = session.query(Podcast).filter(Podcast.muted == True).count()

    click.echo("\nPodcast Statistics:")
    click.echo(f"  Total Podcasts: {click.style(str(total_podcasts), fg='blue')}")
    click.echo(f"  Active Podcasts: {click.style(str(active_podcasts), fg='blue')}")
    click.echo(f"  Muted Podcasts: {click.style(str(muted_podcasts), fg='blue')}")

    click.echo("\nEpisode Statistics:")
    click.echo(f"  Total Episodes: {click.style(str(total_episodes), fg='blue')}")
    click.echo(f"  Episodes with Embeddings: {click.style(str(episodes_with_embeddings), fg='blue')}")

    # Obsidian Settings
    obsidian_items = [
        ('vault_path', config.vault_path),
        ('template', config.note_template)
    ]
    print_section('Obsidian', obsidian_items)
    
    # Whisper Settings
    whisper_items = [
        ('model', config.whisper_model),
        ('language', config.whisper_language or '<auto>'),
        ('cpu_only', config.whisper_cpu_only),
        ('threads', config.whisper_threads)
    ]
    print_section('Whisper', whisper_items)
    
    # OpenRouter Settings
    openrouter_items = [
        ('api_key', config.openrouter_api_key),
        ('model', config.openrouter_model),
        ('processing_model', config.openrouter_processing_model),
        ('topic_sample_size', config.topic_sample_size),
        ('cost_tracking_enabled', config.cost_tracking_enabled),
        ('prompt', config.openrouter_prompt)
    ]
    print_section('OpenRouter', openrouter_items)
    
    # Value Analysis Settings
    value_items = [
        ('enabled', config.value_prompt_enabled),
        ('prompt', config.value_prompt)
    ]
    print_section('Value Analysis', value_items)
    
    # Database Settings
    db_items = [
        ('path', DEFAULT_DB_PATH),
    ]
    print_section('Database', db_items)
    
    click.echo("\n" + click.style("Config File: ", fg='bright_black') + config.config_path)
    click.echo()
    
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
@click.option('--sort', type=click.Choice(['alpha', 'episodes']), default='alpha', help='Sort by name (alpha) or episode count (episodes)')
def list_subscriptions(sort):
    """List all Apple Podcasts subscriptions."""
    from .apple_podcasts import get_subscriptions
    from .models import Podcast, Episode
    from sqlalchemy import func
    session = get_db_session()
    
    # Get subscriptions from Apple Podcasts
    subs = get_subscriptions()
    if not subs:
        click.echo("No Apple Podcasts subscriptions found.")
        return
    
    # Get episode counts for each podcast
    episode_counts = dict(session.query(
        Podcast.feed_url,
        func.count(Episode.id).label('count')
    ).join(Episode, isouter=True).group_by(Podcast.feed_url).all())
    
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
            episode_counts[sub['feed_url']] = 0
    session.commit()
    
    # Get mute states from database
    muted_feeds = {p.feed_url: p.muted for p in session.query(Podcast).all()}
    
    # Split into muted and unmuted lists
    muted_subs = []
    unmuted_subs = []
    for sub in subs:
        if muted_feeds.get(sub['feed_url'], False):
            muted_subs.append(sub)
        else:
            unmuted_subs.append(sub)
            
    # Sort the lists based on the sort option
    if sort == 'episodes':
        muted_subs.sort(key=lambda x: (-episode_counts.get(x['feed_url'], 0), x['title'].lower()))
        unmuted_subs.sort(key=lambda x: (-episode_counts.get(x['feed_url'], 0), x['title'].lower()))
    else:  # alpha
        muted_subs.sort(key=lambda x: x['title'].lower())
        unmuted_subs.sort(key=lambda x: x['title'].lower())
    
    # Show active subscriptions
    click.echo("\nActive Subscriptions:")
    click.echo("-" * 30)
    if unmuted_subs:
        for sub in unmuted_subs:
            episode_count = episode_counts.get(sub['feed_url'], 0)
            episodes_text = f" ({episode_count} episode{'s' if episode_count != 1 else ''})"
            click.echo(f"• {sub['title']}{episodes_text}")
    else:
        click.echo("No active subscriptions")
    
    # Show muted subscriptions
    click.echo("\nMuted Subscriptions:")
    click.echo("-" * 30)
    if muted_subs:
        for sub in muted_subs:
            episode_count = episode_counts.get(sub['feed_url'], 0)
            episodes_text = f" ({episode_count} episode{'s' if episode_count != 1 else ''})"
            click.echo(f"• {sub['title']}{episodes_text}")
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
    from .core import PodcastProcessor
    
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    if lookback <= 0:
        click.echo("Error: Lookback days must be greater than 0")
        return
    
    if lookback > 7:
        click.confirm(f"Warning: Looking back {lookback} days may take a while. Continue?", abort=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    click.echo(f"[{timestamp}] Ingesting episodes from the last {lookback} days...")
    
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    click.echo(f"\n\n[{timestamp}] Ingestion complete!")
    
    # Display cost summary if enabled
    if cost_tracking_enabled:
        click.echo("\n" + format_cost_summary())

@cli.command()
@click.argument('query')
@click.option('--relevance', type=int, default=30, help='Minimum relevance score (0-100) for results')
@click.option('--refresh', is_flag=True, help='Force refresh of the search index before searching')
def search(query, relevance, refresh):
    """Search through podcast content using natural language.
    
    Uses AI to find relevant content even when exact words don't match.
    Results are ranked by relevance to your query.
    
    Examples:
        podsidian search "electric cars impact on climate"
        podsidian search "meditation techniques" --relevance 50
    """
    from .core import PodcastProcessor
    from .models import Podcast, Episode
    from .cost_tracker import init_cost_tracker, format_cost_summary
    
    # Initialize cost tracker if enabled
    cost_tracking_enabled = config.cost_tracking_enabled
    if cost_tracking_enabled:
        init_cost_tracker()
    
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    # Get statistics about searchable content
    total_podcasts = session.query(Podcast).filter_by(muted=False).count()
    total_episodes = session.query(Episode).filter(Episode.vector_embedding.isnot(None)).count()
    click.echo(f"Searching through {click.style(str(total_podcasts), bold=True)} podcasts and {click.style(str(total_episodes), bold=True)} episodes")
    click.echo(f"Minimum relevance threshold: {click.style(f'{relevance}%', fg='yellow')}")
    click.echo("─" * 50)
    
    # Force index refresh if requested
    if refresh:
        click.echo("Refreshing search index...")
        processor._init_annoy_index(force_rebuild=True)
        click.echo("Index refresh complete.")
    
    # Convert relevance to 0-1 scale
    relevance_float = relevance / 100.0
    results = processor.search(query, relevance_threshold=relevance_float)
    
    if not results:
        click.echo("No results found matching your query with the current relevance threshold.")
        click.echo(f"Try lowering the threshold (current: {relevance}%)")
        return
    
    # Group results by podcast
    podcasts = {}
    for result in results:
        if result['podcast'] not in podcasts:
            podcasts[result['podcast']] = []
        podcasts[result['podcast']].append(result)
    
    # Display results grouped by podcast
    for podcast, episodes in podcasts.items():
        click.echo(f"\n{click.style(podcast, fg='blue', bold=True)}:")
        click.echo("-" * len(podcast))
        
        for result in episodes:
            # Show episode title and metadata
            date_str = result['published_at'].strftime("%Y-%m-%d") if result['published_at'] else "No date"
            click.echo(f"\n{click.style(result['episode'], bold=True)} ({date_str})")
            click.echo(f"Relevance: {click.style(f'{result['similarity']}%', fg='green')}")
            
            # Show relevant excerpt
            if result.get('excerpt'):
                click.echo("\nRelevant excerpt:")
                click.echo(f"{click.style('│ ', fg='bright_black')}{result['excerpt']}")
            
    click.echo("\nTip: Use --relevance to adjust the minimum relevance score (0-100)")
    
    # Display cost summary if enabled
    if 'cost_tracking_enabled' in locals() and cost_tracking_enabled:
        click.echo("\n" + format_cost_summary())

@cli.group()
def markdown():
    """Manage markdown exports."""
    pass

@markdown.command(name='list')
def list_markdown():
    """List all markdown files in the vault."""
    from .markdown import list_markdown_files
    from .core import PodcastProcessor
    
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    if not processor.config.vault_path:
        click.echo("Error: No vault path configured")
        return
        
    files = list_markdown_files(processor.config.vault_path, processor)
    if not files:
        click.echo("No markdown files found in vault")
        return
        
    click.echo(f"\nFound {len(files)} markdown files in vault:")
    for file in files:
        hash_str = file['file_hash']
        published_at = file['published_at']
        date_str = published_at.strftime('%Y-%m-%d') if published_at else 'No Date'
        # Check if filename starts with YYYY-MM-DD
        has_date_prefix = file['filename'].startswith(date_str)
        
        # Only show date column if it's not already in filename
        if has_date_prefix:
            click.echo(
                f"  {click.style(hash_str, fg='green')} "
                f"{file['filename']}"
            )
        else:
            click.echo(
                f"  {click.style(hash_str, fg='green')} "
                f"[{click.style(date_str, fg='yellow')}] "
                f"{file['filename']}"
            )

@markdown.command(name='regenerate')
@click.argument('file_hash')
def regenerate_markdown(file_hash):
    """Regenerate a markdown file.
    
    If FILE_HASH is '*', regenerates all markdown files.
    Otherwise regenerates the specified file by its hash.
    """
    from .markdown import list_markdown_files, get_episode_from_markdown
    from .core import PodcastProcessor
    from .cost_tracker import init_cost_tracker, format_cost_summary
    
    # Initialize cost tracker if enabled
    cost_tracking_enabled = config.cost_tracking_enabled
    if cost_tracking_enabled:
        init_cost_tracker()
        
    session = get_db_session()
    processor = PodcastProcessor(session)
    
    if not processor.config.vault_path:
        click.echo("Error: No vault path configured")
        return
    
    files = list_markdown_files(processor.config.vault_path, processor)
    if not files:
        click.echo("No markdown files found in vault")
        return
    
    # Filter files based on hash
    if file_hash == '*':
        files_to_process = files
    else:
        files_to_process = [f for f in files if f['file_hash'] == file_hash]
        if not files_to_process:
            click.echo(f"No markdown file found with hash: {file_hash}")
            return
    
    success = 0
    if file_hash == '*':
        click.echo(f"\nRegenerating {len(files_to_process)} markdown files...")
        with tqdm(total=len(files_to_process)) as pbar:
            for file in files_to_process:
                # Get episode
                filepath = processor.config.vault_path / file['filename']
                episode = get_episode_from_markdown(filepath, processor)
                if not episode:
                    click.echo(f"\nWarning: Could not find episode for {file['filename']}")
                    continue
                
                # Generate markdown using the processor's method
                try:
                    click.echo(f"Regenerating: {file['filename']}")
                    processor._write_to_obsidian(episode)
                    success += 1
                except Exception as e:
                    click.echo(f"\nError regenerating {file['filename']}: {str(e)}")
                pbar.update(1)
        click.echo(f"\nSuccessfully regenerated {success} of {len(files_to_process)} files")
        
        # Display cost summary if enabled
        if cost_tracking_enabled:
            click.echo("\n" + format_cost_summary())
    else:
        file = files_to_process[0]  # We know there's exactly one file
        filepath = processor.config.vault_path / file['filename']
        episode = get_episode_from_markdown(filepath, processor)
        if not episode:
            click.echo(f"Warning: Could not find episode for {file['filename']}")
            return
        
        # Generate markdown using the processor's method
        try:
            click.echo(f"Regenerating: {file['filename']}")
            processor._write_to_obsidian(episode)
            click.echo(f"Successfully regenerated {file['filename']}")
            
            # Display cost summary if enabled
            if cost_tracking_enabled:
                click.echo("\n" + format_cost_summary())
        except Exception as e:
            click.echo(f"Error regenerating {file['filename']}: {str(e)}")

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
@click.option('--stdio', is_flag=True, help='Run in STDIO mode for direct integration with AI agents')
def mcp(port, stdio):
    """Start the MCP service for AI agent integration.
    
    Can run as a HTTP server (default) or in STDIO mode for direct integration.
    """
    session = get_db_session()
    app = create_api(session)
    
    if stdio:
        import sys
        import asyncio
        from .stdio_server import run_stdio_server
        
        # Run in STDIO mode
        print("Starting Podsidian MCP Server in STDIO mode...", file=sys.stderr)
        asyncio.run(run_stdio_server(app))
    else:
        # Run as HTTP server
        uvicorn.run(app, host="0.0.0.0", port=port)



@cli.group()
def backup():
    """Manage database backups."""
    pass

@backup.command(name='create')
def backup_create():
    """Create a new backup of the database."""
    try:
        backup_path = create_backup(DEFAULT_DB_PATH)
        click.echo(f"Created backup at: {backup_path}")
    except Exception as e:
        click.echo(f"Error creating backup: {str(e)}", err=True)

@backup.command(name='list')
def backup_list():
    """List all available backups."""
    backups = list_backups()
    
    if not backups:
        click.echo("No backups found.")
        return
    
    click.echo("Available backups:")
    click.echo("-" * 80)
    
    for backup in backups:
        created = datetime.fromisoformat(backup['created']).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = backup['size'] / (1024 * 1024)
        click.echo(f"• {created} ({size_mb:.1f} MB)")
        click.echo(f"  Path: {backup['path']}")
        
        # Show subscription and episode counts if available
        if backup['subscriptions'] is not None and backup['episodes'] is not None:
            click.echo(f"  Contents: {backup['subscriptions']} subscriptions, {backup['episodes']} episodes")
        
        # Add a blank line between entries
        click.echo("")

@backup.command(name='restore')
@click.argument('date')
def backup_restore(date):
    """Restore database from a backup.
    
    DATE is the backup date in YYYY-MM-DD format.
    Use 'backup list' to see available backups.
    """
    try:
        # Find backup for the given date
        backup_path = find_backup_by_date(date)
        
        # Get info about current and backup databases
        current_size = os.path.getsize(DEFAULT_DB_PATH)
        backup_size = os.path.getsize(backup_path)
        backup_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
        current_time = datetime.fromtimestamp(os.path.getmtime(DEFAULT_DB_PATH))
        
        # Show differences
        click.echo("Restore details:")
        click.echo(f"Selected backup: {backup_path}")
        click.echo(f"Current database size: {current_size / (1024*1024):.1f} MB")
        click.echo(f"Backup database size: {backup_size / (1024*1024):.1f} MB")
        click.echo(f"Current database last modified: {current_time}")
        click.echo(f"Backup database created: {backup_time}")
        click.echo(f"Time difference: {current_time - backup_time}")
        
        if click.confirm("Are you sure you want to restore this backup? This will overwrite your current database.", abort=True):
            restore_backup(date, DEFAULT_DB_PATH)
            click.echo("Backup restored successfully.")
    except Exception as e:
        click.echo(f"Error restoring backup: {str(e)}", err=True)

if __name__ == '__main__':
    cli()
