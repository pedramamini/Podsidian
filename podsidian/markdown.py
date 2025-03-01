"""Markdown file management utilities."""
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import Episode
from .core import PodcastProcessor

def calculate_file_hash(filepath: Path) -> str:
    """Calculate first 8 characters of MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()[:8]

def extract_date_from_filename(filename: str) -> datetime:
    """Extract date from filename in YYYY-MM-DD format."""
    try:
        date_str = filename[:10]
        # Validate it's a proper date
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, IndexError):
        return None

def list_markdown_files(vault_path: Path, processor: PodcastProcessor) -> List[dict]:
    """List all markdown files in the vault with their MD5 hashes."""
    if not vault_path or not vault_path.exists():
        return []
    
    files = []
    for file in vault_path.glob("*.md"):
        # Calculate file hash
        file_hash = calculate_file_hash(file)
        
        # Extract date from filename
        filename_date = extract_date_from_filename(file.name)
        
        files.append({
            'filename': file.name,
            'file_hash': file_hash,
            'published_at': filename_date
        })
    
    # Sort by date (if available) then filename, descending
    return sorted(files, key=lambda x: (x['published_at'] or '0000-00-00', x['filename']), reverse=True)

def get_episode_from_markdown(filepath: Path, processor: PodcastProcessor) -> Optional[Episode]:
    """Get the episode corresponding to a markdown file by finding its audio URL."""
    try:
        print(f"\nReading markdown file: {filepath}")
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Debug: Print the metadata section for inspection
        import re
        metadata_match = re.search(r'# Metadata([\s\S]*?)(?=\n#|\Z)', content)
        if metadata_match:
            metadata_section = metadata_match.group(1)
            print(f"Found metadata section:\n{metadata_section}")
            
            # Look for URL line specifically
            url_line = re.search(r'\*\*URL\*\*:\s*(.*?)(?:\n|$)', metadata_section)
            if url_line:
                print(f"Found URL line: {url_line.group(0)}")
        
        # Try to match URL with a more flexible pattern
        match = re.search(r'\*\*URL\*\*:\s*(https?://[^\s\n]+)', content)
        if not match:
            print("No URL found with standard pattern, trying alternative pattern...")
            # Try an alternative pattern that's more lenient
            match = re.search(r'\*\*URL\*\*:\s*([^\s\n]+)', content)
            
        if not match:
            print("No Apple Podcasts URL found in markdown")
            # Debug: Print first few lines of content
            print("First 200 chars of content:")
            print(content[:200])
            return None
        
        audio_url = match.group(1)
        print(f"Found audio URL: {audio_url}")
        
        # Query database for episode with this audio URL
        from sqlalchemy import text
        # Debug: Print the actual SQL query
        query = processor.db.query(Episode).filter(Episode.audio_url == audio_url)
        print(f"SQL Query: {query}")
        
        # Print the SQL statement
        from sqlalchemy.dialects import sqlite
        sql_statement = str(query.statement.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
        print(f"SQL Statement: {sql_statement}")
        
        episode = query.first()
        if episode:
            print(f"Found matching episode: {episode.title}")
        else:
            print("No matching episode found in database")
            # Debug: Show a sample episode URL from DB
            sample = processor.db.query(Episode).first()
            if sample:
                print(f"Sample episode URL from DB: {sample.audio_url}")
                
            # Try a partial match
            print("Trying partial URL match...")
            # Extract domain part of the URL for partial matching
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', audio_url)
            if domain_match:
                domain = domain_match.group(1)
                print(f"Looking for URLs containing: {domain}")
                
                # Query for episodes with URLs containing this domain
                partial_query = processor.db.query(Episode).filter(Episode.audio_url.like(f"%{domain}%"))
                partial_sql = str(partial_query.statement.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))
                print(f"Partial match SQL: {partial_sql}")
                
                partial_matches = partial_query.all()
                if partial_matches:
                    print(f"Found {len(partial_matches)} potential matches:")
                    for i, ep in enumerate(partial_matches[:3]):  # Show first 3 matches
                        print(f"  {i+1}. {ep.title}: {ep.audio_url}")
                    
                    # Use the first match if available
                    episode = partial_matches[0]
                    print(f"Using first match: {episode.title}")
                    return episode
        return episode
    except Exception as e:
        print(f"Error reading markdown file: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def regenerate_markdown(episode_id: int, processor: PodcastProcessor) -> bool:
    """Regenerate a markdown file from the database using episode ID."""
    episode = processor.db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        return False
        
    try:
        processor._write_to_obsidian(episode)
        return True
    except Exception:
        return False
