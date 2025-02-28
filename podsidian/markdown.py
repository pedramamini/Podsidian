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

def get_episode_from_filename(filename: str, processor: PodcastProcessor) -> Optional[Episode]:
    """Get the episode corresponding to a markdown filename."""
    # Extract date and title from filename
    try:
        date_str = filename[:10]  # YYYY-MM-DD
        title = filename[11:-3]  # Remove date and .md extension
        
        # Find episode by date and similar title
        episodes = processor.db.query(Episode).filter(
            Episode.published_at.cast(str).like(f"{date_str}%")
        ).all()
        
        # Find best match by title similarity
        best_match = None
        best_score = 0
        for episode in episodes:
            from difflib import SequenceMatcher
            score = SequenceMatcher(None, 
                processor._make_safe_filename(episode.title), 
                title
            ).ratio()
            if score > best_score:
                best_score = score
                best_match = episode
                
        return best_match if best_score > 0.8 else None
    except Exception:
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
