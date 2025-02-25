import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import json

def get_backup_dir() -> Path:
    """Get the backup directory path."""
    backup_dir = Path(os.path.expanduser("~/.local/share/podsidian/backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir

def create_backup(db_path: str) -> str:
    """Create a backup of the database with timestamp."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    backup_dir = get_backup_dir()
    backup_path = backup_dir / f"podsidian-{timestamp}.db"
    
    # If a backup from today exists, append an index
    index = 1
    while backup_path.exists():
        backup_path = backup_dir / f"podsidian-{timestamp}-{index}.db"
        index += 1
    
    shutil.copy2(db_path, backup_path)
    return str(backup_path)

def list_backups() -> List[Dict[str, str]]:
    """List all available backups with their details."""
    backup_dir = get_backup_dir()
    backups = []
    
    for backup in sorted(backup_dir.glob("podsidian-*.db")):
        stats = backup.stat()
        backups.append({
            'path': str(backup),
            'size': stats.st_size,
            'created': datetime.fromtimestamp(stats.st_mtime).isoformat()
        })
    
    return backups

def find_backup_by_date(date_str: str) -> str:
    """Find a backup by date string (YYYY-MM-DD).
    Returns the path to the most recent backup for that date.
    Raises FileNotFoundError if no backup exists for that date.
    """
    try:
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")
    
    backup_dir = get_backup_dir()
    pattern = f"podsidian-{date_str}*.db"
    backups = sorted(backup_dir.glob(pattern))
    
    if not backups:
        raise FileNotFoundError(f"No backup found for date {date_str}")
    
    # Return the last backup for that date (in case there are multiple)
    return str(backups[-1])

def restore_backup(date_str: str, db_path: str) -> None:
    """Restore a backup from a specific date, replacing the current database."""
    backup_path = find_backup_by_date(date_str)
    
    # Create a temporary backup of the current database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_backup = f"{db_path}.{timestamp}.bak"
    shutil.copy2(db_path, temp_backup)
    
    try:
        shutil.copy2(backup_path, db_path)
    except Exception as e:
        # If restore fails, try to restore from temporary backup
        shutil.copy2(temp_backup, db_path)
        os.unlink(temp_backup)
        raise Exception(f"Failed to restore backup: {str(e)}")
    
    os.unlink(temp_backup)
