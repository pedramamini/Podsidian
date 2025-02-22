import os
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path

def find_apple_podcast_db() -> Optional[str]:
    """Find the Apple Podcasts SQLite database in the Group Containers directory."""
    group_containers = Path.home() / "Library" / "Group Containers"
    
    if not group_containers.exists():
        return None
    
    for path in group_containers.rglob("MTLibrary.sqlite"):
        if path.is_file():
            return str(path)
    
    return None

def get_subscriptions() -> List[Dict[str, str]]:
    """Get all podcast subscriptions from Apple Podcasts."""
    db_path = find_apple_podcast_db()
    if not db_path:
        raise FileNotFoundError("Apple Podcasts database not found")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get podcast subscriptions
        cursor.execute("""
            SELECT ZTITLE, ZAUTHOR, ZFEEDURL 
            FROM ZMTPODCAST
            WHERE ZFEEDURL IS NOT NULL
        """)
        
        subscriptions = []
        for row in cursor.fetchall():
            subscriptions.append({
                'title': row[0],
                'author': row[1],
                'feed_url': row[2]
            })
        
        return subscriptions
        
    except sqlite3.Error as e:
        raise Exception(f"Error reading Apple Podcasts database: {e}")
    
    finally:
        if 'conn' in locals():
            conn.close()
