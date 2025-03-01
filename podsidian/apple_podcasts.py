import os
import sqlite3
import re
from typing import List, Dict, Optional, Tuple
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

def get_podcast_app_url(audio_url: str, guid: str = None, title: str = None) -> str:
    """Get the podcast:// URL for opening in Apple Podcasts app.
    
    This function tries to extract podcast ID and episode ID from the URL or find the
    corresponding podcast in the Apple Podcasts database using either the audio URL, GUID, or title.
    If found, it returns a podcast:// URL that will open the episode in the Apple Podcasts app.
    
    Args:
        audio_url: The audio URL from the episode
        guid: Optional episode GUID to use for lookup in Apple Podcasts database
        title: Optional episode title to use for lookup in Apple Podcasts database
        
    Returns:
        A podcast:// URL if found, or a generic podcast:// URL if not found
    """
    # Check if this is already an Apple Podcasts URL
    # Format: https://podcasts.apple.com/*/podcast/*/id<PODCAST_ID>?i=<EPISODE_ID>
    apple_pattern = r'podcasts\.apple\.com/[^/]+/podcast/[^/]+/id(\d+)\?i=(\d+)'
    match = re.search(apple_pattern, audio_url)
    if match:
        podcast_id, episode_id = match.groups()
        return f"https://podcasts.apple.com/podcast/id{podcast_id}?i={episode_id}"
    
    # Alternative format: https://podcasts.apple.com/*/podcast/*/id<PODCAST_ID>
    alt_pattern = r'podcasts\.apple\.com/[^/]+/podcast/[^/]+/id(\d+)'
    match = re.search(alt_pattern, audio_url)
    if match:
        podcast_id = match.group(1)
        return f"https://podcasts.apple.com/podcast/id{podcast_id}"
    
    # If not an Apple Podcasts URL, try to query the database
    try:
        db_path = find_apple_podcast_db()
        if not db_path:
            return "https://podcasts.apple.com"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # If we have a GUID, use it for the query (preferred method)
        if guid:
            cursor.execute("""
                SELECT p.ZSTORECOLLECTIONID, e.ZSTORETRACKID 
                FROM ZMTEPISODE e
                JOIN ZMTPODCAST p ON e.ZPODCAST = p.Z_PK
                WHERE e.ZGUID = ?
            """, (guid,))
            
            results = cursor.fetchall()
            if results and results[0][0] and results[0][1]:
                podcast_id, episode_id = results[0]
                return f"https://podcasts.apple.com/podcast/id{podcast_id}?i={episode_id}"
        
        # If no GUID or no results from GUID, try with the audio URL
        # Try matching on a significant portion of the URL path
        if audio_url:
            # Extract filename from URL for more specific matching
            filename_match = re.search(r'/([^/]+\.mp3)', audio_url)
            if filename_match:
                filename = filename_match.group(1)
                cursor.execute("""
                    SELECT p.ZSTORECOLLECTIONID, e.ZSTORETRACKID 
                    FROM ZMTEPISODE e
                    JOIN ZMTPODCAST p ON e.ZPODCAST = p.Z_PK
                    WHERE e.ZASSETURL LIKE ?
                """, (f'%{filename}%',))
                
                results = cursor.fetchall()
                if results and results[0][0] and results[0][1]:
                    podcast_id, episode_id = results[0]
                    return f"https://podcasts.apple.com/podcast/id{podcast_id}?i={episode_id}"
            
            # If still no match, try with domain
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', audio_url)
            if domain_match:
                domain = domain_match.group(1)
                cursor.execute("""
                    SELECT p.ZSTORECOLLECTIONID, e.ZSTORETRACKID 
                    FROM ZMTEPISODE e
                    JOIN ZMTPODCAST p ON e.ZPODCAST = p.Z_PK
                    WHERE e.ZASSETURL LIKE ?
                """, (f'%{domain}%',))
                
                results = cursor.fetchall()
                if results and results[0][0] and results[0][1]:
                    podcast_id, episode_id = results[0]
                    return f"https://podcasts.apple.com/podcast/id{podcast_id}?i={episode_id}"
        
        # If still no match and we have a title, try matching by title
        if title:
            # Clean the title and extract key words for matching
            # Remove common prefixes like "BONUS:" and clean up special characters
            clean_title = re.sub(r'^(BONUS|EPISODE|PREVIEW|TRAILER|TEASER):\s*', '', title, flags=re.IGNORECASE)
            clean_title = re.sub(r'[^\w\s]', '', clean_title).strip()
            
            # Extract significant words (longer than 3 chars, not common words)
            common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'what', 'your', 'are', 'how'}
            words = [word for word in clean_title.split() if len(word) > 3 and word.lower() not in common_words]
            
            # Use the first 3 significant words for matching (or fewer if not enough words)
            significant_words = words[:min(3, len(words))]
            
            if significant_words:
                # Build a query that checks for each significant word
                query = """
                    SELECT p.ZSTORECOLLECTIONID, e.ZSTORETRACKID, e.ZTITLE
                    FROM ZMTEPISODE e
                    JOIN ZMTPODCAST p ON e.ZPODCAST = p.Z_PK
                    WHERE 
                """
                
                conditions = []
                params = []
                
                for word in significant_words:
                    conditions.append("e.ZTITLE LIKE ?")
                    params.append(f"%{word}%")
                
                query += " AND ".join(conditions)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # If we get exactly one result, use it
                if len(results) == 1 and results[0][0] and results[0][1]:
                    podcast_id, episode_id = results[0][0], results[0][1]
                    return f"https://podcasts.apple.com/podcast/id{podcast_id}?i={episode_id}"
                
                # If we get multiple results, try to find the best match
                elif len(results) > 1:
                    best_match = None
                    best_match_score = 0
                    
                    for result in results:
                        if result[0] and result[1] and result[2]:  # Ensure we have podcast_id, episode_id, and title
                            # Simple scoring: count how many words from our significant words appear in the title
                            result_title = result[2].lower()
                            score = sum(1 for word in significant_words if word.lower() in result_title)
                            
                            # If this is a better match than what we've seen so far, update
                            if score > best_match_score:
                                best_match = result
                                best_match_score = score
                    
                    # If we found a good match (more than half the words match), use it
                    if best_match and best_match_score >= len(significant_words) / 2:
                        podcast_id, episode_id = best_match[0], best_match[1]
                        return f"https://podcasts.apple.com/podcast/id{podcast_id}?i={episode_id}"
        
        return "https://podcasts.apple.com"
        
    except sqlite3.Error as e:
        print(f"Error querying Apple Podcasts database: {e}")
        return "https://podcasts.apple.com"
    
    finally:
        if 'conn' in locals():
            conn.close()

