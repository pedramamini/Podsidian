from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime

from .models import Episode
from .core import PodcastProcessor

app = FastAPI(title="Podsidian MCP API")

def create_api(db_session: Session):
    processor = PodcastProcessor(db_session)
    
    @app.get("/api/v1/search")
    def search(query: str, limit: int = 10) -> List[Dict]:
        """Search through podcast transcripts."""
        return processor.search(query, limit)
    
    @app.get("/api/v1/episodes")
    def list_episodes(limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all processed episodes."""
        episodes = db_session.query(Episode).order_by(
            Episode.published_at.desc()
        ).offset(offset).limit(limit).all()
        
        return [{
            'id': episode.id,
            'podcast': episode.podcast.title,
            'title': episode.title,
            'description': episode.description,
            'published_at': episode.published_at,
            'has_transcript': episode.transcript is not None
        } for episode in episodes]
    
    @app.get("/api/v1/episodes/{episode_id}")
    def get_episode(episode_id: int) -> Dict:
        """Get specific episode details and transcript."""
        episode = db_session.query(Episode).filter_by(id=episode_id).first()
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
            
        return {
            'id': episode.id,
            'podcast': episode.podcast.title,
            'title': episode.title,
            'description': episode.description,
            'published_at': episode.published_at,
            'transcript': episode.transcript
        }
    
    return app
