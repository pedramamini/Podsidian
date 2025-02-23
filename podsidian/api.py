from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime

from .models import Episode
from .core import PodcastProcessor

app = FastAPI(title="Podsidian MCP API")

def create_api(db_session: Session):
    processor = PodcastProcessor(db_session)
    
    @app.get("/api/v1/search/semantic")
    def semantic_search(query: str, limit: int = 10, relevance: int = 25) -> List[Dict]:
        """Search through podcast transcripts using semantic similarity.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            relevance: Minimum relevance score (0-100) for results
        """
        # Convert relevance to 0-1 scale
        relevance_float = relevance / 100.0
        results = processor.search(query, limit=limit, relevance_threshold=relevance_float)
        
        # Convert similarities to percentages
        for result in results:
            result['similarity'] = int(result['similarity'] * 100)
            
        return results
        
    @app.get("/api/v1/search/keyword")
    def keyword_search(keyword: str, limit: int = 10) -> List[Dict]:
        """Search through podcast transcripts for exact keyword matches.
        
        Args:
            keyword: Exact text to search for (case-insensitive)
            limit: Maximum number of results to return
        """
        return processor.keyword_search(keyword, limit=limit)
    
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
