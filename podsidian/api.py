from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime

from .models import Episode, Podcast
from .core import PodcastProcessor
from .apple_podcasts import get_subscriptions

# API Models
class SearchResult(BaseModel):
    podcast: str = Field(..., description="Name of the podcast")
    episode: str = Field(..., description="Episode title")
    published_at: datetime = Field(..., description="Publication date")
    similarity: float = Field(..., description="Relevance score (0-100)")
    snippet: str = Field(..., description="Matching transcript snippet")

class EpisodeInfo(BaseModel):
    id: int = Field(..., description="Episode ID")
    podcast: str = Field(..., description="Podcast name")
    title: str = Field(..., description="Episode title")
    description: str = Field(..., description="Episode description")
    published_at: datetime = Field(..., description="Publication date")
    has_transcript: bool = Field(..., description="Whether episode has been transcribed")

class SubscriptionInfo(BaseModel):
    title: str = Field(..., description="Podcast title")
    author: str = Field(..., description="Podcast author")
    feed_url: str = Field(..., description="RSS feed URL")
    muted: bool = Field(..., description="Whether podcast is muted")

# FastAPI app with documentation
app = FastAPI(
    title="Podsidian API",
    description="""
    Podsidian API for podcast transcription and semantic search.
    
    Features:
    - Semantic search across podcast transcripts
    - Keyword-based transcript search
    - Episode management
    - Podcast subscription management
    """,
    version="1.0.0",
    openapi_tags=[
        {"name": "discovery", "description": "Discover server capabilities"},
        {"name": "search", "description": "Search through podcast transcripts"},
        {"name": "episodes", "description": "Manage podcast episodes"},
        {"name": "subscriptions", "description": "Manage podcast subscriptions"}
    ]
)

def create_api(db_session: Session):
    processor = PodcastProcessor(db_session)
    
    @app.get("/", tags=["discovery"])
    def get_capabilities() -> Dict:
        """Get server capabilities and available endpoints.
        
        This is the root endpoint that MCP clients use to discover
        what functionality is available from the server.
        
        Returns:
            Dictionary of server capabilities and endpoints
        """
        return {
            "name": "Podsidian MCP",
            "version": "1.0.0",
            "capabilities": {
                "search": {
                    "semantic": {
                        "endpoint": "/api/v1/search/semantic",
                        "description": "Search transcripts using natural language",
                        "parameters": {
                            "query": "Search query string",
                            "limit": "Maximum results (default: 10)",
                            "relevance": "Minimum score 0-100 (default: 25)"
                        }
                    },
                    "keyword": {
                        "endpoint": "/api/v1/search/keyword",
                        "description": "Search transcripts for exact matches",
                        "parameters": {
                            "keyword": "Text to search for",
                            "limit": "Maximum results (default: 10)"
                        }
                    }
                },
                "episodes": {
                    "list": {
                        "endpoint": "/api/v1/episodes",
                        "description": "List processed episodes"
                    },
                    "get": {
                        "endpoint": "/api/v1/episodes/{episode_id}",
                        "description": "Get episode details and transcript"
                    }
                },
                "subscriptions": {
                    "list": {
                        "endpoint": "/api/v1/subscriptions",
                        "description": "List podcast subscriptions"
                    },
                    "mute": {
                        "endpoint": "/api/v1/subscriptions/{title}/mute",
                        "description": "Mute a podcast"
                    },
                    "unmute": {
                        "endpoint": "/api/v1/subscriptions/{title}/unmute",
                        "description": "Unmute a podcast"
                    }
                }
            }
        }
    
    @app.get("/api/v1/search/semantic", response_model=List[SearchResult], tags=["search"])
    def semantic_search(query: str, limit: int = 10, relevance: int = 25) -> List[Dict]:
        """Search through podcast transcripts using semantic similarity.
        
        This endpoint uses AI embeddings to find semantically similar content
        in podcast transcripts, even if the exact words don't match.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results (default: 10)
            relevance: Minimum relevance score 0-100 (default: 25)
            
        Returns:
            List of matching transcript segments with relevance scores
        """
        # Convert relevance to 0-1 scale
        relevance_float = relevance / 100.0
        results = processor.search(query, limit=limit, relevance_threshold=relevance_float)
        
        # Convert similarities to percentages
        for result in results:
            result['similarity'] = int(result['similarity'] * 100)
            
        return results
        
    @app.get("/api/v1/search/keyword", response_model=List[SearchResult], tags=["search"])
    def keyword_search(keyword: str, limit: int = 10) -> List[Dict]:
        """Search through podcast transcripts for exact keyword matches.
        
        This endpoint performs case-insensitive exact text matching within transcripts.
        
        Args:
            keyword: Text to search for
            limit: Maximum number of results (default: 10)
            
        Returns:
            List of transcript segments containing the keyword
        """
        results = processor.keyword_search(keyword, limit=limit)
        return results
    
    @app.get("/api/v1/episodes", response_model=List[EpisodeInfo], tags=["episodes"])
    def list_episodes(limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all processed episodes.
        
        Returns a paginated list of episodes that have been processed,
        ordered by publication date (newest first).
        
        Args:
            limit: Maximum number of episodes to return (default: 100)
            offset: Number of episodes to skip (default: 0)
            
        Returns:
            List of episode information
        """
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
    
    @app.get("/api/v1/episodes/{episode_id}", response_model=Dict, tags=["episodes"])
    def get_episode(episode_id: int) -> Dict:
        """Get specific episode details and transcript.
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            Episode details including transcript if available
            
        Raises:
            404: Episode not found
        """
        episode = db_session.query(Episode).filter_by(id=episode_id).first()
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
            
        return {
            'id': episode.id,
            'podcast': episode.podcast.title,
            'title': episode.title,
            'description': episode.description,
            'published_at': episode.published_at,
            'transcript': episode.transcript,
            'summary': episode.summary
        }
    
    @app.get("/api/v1/subscriptions", response_model=List[SubscriptionInfo], tags=["subscriptions"])
    def list_subscriptions() -> List[Dict]:
        """List all podcast subscriptions with their mute state.
        
        Returns information about all podcasts you're subscribed to in Apple Podcasts,
        along with their mute state in Podsidian.
        
        Returns:
            List of subscription information
        """
        # Get subscriptions from Apple Podcasts
        subs = get_subscriptions()
        if not subs:
            return []
        
        # Ensure all podcasts exist in database
        for sub in subs:
            podcast = db_session.query(Podcast).filter_by(feed_url=sub['feed_url']).first()
            if not podcast:
                podcast = Podcast(
                    title=sub['title'],
                    author=sub['author'],
                    feed_url=sub['feed_url'],
                    muted=False
                )
                db_session.add(podcast)
        db_session.commit()
        
        # Get mute states from database
        muted_feeds = {p.feed_url: p.muted for p in db_session.query(Podcast).all()}
        
        return [{
            'title': sub['title'],
            'author': sub['author'],
            'feed_url': sub['feed_url'],
            'muted': muted_feeds.get(sub['feed_url'], False)
        } for sub in subs]
    
    @app.post("/api/v1/subscriptions/{title}/mute", response_model=Dict, tags=["subscriptions"])
    def mute_subscription(title: str) -> Dict:
        """Mute a podcast subscription by title.
        
        Muted podcasts will not be processed during ingestion.
        
        Args:
            title: Title of the podcast to mute
            
        Returns:
            Updated subscription info
            
        Raises:
            404: Podcast not found
        """
        podcast = db_session.query(Podcast).filter_by(title=title).first()
        if not podcast:
            raise HTTPException(status_code=404, detail="Podcast not found")
            
        podcast.muted = True
        db_session.commit()
        
        return {
            'title': podcast.title,
            'author': podcast.author,
            'feed_url': podcast.feed_url,
            'muted': True
        }
    
    @app.post("/api/v1/subscriptions/{title}/unmute", response_model=Dict, tags=["subscriptions"])
    def unmute_subscription(title: str) -> Dict:
        """Unmute a podcast subscription by title.
        
        Args:
            title: Title of the podcast to unmute
            
        Returns:
            Updated subscription info
            
        Raises:
            404: Podcast not found
        """
        podcast = db_session.query(Podcast).filter_by(title=title).first()
        if not podcast:
            raise HTTPException(status_code=404, detail="Podcast not found")
            
        podcast.muted = False
        db_session.commit()
        
        return {
            'title': podcast.title,
            'author': podcast.author,
            'feed_url': podcast.feed_url,
            'muted': False
        }
    
    return app
