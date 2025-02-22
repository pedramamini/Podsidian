import os
import json
import tempfile
import feedparser
import whisper
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from .models import Podcast, Episode
from .apple_podcasts import get_subscriptions

class PodcastProcessor:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.whisper_model = None
        self.embedding_model = None
        
        # Import here to avoid circular imports
        from .config import config
        self.config = config
        
    def _load_whisper(self):
        """Lazy load whisper model."""
        if not self.whisper_model:
            self.whisper_model = whisper.load_model("base")
    
    def _load_embedding_model(self):
        """Lazy load sentence transformer model."""
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _download_audio(self, url: str) -> str:
        """Download audio file to temporary location."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create temp file with .mp3 extension for whisper
        temp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp.write(chunk)
        temp.close()
        return temp.name
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file using whisper."""
        self._load_whisper()
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text."""
        self._load_embedding_model()
        return self.embedding_model.encode(text).tolist()
    
    def _get_summary(self, transcript: str) -> str:
        """Get AI-generated summary using OpenRouter."""
        if not self.config.openrouter_api_key:
            return "No OpenRouter API key configured. Summary not available."
            
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/pedramamini/podsidian",  # Required by OpenRouter
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": self.config.openrouter_model,
                "messages": [{
                    "role": "user",
                    "content": self.config.openrouter_prompt.format(transcript=transcript)
                }]
            }
        )
        
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _write_to_obsidian(self, episode: Episode):
        """Write episode transcript and summary to Obsidian vault if configured."""
        vault_path = self.config.vault_path
        if not vault_path:
            return
            
        podcast_dir = vault_path / "Podcasts" / episode.podcast.title
        podcast_dir.mkdir(parents=True, exist_ok=True)
        
        # Get AI summary if available
        summary = self._get_summary(episode.transcript) if episode.transcript else ""
        
        # Format note using template
        note_content = self.config.note_template.format(
            title=episode.title,
            podcast_title=episode.podcast.title,
            published_at=episode.published_at,
            audio_url=episode.audio_url,
            summary=summary,
            transcript=episode.transcript or "Transcript not available"
        )
        
        md_file = podcast_dir / f"{episode.title}.md"
        with md_file.open('w') as f:
            f.write(note_content)
    
    def ingest_subscriptions(self):
        """Ingest all Apple Podcast subscriptions and new episodes."""
        subscriptions = get_subscriptions()
        
        for sub in subscriptions:
            # Add or update podcast
            podcast = self.db.query(Podcast).filter_by(feed_url=sub['feed_url']).first()
            if not podcast:
                podcast = Podcast(
                    title=sub['title'],
                    author=sub['author'],
                    feed_url=sub['feed_url']
                )
                self.db.add(podcast)
                self.db.commit()
            
            # Parse feed
            feed = feedparser.parse(sub['feed_url'])
            
            for entry in feed.entries:
                # Check if episode exists
                guid = entry.get('id', entry.get('link'))
                if not guid:
                    continue
                    
                existing = self.db.query(Episode).filter_by(guid=guid).first()
                if existing:
                    continue
                
                # Find audio URL
                audio_url = None
                for link in entry.get('links', []):
                    if link.get('type', '').startswith('audio/'):
                        audio_url = link['href']
                        break
                
                if not audio_url:
                    continue
                
                # Create new episode
                episode = Episode(
                    podcast_id=podcast.id,
                    guid=guid,
                    title=entry.get('title', 'Untitled Episode'),
                    description=entry.get('description', ''),
                    published_at=datetime(*entry.published_parsed[:6]) if 'published_parsed' in entry else None,
                    audio_url=audio_url
                )
                
                try:
                    # Download and transcribe
                    temp_path = self._download_audio(audio_url)
                    episode.transcript = self._transcribe_audio(temp_path)
                    episode.vector_embedding = json.dumps(self._generate_embedding(episode.transcript))
                    
                    # Write to Obsidian if configured
                    self._write_to_obsidian(episode)
                    
                    # Cleanup
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Error processing episode {episode.title}: {e}")
                    continue
                
                self.db.add(episode)
                self.db.commit()
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search through podcast transcripts using vector similarity."""
        self._load_embedding_model()
        query_embedding = self._generate_embedding(query)
        
        # This is a simple implementation - in production you'd want to use a proper vector database
        results = []
        for episode in self.db.query(Episode).filter(Episode.vector_embedding.isnot(None)):
            episode_embedding = json.loads(episode.vector_embedding)
            # Compute cosine similarity
            similarity = sum(a * b for a, b in zip(query_embedding, episode_embedding))
            results.append({
                'episode': episode,
                'similarity': similarity
            })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return [
            {
                'podcast': r['episode'].podcast.title,
                'episode': r['episode'].title,
                'transcript': r['episode'].transcript,
                'published_at': r['episode'].published_at,
                'similarity': r['similarity']
            }
            for r in results[:limit]
        ]
