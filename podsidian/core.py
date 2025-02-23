import os
import re
import json
import time
import tempfile
import feedparser
import whisper
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from sqlalchemy.orm import Session

# Set tokenizers parallelism before importing transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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
        """Lazy load whisper model with configured options."""
        if not self.whisper_model:
            import torch
            import warnings
            import whisper

            if hasattr(self, '_progress_callback') and self._debug:
                self._progress_callback({
                    'stage': 'debug',
                    'message': f'Loading Whisper model: {self.config.whisper_model}'
                })

            # Suppress FP16 warning on CPU
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

            # Set device based on config and availability
            if self.config.whisper_cpu_only:
                device = "cpu"
                if self.config.whisper_threads > 0:
                    torch.set_num_threads(self.config.whisper_threads)
                    if hasattr(self, '_progress_callback') and self._debug:
                        self._progress_callback({
                            'stage': 'debug',
                            'message': f'Set torch threads to {self.config.whisper_threads}'
                        })
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if hasattr(self, '_progress_callback') and self._debug:
                    self._progress_callback({
                        'stage': 'debug',
                        'message': f'Using device: {device}'
                    })

            self.whisper_model = whisper.load_model(
                self.config.whisper_model,
                device=device
            )
            
            if hasattr(self, '_progress_callback') and self._debug:
                self._progress_callback({
                    'stage': 'debug',
                    'message': 'Whisper model loaded successfully'
                })

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

    def _detect_topic(self, title: str, transcript_sample: str) -> str:
        """Detect the main topic/domain of the podcast using OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/pedramamini/podsidian",
        }

        # Prompt to detect the professional domain
        prompt = f"""Given the following podcast title and transcript sample, determine the specific professional or technical domain this content belongs to. Focus on identifying specialized fields that might have unique terminology (e.g., Brazilian Jiu-Jitsu, Quantum Physics, Constitutional Law, etc.).

Title: {title}

Transcript Sample:
{transcript_sample}

Respond with just the domain name, nothing else."""

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": self.config.openrouter_processing_model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _correct_transcript(self, transcript: str, domain: str) -> str:
        """Correct transcript using domain-specific knowledge."""
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/pedramamini/podsidian",
        }

        # Prompt for domain-specific correction
        prompt = f"""You are a professional transcriptionist with extensive expertise in {domain}. Your task is to correct any technical terms, jargon, or domain-specific language in this transcript that might have been misinterpreted during speech-to-text conversion.

Focus on:
1. Technical terminology specific to {domain}
2. Names of key figures or concepts in the field
3. Specialized vocabulary and acronyms
4. Common terms that might have been confused with domain-specific ones

Transcript:
{transcript}

Provide the corrected transcript, maintaining all original formatting and structure."""

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": self.config.openrouter_processing_model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _transcribe_audio(self, audio_path: str, title: str, progress_callback=None, debug: bool = False) -> str:
        """Transcribe audio file using whisper with configured options and process with domain expertise."""
        # Store progress callback and debug flag for use in _load_whisper
        self._progress_callback = progress_callback
        self._debug = debug
        
        self._load_whisper()

        # Prepare transcription options
        options = {}
        if self.config.whisper_language:
            options['language'] = self.config.whisper_language
            
        # Set number of threads for transcription
        if hasattr(self.config, 'num_threads'):
            options['num_threads'] = self.config.num_threads
            
        if debug and progress_callback:
            progress_callback({
                'stage': 'debug',
                'message': f'Transcription options: {options}'
            })

        # Create a wrapper for tqdm that updates our progress
        if progress_callback:
            from tqdm import tqdm
            original_tqdm = tqdm.__init__
            last_update_time = [time.time()]

            def custom_tqdm_init(self, *args, **kwargs):
                original_tqdm(self, *args, **kwargs)
                self.progress_callback = progress_callback
                self._original_update = self.update
                self._start_time = time.time()

                def custom_update(n=1):
                    current_time = time.time()
                    if current_time - last_update_time[0] >= 0.5:
                        self._original_update(n)
                        progress_info = {
                            'stage': 'transcribing_progress',
                            'progress': self.n / self.total if self.total else 0
                        }
                        
                        if debug:
                            elapsed = current_time - self._start_time
                            rate = self.n / elapsed if elapsed > 0 else 0
                            eta = (self.total - self.n) / rate if rate > 0 else 0
                            progress_info['debug'] = (
                                f'frames={self.n}/{self.total} | '
                                f'rate={rate:.1f} frames/s | '
                                f'elapsed={elapsed:.1f}s | '
                                f'eta={eta:.1f}s'
                            )
                            
                        self.progress_callback(progress_info)
                        last_update_time[0] = current_time

                self.update = custom_update

            tqdm.__init__ = custom_tqdm_init

        try:
            if debug and progress_callback:
                progress_callback({
                    'stage': 'debug',
                    'message': f'Starting transcription of {os.path.basename(audio_path)}'
                })
                
            # Get initial transcript from Whisper
            result = self.whisper_model.transcribe(audio_path, **options)
            
            if debug and progress_callback:
                progress_callback({
                    'stage': 'debug',
                    'message': f'Transcription completed, text length: {len(result["text"])} chars'
                })
        except Exception as e:
            if debug and progress_callback:
                import traceback
                progress_callback({
                    'stage': 'debug',
                    'message': f'Transcription error: {str(e)}\n{traceback.format_exc()}'
                })
            raise

        # Restore original tqdm if we modified it
        if progress_callback:
            tqdm.__init__ = original_tqdm

        initial_transcript = result["text"]

        # Take a sample for topic detection
        sample_size = min(len(initial_transcript), self.config.topic_sample_size)
        transcript_sample = initial_transcript[:sample_size]

        # Detect the domain/topic
        domain = self._detect_topic(title, transcript_sample)

        # Correct the transcript using domain expertise
        corrected_transcript = self._correct_transcript(initial_transcript, domain)

        return corrected_transcript

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

    def _make_safe_filename(self, s: str) -> str:
        """Convert string to a safe filename."""
        # Remove content within brackets and parentheses
        s = re.sub(r'\s*\[[^\]]*\]\s*', '', s)  # Remove [content]
        s = re.sub(r'\s*\([^\)]*\)\s*', '', s)  # Remove (content)
        
        # Remove or replace unsafe characters
        s = re.sub(r'[<>:"/\\|?*]', '', s)
        # Replace multiple spaces with single space
        s = re.sub(r'\s+', ' ', s)
        # Remove leading/trailing spaces and dots
        s = s.strip('. ')
        # Ensure filename isn't too long (max 255 chars including extension)
        return s[:250]

    def _write_to_obsidian(self, episode: Episode):
        """Write episode transcript and summary to Obsidian vault if configured."""
        vault_path = self.config.vault_path
        if not vault_path:
            return

        if not episode.podcast:
            raise Exception("Episode has no associated podcast")

        if not episode.podcast.title:
            raise Exception("Podcast has no title")

        # Create filename: YYYY-MM-DD title
        date_str = episode.published_at.strftime("%Y-%m-%d") if episode.published_at else "no-date"
        safe_title = self._make_safe_filename(episode.title)
        filename = f"{date_str} {safe_title}.md"

        # Get AI summary if available
        summary = self._get_summary(episode.transcript) if episode.transcript else ""

        # Format note using template
        note_content = self.config.note_template.format(
            title=episode.title,
            podcast_title=episode.podcast.title,
            published_at=episode.published_at.strftime('%Y-%m-%d') if episode.published_at else 'Unknown',
            audio_url=episode.audio_url,
            summary=summary,
            transcript=episode.transcript or "Transcript not available"
        )

        md_file = vault_path / filename
        with md_file.open('w') as f:
            f.write(note_content)

    def ingest_subscriptions(self, lookback_days: int = 7, progress_callback=None, debug: bool = False):
        """Ingest all Apple Podcast subscriptions and new episodes.

        Args:
            lookback_days: Only process episodes published within this many days (default: 7)
            progress_callback: Optional callback for progress updates, receives dict with:
                - stage: str ('podcast' or 'episode')
                - podcast: dict with podcast info
                - current: int (current count)
                - total: int (total count)
                - episode: dict with episode info (if stage == 'episode')
        """
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        subscriptions = get_subscriptions()

        # Initial stats
        if progress_callback:
            progress_callback({
                'stage': 'init',
                'total_podcasts': len(subscriptions)
            })

        # Filter out ignored podcasts
        subscriptions = [s for s in subscriptions if s['title'] not in self.config.ignore_podcasts]

        for podcast_idx, sub in enumerate(subscriptions, 1):
            # Progress update for podcast
            if progress_callback:
                progress_callback({
                    'stage': 'podcast',
                    'podcast': sub,
                    'current': podcast_idx,
                    'total': len(subscriptions)
                })

            # Add or update podcast
            try:
                with self.db.no_autoflush:
                    podcast = self.db.query(Podcast).filter_by(feed_url=sub['feed_url']).first()
                    if not podcast:
                        podcast = Podcast(
                            title=sub['title'],
                            author=sub['author'],
                            feed_url=sub['feed_url']
                        )
                        self.db.add(podcast)
                        self.db.commit()

                    # Verify podcast was created successfully
                    podcast = self.db.query(Podcast).filter_by(feed_url=sub['feed_url']).first()
                    if not podcast or not podcast.title:
                        raise Exception("Failed to create/retrieve podcast")
            except Exception as e:
                if progress_callback:
                    progress_callback({
                        'stage': 'error',
                        'error': f"Failed to process podcast {sub['title']}: {str(e)}"
                    })
                continue

            # Parse feed
            feed = feedparser.parse(sub['feed_url'])
            recent_entries = []

            # First pass: collect recent entries
            for entry in feed.entries:
                # Parse published date
                published_at = datetime(*entry.published_parsed[:6]) if 'published_parsed' in entry else None
                if published_at and published_at >= cutoff_date:
                    recent_entries.append((entry, published_at))

            # Progress update for episodes
            if progress_callback and recent_entries:
                progress_callback({
                    'stage': 'episodes_found',
                    'podcast': sub,
                    'total': len(recent_entries)
                })

            # Process recent entries
            for entry_idx, (entry, published_at) in enumerate(recent_entries, 1):
                # Check if episode exists
                guid = entry.get('id', entry.get('link'))
                if not guid:
                    continue

                # Parse published date
                published_at = datetime(*entry.published_parsed[:6]) if 'published_parsed' in entry else None

                # Skip if too old or already exists
                if not published_at or published_at < cutoff_date:
                    continue

                # Check for existing episode with autoflush disabled
                with self.db.no_autoflush:
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

                # Extract episode metadata safely
                episode_title = entry.get('title')
                if not episode_title:
                    if progress_callback:
                        progress_callback({
                            'stage': 'error',
                            'podcast': sub,
                            'episode': {'title': 'Unknown'},
                            'error': 'Episode missing title'
                        })
                    continue

                # Create new episode
                try:
                    episode = Episode(
                        podcast=podcast,  # Set the relationship directly
                        podcast_id=podcast.id,
                        guid=guid,
                        title=episode_title,
                        description=entry.get('description', ''),
                        published_at=published_at,  # We already have this from earlier
                        audio_url=audio_url
                    )

                    # Verify the relationship is set
                    if not episode.podcast or not episode.podcast.title:
                        raise Exception("Failed to establish podcast relationship")
                except Exception as e:
                    if progress_callback:
                        progress_callback({
                            'stage': 'error',
                            'error': f"Failed to create episode: {str(e)}"
                        })
                    continue

                try:
                    if progress_callback:
                        progress_callback({
                            'stage': 'episode_start',
                            'podcast': sub,
                            'episode': {
                                'title': episode.title,
                                'published_at': published_at
                            },
                            'current': entry_idx,
                            'total': len(recent_entries)
                        })

                    # Download audio
                    if progress_callback:
                        progress_callback({'stage': 'downloading', 'podcast': sub, 'episode': {'title': episode.title}})
                    try:
                        temp_path = self._download_audio(audio_url)
                    except requests.exceptions.RequestException as e:
                        raise Exception(f"Failed to download audio: {str(e)}")

                    # Transcribe
                    if progress_callback:
                        progress_callback({'stage': 'transcribing', 'podcast': sub, 'episode': {'title': episode.title}})
                    try:
                        episode.transcript = self._transcribe_audio(temp_path, episode.title, progress_callback, debug=debug)
                    except Exception as e:
                        raise Exception(f"Failed to transcribe audio: {str(e)}")

                    # Generate embedding
                    if progress_callback:
                        progress_callback({'stage': 'embedding', 'podcast': sub, 'episode': {'title': episode.title}})
                    try:
                        episode.vector_embedding = json.dumps(self._generate_embedding(episode.transcript))
                    except Exception as e:
                        raise Exception(f"Failed to generate embedding: {str(e)}")

                    # Write to Obsidian if configured
                    if progress_callback:
                        progress_callback({'stage': 'exporting', 'podcast': sub, 'episode': {'title': episode.title}})
                    try:
                        self._write_to_obsidian(episode)
                    except Exception as e:
                        raise Exception(f"Failed to write to Obsidian: {str(e)}")

                    # Report completion
                    if progress_callback:
                        progress_callback({
                            'stage': 'episode_complete',
                            'podcast': sub,
                            'episode': {'title': episode.title}
                        })

                    # Cleanup
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Error processing episode {episode.title}: {e}")
                    if progress_callback:
                        progress_callback({
                            'stage': 'error',
                            'podcast': sub,
                            'episode': {'title': episode.title},
                            'error': str(e)
                        })
                    continue

                self.db.add(episode)
                self.db.commit()

    def _find_keyword_excerpt(self, keyword: str, transcript: str, context_chars: int = 100) -> str:
        """Find an excerpt of text containing the keyword with surrounding context."""
        keyword_lower = keyword.lower()
        transcript_lower = transcript.lower()
        
        # Find the position of the keyword
        pos = transcript_lower.find(keyword_lower)
        if pos == -1:
            return ""
            
        # Get surrounding context
        start = max(0, pos - context_chars)
        end = min(len(transcript), pos + len(keyword) + context_chars)
        
        # Add ellipsis if we truncated the text
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(transcript) else ""
        
        # Get the excerpt with original casing
        excerpt = transcript[start:end]
        
        return prefix + excerpt + suffix
        
    def keyword_search(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search through podcast transcripts for exact keyword matches.
        
        Args:
            keyword: Exact text to search for (case-insensitive)
            limit: Maximum number of results to return
            
        Returns:
            List of dicts containing episodes with matching transcripts
        """
        results = []
        keyword_lower = keyword.lower()
        
        # Search through all episodes with transcripts
        for episode in self.db.query(Episode).filter(Episode.transcript.isnot(None)):
            if keyword_lower in episode.transcript.lower():
                # Find a relevant excerpt containing the keyword
                excerpt = self._find_keyword_excerpt(keyword, episode.transcript)
                
                results.append({
                    'podcast': episode.podcast.title,
                    'episode': episode.title,
                    'published_at': episode.published_at,
                    'excerpt': excerpt
                })
                
                if len(results) >= limit:
                    break
                    
        return results
        
    def search(self, query: str, limit: int = 10, relevance_threshold: float = 0.60) -> List[Dict]:
        """Search through podcast transcripts using vector similarity.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            relevance_threshold: Minimum similarity score (0.0 to 1.0) for results
            
        Returns:
            List of dicts containing search results above the relevance threshold
        """
        self._load_embedding_model()
        query_embedding = self._generate_embedding(query)

        # This is a simple implementation - in production you'd want to use a proper vector database
        results = []
        for episode in self.db.query(Episode).filter(Episode.vector_embedding.isnot(None)):
            episode_embedding = json.loads(episode.vector_embedding)
            # Compute cosine similarity
            similarity = sum(a * b for a, b in zip(query_embedding, episode_embedding))
            
            # Only include results above threshold
            if similarity >= relevance_threshold:
                # Find a relevant excerpt
                excerpt = self._find_keyword_excerpt(query, episode.transcript)
                
                results.append({
                    'episode': episode,
                    'similarity': similarity,
                    'excerpt': excerpt
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
