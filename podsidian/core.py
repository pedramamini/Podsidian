import os
import re
import json
import time
import tempfile
import feedparser
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from sqlalchemy.orm import Session

from .models import Podcast, Episode
from .apple_podcasts import get_subscriptions

class PodcastProcessor:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.whisper_model = None
        self.embedding_model = None
        self.annoy_index = None
        self.episode_map = {}  # Maps Annoy index to Episode ID

        # Import here to avoid circular imports
        from .config import config
        self.config = config

    def _load_whisper(self):
        """Lazy load whisper model with configured options."""
        if not self.whisper_model:
            import torch
            import warnings
            import whisper

            # Set tokenizers parallelism
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            # Set ffmpeg path in environment if configured
            if self.config.ffmpeg_path:
                os.environ['PATH'] = f"{os.path.dirname(self.config.ffmpeg_path)}:{os.environ.get('PATH', '')}"
                if hasattr(self, '_progress_callback') and self._debug:
                    self._progress_callback({
                        'stage': 'debug',
                        'message': f'Set ffmpeg path: {self.config.ffmpeg_path}'
                    })

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
        """Lazy load sentence transformer model.

        Uses a powerful multilingual model (paraphrase-multilingual-mpnet-base-v2)
        that's better suited for semantic search and natural language understanding.
        """
        if not self.embedding_model:
            # This model is better at capturing semantic meaning across languages
            # and produces higher quality sentence embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

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

    def _download_transcript(self, transcript_url: str, progress_callback=None) -> str:
        """Download and process an external transcript.

        Args:
            transcript_url: URL to the transcript file
            progress_callback: Optional callback for progress updates

        Returns:
            Processed transcript text
        """
        if progress_callback:
            progress_callback({
                'stage': 'downloading_transcript',
                'message': f'Downloading external transcript from {transcript_url}'
            })
            
        response = requests.get(transcript_url, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        
        if progress_callback:
            progress_callback({
                'stage': 'processing_transcript',
                'message': f'Processing external transcript (format: {content_type})'
            })
        
        # Process different transcript formats
        if 'json' in content_type or transcript_url.endswith('.json'):
            # Try to parse JSON transcript (common format)
            try:
                data = response.json()
                # Handle different JSON transcript formats
                if isinstance(data, list) and all('text' in item for item in data):
                    # Format with list of segments with text
                    return ' '.join(item['text'] for item in data)
                elif 'transcript' in data:
                    # Simple format with transcript field
                    return data['transcript']
                elif 'results' in data and 'transcripts' in data['results']:
                    # AWS Transcribe format
                    return data['results']['transcripts'][0]['transcript']
                else:
                    # Unknown format, return the raw text
                    return json.dumps(data)
            except json.JSONDecodeError:
                # If JSON parsing fails, treat as plain text
                return response.text
        elif 'text/vtt' in content_type or transcript_url.endswith('.vtt'):
            # WebVTT format
            lines = []
            for line in response.text.splitlines():
                # Skip WebVTT headers, timestamps, and empty lines
                if not line.strip() or line.startswith('WEBVTT') or '-->' in line or line[0].isdigit():
                    continue
                lines.append(line)
            return ' '.join(lines)
        elif 'text/srt' in content_type or transcript_url.endswith('.srt'):
            # SRT format
            lines = []
            for line in response.text.splitlines():
                # Skip SRT headers, timestamps, and empty lines
                if not line.strip() or '-->' in line or line[0].isdigit():
                    continue
                lines.append(line)
            return ' '.join(lines)
        else:
            # Default to plain text
            return response.text
    
    def _detect_topic(self, title: str, transcript_sample: str) -> str:
        """Detect the main topic/domain of the podcast using OpenRouter."""
        # Import cost tracker here to avoid circular imports
        from .cost_tracker import track_api_call
        
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
        response_data = response.json()
        
        # Track the cost of this API call if enabled
        if self.config.cost_tracking_enabled:
            track_api_call(response_data, self.config.openrouter_processing_model)
            
        return response_data["choices"][0]["message"]["content"].strip()

    def _correct_transcript(self, transcript: str, domain: str) -> str:
        """Correct transcript using domain-specific knowledge."""
        # Import cost tracker here to avoid circular imports
        from .cost_tracker import track_api_call
        
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

Provide your response in the following format:

CORRECTED TRANSCRIPT:
[Your corrected transcript here, maintaining all original formatting and structure]

CHANGES MADE:
- List each significant correction you made
- Include the original text and what you changed it to
- If no changes were needed, state that"""

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
        response_data = response.json()
        
        # Track the cost of this API call if enabled
        if self.config.cost_tracking_enabled:
            track_api_call(response_data, self.config.openrouter_processing_model)

        # Parse response
        content = response_data["choices"][0]["message"]["content"].strip()

        # Split into transcript and changes
        parts = content.split("\nCHANGES MADE:")
        if len(parts) == 2:
            corrected_transcript = parts[0].replace("CORRECTED TRANSCRIPT:\n", "").strip()
            changes = parts[1].strip()
        else:
            corrected_transcript = content
            changes = "No changes reported"

        if hasattr(self, '_progress_callback') and self._progress_callback:
            self._progress_callback({
                'stage': 'transcript_correction',
                'message': f"Domain Expert: {domain}\nChanges Made:\n{changes}"
            })

        return corrected_transcript

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
            
            # Track audio duration for cost tracking
            if self.config.cost_tracking_enabled:
                from .cost_tracker import track_api_call
                track_api_call({}, f"whisper/{self.config.whisper_model}", result.get('segments', [{}])[-1].get('end', 0))

            if progress_callback:
                progress_callback({
                    'stage': 'transcription',
                    'message': f'Raw transcript length: {len(result["text"])} chars'
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
        if progress_callback:
            progress_callback({
                'stage': 'domain_detection',
                'message': 'Detecting podcast domain for specialized transcript correction...'
            })
        domain = self._detect_topic(title, transcript_sample)
        if progress_callback:
            progress_callback({
                'stage': 'domain_detected',
                'message': f'Detected domain: {domain}'
            })

        # Correct the transcript using domain expertise
        corrected_transcript = self._correct_transcript(initial_transcript, domain)

        return corrected_transcript

    def _generate_embedding(self, text: str, normalize: bool = True) -> 'np.ndarray':
        """Generate vector embedding for text.

        Args:
            text: Text to generate embedding for
            normalize: Whether to normalize the embedding vector (default: True)

        Returns:
            Numpy array representing the text embedding
        """
        self._load_embedding_model()
        embedding = self.embedding_model.encode([text])[0]
        if normalize:
            import numpy as np
            embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _get_summary(self, transcript: str) -> str:
        """Get AI-generated summary using OpenRouter."""
        if not self.config.openrouter_api_key:
            return "No OpenRouter API key configured. Summary not available."

        import requests
        # Import cost tracker here to avoid circular imports
        from .cost_tracker import track_api_call

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
        response_data = response.json()
        
        # Track the cost of this API call if enabled
        if self.config.cost_tracking_enabled:
            track_api_call(response_data, self.config.openrouter_model)
            
        return response_data["choices"][0]["message"]["content"]

    def _get_value_analysis(self, transcript: str) -> str:
        """Get value analysis using OpenRouter if enabled."""
        if not self.config.value_prompt_enabled or not self.config.openrouter_api_key:
            return ""

        import requests
        # Import cost tracker here to avoid circular imports
        from .cost_tracker import track_api_call

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/pedramamini/podsidian",
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": self.config.openrouter_model,
                "messages": [{
                    "role": "user",
                    "content": self.config.value_prompt.format(transcript=transcript)
                }]
            }
        )

        response.raise_for_status()
        response_data = response.json()
        
        # Track the cost of this API call if enabled
        if self.config.cost_tracking_enabled:
            track_api_call(response_data, self.config.openrouter_model)
            
        return response_data["choices"][0]["message"]["content"]

    def _make_safe_filename(self, s: str) -> str:
        """Convert string to a safe filename that's safe for macOS."""
        # Remove content within brackets and parentheses
        s = re.sub(r'\s*\[[^\]]*\]\s*', '', s)  # Remove [content]
        s = re.sub(r'\s*\([^\)]*\)\s*', '', s)  # Remove (content)

        # Remove or replace unsafe characters for macOS
        # macOS doesn't allow these characters: /, :, [], |, ^, #
        # The colon (:) is especially problematic on macOS
        s = re.sub(r'[<>:"/\\|?*\[\]\|\^#]', '', s)  # Remove all unsafe chars including colon
        
        # Replace multiple spaces with single space
        s = re.sub(r'\s+', ' ', s)
        
        # Remove leading/trailing spaces and dots
        s = s.strip('. ')
        
        # macOS has issues with files starting with a dot (hidden files)
        if s.startswith('.'):
            s = 'file_' + s
            
        # macOS doesn't like filenames ending with a space
        s = s.rstrip()
        
        # macOS doesn't like filenames that are just a dot
        if s in ['.', '..'] or not s:
            s = 'untitled_file'
            
        # Ensure filename isn't too long (max 255 bytes in UTF-8 for macOS)
        # Using 240 to be safe with UTF-8 encoding and extension
        return s[:240]

    def _get_podcast_app_url(self, audio_url: str, guid: str = None, title: str = None) -> str:
        """Get the podcast:// URL for opening in Apple Podcasts app.

        This uses the apple_podcasts module to find the appropriate URL.
        If a GUID is provided, it will be used to look up the podcast in the Apple Podcasts database.
        If no match is found by GUID, it will try to match by title.

        Args:
            audio_url: The audio URL from the episode
            guid: Optional episode GUID to use for lookup in Apple Podcasts database
            title: Optional episode title to use for lookup in Apple Podcasts database
        """
        from .apple_podcasts import get_podcast_app_url
        return get_podcast_app_url(audio_url, guid, title)

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

        # Get value analysis if enabled
        value_analysis = self._get_value_analysis(episode.transcript) if episode.transcript else ""

        # Calculate transcript word count if transcript exists
        transcript_wordcount = len(episode.transcript.split()) if episode.transcript else 0

        # Extract podcast app URL using audio URL, GUID, and title
        podcasts_app_url = self._get_podcast_app_url(episode.audio_url, episode.guid, episode.title) if episode.audio_url else "N/A"

        # Format note using template
        note_content = self.config.note_template.format(
            title=episode.title,
            podcast_title=episode.podcast.title,
            published_at=episode.published_at.strftime('%Y-%m-%d') if episode.published_at else 'Unknown',
            audio_url=episode.audio_url,
            podcasts_app_url=podcasts_app_url,
            summary=summary,
            value_analysis=value_analysis,
            transcript=episode.transcript or "Transcript not available",
            episode_id=episode.id,
            episode_wordcount=transcript_wordcount,
            podcast_guid=episode.guid
        )

        md_file = vault_path / filename
        with md_file.open('w') as f:
            f.write(note_content)

        # Mark episode as processed
        episode.processed_at = datetime.utcnow()
        self.db.commit()

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
                            feed_url=sub['feed_url'],
                            muted=False
                        )
                        self.db.add(podcast)
                        self.db.commit()

                    # Skip muted podcasts
                    if podcast.muted:
                        if progress_callback:
                            progress_callback({
                                'stage': 'skip',
                                'message': f"Skipping muted podcast: {podcast.title}"
                            })
                        continue

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
                        # If episode exists but hasn't been processed, process it
                        if not existing.processed_at:
                            episode = existing
                        else:
                            continue

                # Find audio URL
                audio_url = None
                transcript_url = None
                
                # Look for audio and transcript links
                for link in entry.get('links', []):
                    link_type = link.get('type', '').lower()
                    if link_type.startswith('audio/'):
                        audio_url = link['href']
                    elif link_type in ['application/json', 'text/vtt', 'text/srt', 'text/plain'] or \
                         link.get('rel', '') == 'transcript' or \
                         'transcript' in link.get('href', '').lower():
                        transcript_url = link['href']
                
                # Also check for transcript in enclosures
                if not transcript_url and 'enclosures' in entry:
                    for enclosure in entry.get('enclosures', []):
                        enclosure_type = enclosure.get('type', '').lower()
                        if enclosure_type in ['application/json', 'text/vtt', 'text/srt', 'text/plain'] or \
                           'transcript' in enclosure.get('href', '').lower():
                            transcript_url = enclosure.get('href')
                            break
                
                # Skip if no audio URL found
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
                        guid=guid,
                        title=episode_title,
                        description=entry.get('description', ''),
                        published_at=published_at,  # We already have this from earlier
                        audio_url=audio_url,
                        transcript_url=transcript_url
                    )

                    # Add episode to session before setting relationships
                    self.db.add(episode)
                    episode.podcast = podcast
                    episode.podcast_id = podcast.id

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
                        start_time = time.time()
                        temp_path = self._download_audio(audio_url)
                        download_time = time.time() - start_time

                        if progress_callback:
                            progress_callback({
                                'stage': 'timing',
                                'message': f'Audio download took {download_time:.2f} seconds'
                            })
                    except requests.exceptions.RequestException as e:
                        raise Exception(f"Failed to download audio: {str(e)}")

                    # Check if we have an external transcript URL
                    if episode.transcript_url:
                        if progress_callback:
                            progress_callback({
                                'stage': 'external_transcript',
                                'podcast': sub, 
                                'episode': {'title': episode.title},
                                'message': f'Using external transcript from {episode.transcript_url}'
                            })
                        try:
                            start_time = time.time()
                            episode.transcript = self._download_transcript(episode.transcript_url, progress_callback)
                            episode.transcript_source = 'external'
                            transcript_time = time.time() - start_time

                            if progress_callback:
                                progress_callback({
                                    'stage': 'timing',
                                    'message': f'External transcript processing took {transcript_time:.2f} seconds'
                                })
                        except Exception as e:
                            if progress_callback:
                                progress_callback({
                                    'stage': 'warning',
                                    'message': f'Failed to use external transcript, falling back to Whisper: {str(e)}'
                                })
                            # Fall back to Whisper transcription
                            episode.transcript_url = None
                    
                    # If no external transcript or it failed, use Whisper
                    if not episode.transcript_url:
                        if progress_callback:
                            progress_callback({
                                'stage': 'info',
                                'message': 'No external transcript found, falling back to local transcription'
                            })
                            progress_callback({'stage': 'transcribing', 'podcast': sub, 'episode': {'title': episode.title}})
                        try:
                            start_time = time.time()
                            episode.transcript = self._transcribe_audio(temp_path, episode.title, progress_callback, debug=debug)
                            episode.transcript_source = 'whisper'
                            transcribe_time = time.time() - start_time

                            if progress_callback:
                                progress_callback({
                                    'stage': 'timing',
                                    'message': f'Audio transcription took {transcribe_time:.2f} seconds'
                                })
                        except Exception as e:
                            raise Exception(f"Failed to transcribe audio: {str(e)}")

                    # Generate embedding
                    if progress_callback:
                        progress_callback({'stage': 'embedding', 'podcast': sub, 'episode': {'title': episode.title}})
                    try:
                        start_time = time.time()
                        embedding = self._generate_embedding(episode.transcript)
                        episode.vector_embedding = json.dumps(embedding.tolist())
                        embedding_time = time.time() - start_time

                        if progress_callback:
                            progress_callback({
                                'stage': 'timing',
                                'message': f'Embedding generation took {embedding_time:.2f} seconds'
                            })
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

                # Rebuild Annoy index after successful episode processing
                if episode.vector_embedding:
                    try:
                        start_time = time.time()
                        self._init_annoy_index(force_rebuild=True)
                        index_time = time.time() - start_time

                        if progress_callback:
                            progress_callback({
                                'stage': 'timing',
                                'message': f'Index rebuild took {index_time:.2f} seconds'
                            })
                    except Exception as e:
                        if progress_callback:
                            progress_callback({
                                'stage': 'error',
                                'message': f'Failed to rebuild search index: {str(e)}'
                            })

    def _find_relevant_excerpt(self, query: str, transcript: str, context_chars: int = None) -> str:
        """Find the most relevant excerpt from the transcript for the given query.

        Uses keyword matching and surrounding context to find relevant excerpts.
        This is a faster alternative to semantic search for excerpt finding.

        Args:
            query: Search query
            transcript: Full transcript text
            context_chars: Number of characters of context to include

        Returns:
            Most relevant excerpt from the transcript
        """
        if not transcript:
            return ""
            
        # Use configured excerpt length if not specified
        if context_chars is None:
            context_chars = self.config.search_excerpt_length

        # Use simpler keyword matching for excerpts
        query_terms = query.lower().split()
        transcript_lower = transcript.lower()

        # Find positions of query terms
        positions = []
        for term in query_terms:
            pos = transcript_lower.find(term)
            if pos != -1:
                positions.append(pos)

        if not positions:
            # If no exact matches, take a chunk from the beginning
            start = 0
            end = min(300, len(transcript))
        else:
            # Use the position with the most surrounding term matches
            best_pos = max(positions, key=lambda p: sum(1 for pos in positions if abs(pos - p) < 200))
            start = max(0, best_pos - context_chars)
            end = min(len(transcript), best_pos + context_chars)

        # Add ellipsis for truncated text
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(transcript) else ""

        # Get the excerpt and clean it up
        excerpt = transcript[start:end].strip()

        # Ensure we don't break in the middle of a word
        if prefix:
            excerpt = excerpt[excerpt.find(" ")+1:]
        if suffix:
            excerpt = excerpt[:excerpt.rfind(" ")]

        return prefix + excerpt + suffix



    def _init_annoy_index(self, force_rebuild: bool = False) -> None:
        """Initialize or load the Annoy index.

        Args:
            force_rebuild: If True, rebuild the index from scratch using database contents
        """
        # Get embedding dimension from model
        if self.embedding_model is None:
            self._load_embedding_model()
        sample_embedding = self.embedding_model.encode(["sample text"])[0]
        embedding_dim = len(sample_embedding)

        # Create new index
        from annoy import AnnoyIndex
        self.annoy_index = AnnoyIndex(embedding_dim, self.config.annoy_metric)

        # Load existing index if it exists and we're not forcing a rebuild
        index_path = self.config.annoy_index_path
        episodes_with_embeddings = self.db.query(Episode).filter(Episode.vector_embedding.isnot(None)).all()

        if os.path.exists(index_path) and not force_rebuild:
            self.annoy_index.load(index_path)
            # Check if we need to rebuild by comparing episode count
            if len(episodes_with_embeddings) <= self.annoy_index.get_n_items():
                # Just rebuild episode map without modifying index
                self.episode_map = {}
                for idx, episode in enumerate(episodes_with_embeddings):
                    self.episode_map[idx] = episode.id
                return

        # If we get here, we need to rebuild the index
        self.episode_map = {}
        for episode in episodes_with_embeddings:
            import numpy as np
            idx = len(self.episode_map)
            self.episode_map[idx] = episode.id
            self.annoy_index.add_item(idx, np.array(json.loads(episode.vector_embedding)))

        # Build and save index
        self.annoy_index.build(self.config.annoy_n_trees)
        self.annoy_index.save(index_path)

    def search(self, query: str, limit: int = 10, relevance_threshold: float = 0.60) -> List[Dict]:
        """Search through podcast content using natural language understanding.

        This implementation is inspired by Spotify's podcast search, using semantic
        similarity to match content even when exact keywords don't match. It uses
        Annoy for fast approximate nearest neighbor search.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            relevance_threshold: Minimum similarity score (0.0 to 1.0) for results

        Returns:
            List of dicts containing search results above the relevance threshold
        """
        self._load_embedding_model()

        # Initialize Annoy index if needed
        if self.annoy_index is None:
            self._init_annoy_index()

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Get nearest neighbors from Annoy
        n_results = min(limit * 2, self.annoy_index.get_n_items())  # Get extra results for filtering
        indices, distances = self.annoy_index.get_nns_by_vector(
            query_embedding, n_results, include_distances=True)

        # Convert distances to similarities (Annoy returns squared L2 distance for 'angular')
        similarities = [1 - (d / 2) for d in distances]  # Convert to cosine similarity

        results = []
        for idx, similarity in zip(indices, similarities):
            if similarity < relevance_threshold:
                continue

        # Batch fetch all potential episodes
        episode_ids = [self.episode_map[idx] for idx in indices]
        episodes = {e.id: e for e in self.db.query(Episode).filter(Episode.id.in_(episode_ids)).all()}

        for idx, similarity in zip(indices, similarities):
            if similarity < relevance_threshold:
                continue

            # Get episode from cached results
            episode = episodes.get(self.episode_map[idx])
            if not episode:
                continue

            # Get episode metadata
            title = episode.title
            description = episode.description or ''
            podcast_title = episode.podcast.title

            # Create rich text representation for better context
            rich_text = f"{podcast_title}. {title}. {description}"

            # Boost score if query terms appear in title/description
            query_terms = set(query.lower().split())
            text_terms = set(rich_text.lower().split())
            term_overlap = len(query_terms & text_terms) / len(query_terms)

            # Combine semantic and term-based scores
            combined_score = (0.7 * similarity) + (0.3 * term_overlap)

            # Only include results above threshold
            if combined_score >= relevance_threshold:
                # Find most relevant excerpt
                excerpt = self._find_relevant_excerpt(query, episode.transcript)

                results.append({
                    'episode': episode,
                    'similarity': combined_score,
                    'excerpt': excerpt
                })

        # Sort by combined score and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return [
            {
                'podcast': r['episode'].podcast.title,
                'episode': r['episode'].title,
                'excerpt': r['excerpt'],
                'published_at': r['episode'].published_at,
                'similarity': round(r['similarity'] * 100),  # Convert to percentage
                'id': r['episode'].id  # Add episode ID
            }
            for r in results[:limit]
        ]
