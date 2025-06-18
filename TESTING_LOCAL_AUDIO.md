# Testing Local Audio File Functionality

This guide helps you test the `podsidian ingest-local` command that was recently implemented.

## Quick Start

### Option 1: Use the Test Helper Scripts

1. **Find existing audio files:**
   ```bash
   python test_local_audio.py
   ```

2. **Create a test audio file:**
   ```bash
   chmod +x create_test_audio.sh
   ./create_test_audio.sh
   ```

3. **Test the command:**
   ```bash
   podsidian ingest-local test_audio.wav --debug
   ```

### Option 2: Manual Testing

1. **Find or create an audio file** in one of these formats:
   - `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.wma`

2. **Test the basic command:**
   ```bash
   podsidian ingest-local /path/to/your/audio/file.mp3
   ```

3. **Test with custom title and debug output:**
   ```bash
   podsidian ingest-local /path/to/your/audio/file.mp3 --title "My Test Episode" --debug
   ```

## What the Command Does

The `podsidian ingest-local` command processes a local audio file through the same pipeline as podcast episodes:

1. **Validation**: Checks if the file exists and is a supported audio format
2. **Podcast Creation**: Creates a "Local Files" podcast container if it doesn't exist
3. **Episode Metadata**: Generates episode metadata using the filename and file stats
4. **Transcription**: Uses OpenAI Whisper to transcribe the audio
5. **AI Processing**: Applies the same AI correction and domain detection as regular episodes
6. **Embedding Generation**: Creates semantic embeddings for search functionality
7. **Obsidian Export**: Exports the transcript as a markdown file to your Obsidian vault
8. **Database Storage**: Saves all data to the local database
9. **Search Index**: Updates the Annoy index for semantic search

## Expected Output

When you run the command successfully, you should see output like:

```
[2024-06-18 12:34:56] Processing local audio file: test_audio.wav
  → Validating audio file...
  → Transcribing audio...
  → Transcribing: [==============================] 100%
  → Generating embeddings...
  → Exporting to Obsidian...
  ✓ Processing complete
    Audio: 10.5 seconds
    Tokens: 1,234
    Cost: $0.001234

[2024-06-18 12:34:58] Local file processing complete!
Episode ID: 42
```

## Creating Test Audio Files

### Method 1: Text-to-Speech (macOS)
```bash
say "This is a test audio file for Podsidian" -o test.aiff
ffmpeg -i test.aiff test.wav  # Convert to WAV if needed
```

### Method 2: Text-to-Speech (Linux)
```bash
# Install espeak first: sudo apt-get install espeak
espeak "This is a test audio file for Podsidian" -w test.wav
```

### Method 3: FFmpeg Tone Generation
```bash
# Create a 10-second 440Hz tone
ffmpeg -f lavfi -i 'sine=frequency=440:duration=10' -ar 16000 test.wav
```

### Method 4: Record from Microphone
```bash
# Linux (ALSA)
ffmpeg -f alsa -i default -t 10 -ar 16000 test.wav

# macOS
ffmpeg -f avfoundation -i ':0' -t 10 -ar 16000 test.wav
```

### Method 5: Download Sample Audio
```bash
curl -o test.wav 'https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav'
```

## Troubleshooting

### File Format Issues
- **Error**: "Unsupported audio file format"
- **Solution**: Use one of the supported formats: mp3, wav, m4a, aac, ogg, flac, wma

### File Not Found
- **Error**: "Audio file not found"
- **Solution**: Use absolute paths or ensure the file exists: `ls -la /path/to/file.wav`

### Whisper Issues
- **Error**: Transcription fails
- **Solution**: Check if Whisper model is installed: `python -c "import whisper; print('OK')"`

### Configuration Issues
- **Error**: No vault path configured
- **Solution**: Run `podsidian init` and configure your settings in `~/.config/podsidian/config.toml`

### OpenRouter API Issues
- **Error**: AI processing fails
- **Solution**: Check your OpenRouter API key in the config file

## Verification Steps

After running `podsidian ingest-local`, verify the results:

1. **Check the database:**
   ```bash
   podsidian episodes
   # Should show your local file episode
   ```

2. **Check Obsidian vault:**
   - Look for a new markdown file in your configured vault path
   - The filename should include the date and episode title

3. **Test search functionality:**
   ```bash
   podsidian search "keywords from your audio"
   # Should find your local episode if it contains those words
   ```

4. **Check the "Local Files" podcast:**
   ```bash
   podsidian subscriptions list
   # Should show "Local Files" as a podcast
   ```

## Advanced Testing

### Test Multiple Files
```bash
# Process multiple local files
podsidian ingest-local file1.wav --title "Test Episode 1"
podsidian ingest-local file2.mp3 --title "Test Episode 2"
```

### Test Different Formats
```bash
# Test various audio formats
for ext in wav mp3 m4a aac ogg; do
  if [ -f "test.$ext" ]; then
    podsidian ingest-local "test.$ext" --title "Test $ext format"
  fi
done
```

### Test Error Handling
```bash
# Test with non-existent file
podsidian ingest-local nonexistent.wav

# Test with non-audio file
touch test.txt
podsidian ingest-local test.txt
```

## Cost Considerations

Local audio processing uses OpenRouter API for:
- AI-powered transcript correction
- Domain detection for better context
- Value analysis (if enabled)

Monitor costs with:
```bash
podsidian show-config  # Shows cost tracking status
```

Enable cost tracking in your config file:
```toml
cost_tracking_enabled = true
```

## Integration with Existing Workflow

Local audio files integrate seamlessly with your existing Podsidian workflow:

- **Search**: Local episodes appear in semantic search results
- **Obsidian**: Local episodes export to your vault like podcast episodes  
- **Database**: Local episodes are stored with episode IDs for easy reference
- **Backups**: Local episodes are included in database backups

The "Local Files" podcast acts as a container for all local audio, keeping them organized separately from your podcast subscriptions.