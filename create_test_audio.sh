#!/bin/bash

# Create test audio files for testing Podsidian local audio functionality
# This script tries multiple methods to create a test audio file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/test_audio.wav"

echo "🎵 Creating test audio file for Podsidian local audio testing"
echo "Output file: $OUTPUT_FILE"
echo ""

# Method 1: Try macOS text-to-speech
if command -v say >/dev/null 2>&1; then
    echo "📢 Using macOS text-to-speech..."
    TEXT="This is a test audio file for Podsidian local audio ingestion. The transcription should capture this text and demonstrate the local audio file functionality working correctly. This test will verify that Whisper can transcribe the audio and that the embedding generation works properly."
    
    # Create AIFF first, then convert to WAV
    TEMP_AIFF="$SCRIPT_DIR/temp_test.aiff"
    say "$TEXT" -o "$TEMP_AIFF"
    
    if command -v ffmpeg >/dev/null 2>&1; then
        echo "🔄 Converting to WAV format..."
        ffmpeg -i "$TEMP_AIFF" -ar 16000 -ac 1 "$OUTPUT_FILE" -y
        rm -f "$TEMP_AIFF"
    else
        mv "$TEMP_AIFF" "$OUTPUT_FILE"
    fi
    
    echo "✅ Test audio created using macOS text-to-speech"
    
# Method 2: Try Linux espeak
elif command -v espeak >/dev/null 2>&1; then
    echo "📢 Using espeak text-to-speech..."
    TEXT="This is a test audio file for Podsidian local audio ingestion. The transcription should capture this text and demonstrate the local audio file functionality working correctly."
    espeak "$TEXT" -w "$OUTPUT_FILE" -s 150
    echo "✅ Test audio created using espeak"
    
# Method 3: Try FFmpeg tone generation
elif command -v ffmpeg >/dev/null 2>&1; then
    echo "🎵 Using FFmpeg to generate test tone..."
    # Create a more complex tone sequence to make it interesting
    ffmpeg -f lavfi -i "sine=frequency=440:duration=2,sine=frequency=554:duration=2,sine=frequency=659:duration=2,sine=frequency=880:duration=2" \
           -filter_complex "concat=n=4:v=0:a=1" \
           -ac 1 -ar 16000 "$OUTPUT_FILE" -y
    echo "✅ Test audio created using FFmpeg tone generation"
    
# Method 4: Try downloading a sample file
elif command -v curl >/dev/null 2>&1; then
    echo "📥 Downloading sample audio file..."
    curl -L -o "$OUTPUT_FILE" "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav" || {
        echo "❌ Failed to download sample file"
        exit 1
    }
    echo "✅ Test audio downloaded from web"
    
else
    echo "❌ No suitable tools found for creating test audio"
    echo ""
    echo "To create test audio, install one of these:"
    echo "  macOS: Uses built-in 'say' command"
    echo "  Linux: sudo apt-get install espeak"
    echo "  Cross-platform: Install FFmpeg"
    echo ""
    echo "Or manually create/download an audio file and place it at:"
    echo "  $OUTPUT_FILE"
    exit 1
fi

# Verify the file was created
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(wc -c < "$OUTPUT_FILE")
    echo ""
    echo "📊 File created successfully:"
    echo "  Path: $OUTPUT_FILE"
    echo "  Size: $FILE_SIZE bytes ($(echo "scale=1; $FILE_SIZE / 1024" | bc -l) KB)"
    
    # Show file type if 'file' command is available
    if command -v file >/dev/null 2>&1; then
        echo "  Type: $(file "$OUTPUT_FILE")"
    fi
    
    echo ""
    echo "🧪 To test your ingest-local command:"
    echo "  podsidian ingest-local '$OUTPUT_FILE'"
    echo "  podsidian ingest-local '$OUTPUT_FILE' --title 'My Test Audio' --debug"
    echo ""
    echo "🔍 To run the test helper script:"
    echo "  python test_local_audio.py '$OUTPUT_FILE'"
    
else
    echo "❌ Failed to create test audio file"
    exit 1
fi