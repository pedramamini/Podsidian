#!/usr/bin/env python3
"""
Test script for the local audio file functionality in Podsidian.
This script helps you test the `podsidian ingest-local` command by:
1. Checking for existing audio files on the system
2. Providing suggestions for creating test audio files
3. Running basic validation tests
"""

import os
import sys
import subprocess
from pathlib import Path
import mimetypes

# Supported audio formats (from core.py)
SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
SUPPORTED_MIME_TYPES = {'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/aac', 'audio/ogg', 'audio/flac', 'audio/x-ms-wma'}

def find_audio_files(search_paths=None):
    """Find audio files in common locations."""
    if search_paths is None:
        search_paths = [
            Path.home() / "Downloads",
            Path.home() / "Music", 
            Path.home() / "Documents",
            Path("/tmp"),
            Path("/var/tmp"),
            Path.cwd()
        ]
    
    found_files = []
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        print(f"Searching in: {search_path}")
        
        try:
            # Search for audio files (limit depth to avoid scanning entire filesystem)
            for file_path in search_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Validate it's actually an audio file
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    if mime_type and mime_type in SUPPORTED_MIME_TYPES:
                        file_size = file_path.stat().st_size
                        if file_size > 1024:  # At least 1KB
                            found_files.append({
                                'path': file_path,
                                'size': file_size,
                                'mime_type': mime_type,
                                'extension': file_path.suffix.lower()
                            })
                            
                            # Limit to first few files per directory to avoid overwhelming output
                            if len([f for f in found_files if f['path'].parent == file_path.parent]) >= 3:
                                break
                                
        except PermissionError:
            continue
        except Exception as e:
            print(f"Warning: Error searching {search_path}: {e}")
            continue
    
    return found_files

def create_test_audio_suggestions():
    """Provide suggestions for creating test audio files."""
    suggestions = [
        {
            "method": "Text-to-Speech (macOS)",
            "command": "say 'This is a test audio file for Podsidian local audio ingestion. The transcription should capture this text and demonstrate the functionality working correctly.' -o test_audio.aiff && ffmpeg -i test_audio.aiff test_audio.wav",
            "description": "Uses macOS built-in text-to-speech to create a test audio file"
        },
        {
            "method": "Text-to-Speech (Linux - espeak)",
            "command": "espeak 'This is a test audio file for Podsidian local audio ingestion. The transcription should capture this text and demonstrate the functionality working correctly.' -w test_audio.wav",
            "description": "Uses espeak (install with: sudo apt-get install espeak)"
        },
        {
            "method": "FFmpeg tone generation",
            "command": "ffmpeg -f lavfi -i 'sine=frequency=440:duration=10' -ac 1 -ar 16000 test_audio.wav",
            "description": "Creates a 10-second 440Hz tone using FFmpeg"
        },
        {
            "method": "Record with FFmpeg (microphone)",
            "command": "ffmpeg -f alsa -i default -t 10 -ac 1 -ar 16000 test_recording.wav",
            "description": "Records 10 seconds from default microphone (Linux with ALSA)"
        },
        {
            "method": "Record with FFmpeg (macOS)",
            "command": "ffmpeg -f avfoundation -i ':0' -t 10 -ac 1 -ar 16000 test_recording.wav",
            "description": "Records 10 seconds from default microphone (macOS)"
        },
        {
            "method": "Download sample audio",
            "command": "curl -o test_sample.wav 'https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav'",
            "description": "Downloads a sample audio file from the web"
        }
    ]
    
    return suggestions

def test_podsidian_command(audio_file_path):
    """Test the podsidian ingest-local command with the given audio file."""
    if not Path(audio_file_path).exists():
        return False, f"File does not exist: {audio_file_path}"
    
    try:
        # Test the command (dry run style - just check if it would work)
        print(f"\nTesting: podsidian ingest-local {audio_file_path}")
        print("Command would process:")
        print(f"  File: {audio_file_path}")
        print(f"  Size: {Path(audio_file_path).stat().st_size:,} bytes")
        
        mime_type, _ = mimetypes.guess_type(audio_file_path)
        print(f"  MIME type: {mime_type}")
        
        # Check if file format is supported
        extension = Path(audio_file_path).suffix.lower()
        if extension in SUPPORTED_EXTENSIONS:
            print(f"  Format: ✓ Supported ({extension})")
        else:
            print(f"  Format: ✗ Unsupported ({extension})")
            return False, f"Unsupported format: {extension}"
            
        return True, "File appears compatible with podsidian ingest-local"
        
    except Exception as e:
        return False, f"Error testing file: {e}"

def main():
    print("🎵 Podsidian Local Audio Testing Helper")
    print("=" * 50)
    
    # Check for existing audio files
    print("\n1. Searching for existing audio files...")
    audio_files = find_audio_files()
    
    if audio_files:
        print(f"\n✓ Found {len(audio_files)} audio files:")
        for i, file_info in enumerate(audio_files[:10], 1):  # Show first 10
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"   {i}. {file_info['path']}")
            print(f"      Size: {size_mb:.1f} MB, Type: {file_info['mime_type']}")
        
        if len(audio_files) > 10:
            print(f"   ... and {len(audio_files) - 10} more files")
            
        # Test the first file
        test_file = audio_files[0]['path']
        print(f"\n2. Testing first file with podsidian...")
        success, message = test_podsidian_command(str(test_file))
        print(f"Result: {message}")
        
        if success:
            print(f"\n🎉 Ready to test! Run:")
            print(f"   podsidian ingest-local '{test_file}'")
            print(f"   podsidian ingest-local '{test_file}' --title 'My Test Audio' --debug")
        
    else:
        print("\n✗ No existing audio files found in common locations")
        
    # Show suggestions for creating test audio
    print(f"\n3. Suggestions for creating test audio files:")
    suggestions = create_test_audio_suggestions()
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n   {i}. {suggestion['method']}")
        print(f"      {suggestion['description']}")
        print(f"      Command: {suggestion['command']}")
    
    print(f"\n4. After creating a test audio file, test it with:")
    print(f"   python {__file__} <path_to_audio_file>")
    
    # If an audio file path was provided as argument, test it
    if len(sys.argv) > 1:
        test_file_path = sys.argv[1]
        print(f"\n5. Testing provided file: {test_file_path}")
        success, message = test_podsidian_command(test_file_path)
        print(f"Result: {message}")
        
        if success:
            print(f"\n🎉 File is ready for testing! Run:")
            print(f"   podsidian ingest-local '{test_file_path}'")
            print(f"   podsidian ingest-local '{test_file_path}' --title 'Custom Title' --debug")
    
    print(f"\n📚 For more information:")
    print(f"   - Check supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print(f"   - Use --debug flag to see detailed processing steps")
    print(f"   - Use --title to set a custom episode title")

if __name__ == "__main__":
    main()