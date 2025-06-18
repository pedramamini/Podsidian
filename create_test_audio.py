#!/usr/bin/env python3
"""
Create a simple test audio file for testing the ingest-local functionality.
This script generates a short WAV file with a spoken test message.
"""

import numpy as np
import wave
import os
from pathlib import Path

def create_test_audio(output_path: str, duration_seconds: float = 10.0, sample_rate: int = 16000):
    """Create a simple test audio file with a tone and text-to-speech if available."""
    
    # Try to use text-to-speech first
    try:
        import pyttsx3
        
        # Initialize text-to-speech engine
        engine = pyttsx3.init()
        
        # Set properties for better quality
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        
        # Create a temporary file for TTS output
        temp_path = output_path.replace('.wav', '_temp.wav')
        
        # Generate speech
        text = """Hello, this is a test audio file for Podsidian local audio ingestion. 
                  This file contains a brief spoken message that will be transcribed by Whisper
                  and processed through the Podsidian pipeline. The transcription should capture
                  this text and demonstrate the local audio file functionality working correctly."""
        
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        # Check if TTS file was created successfully
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            # Move the file to the final location
            os.rename(temp_path, output_path)
            return True
            
    except ImportError:
        print("pyttsx3 not available, falling back to tone generation")
    except Exception as e:
        print(f"TTS failed: {e}, falling back to tone generation")
    
    # Fallback: Create a simple tone-based audio file
    print("Creating tone-based test audio...")
    
    # Generate a simple tone sequence
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Create a sequence of different tones to make it interesting
    audio = np.zeros_like(t)
    
    # Add multiple frequency components
    frequencies = [440, 554, 659, 880]  # A major chord progression
    for i, freq in enumerate(frequencies):
        start_time = i * (duration_seconds / len(frequencies))
        end_time = (i + 1) * (duration_seconds / len(frequencies))
        
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        if end_idx <= len(audio):
            tone_segment = t[start_idx:end_idx] - start_time
            audio[start_idx:end_idx] += 0.3 * np.sin(2 * np.pi * freq * tone_segment)
    
    # Add some variation to make it more speech-like
    envelope = np.exp(-t / (duration_seconds / 2))  # Decay envelope
    audio *= envelope
    
    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return True

def main():
    # Create test audio in the project directory
    project_root = Path(__file__).parent
    test_audio_path = project_root / "test_audio.wav"
    
    print(f"Creating test audio file: {test_audio_path}")
    
    try:
        success = create_test_audio(str(test_audio_path))
        if success and os.path.exists(test_audio_path):
            file_size = os.path.getsize(test_audio_path)
            print(f"✓ Test audio file created successfully!")
            print(f"  Path: {test_audio_path}")
            print(f"  Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"\nTo test your ingest-local command, run:")
            print(f"  podsidian ingest-local {test_audio_path}")
            print(f"  podsidian ingest-local {test_audio_path} --title \"My Test Audio\" --debug")
        else:
            print("✗ Failed to create test audio file")
            return 1
            
    except Exception as e:
        print(f"✗ Error creating test audio: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())