import os
import glob
import whisper
from pathlib import Path

def main():
    print("\nAvailable Whisper models:")
    for model in whisper.available_models():
        print(f"- {model}")
    
    print("\nDownloaded models:")
    cache_dir = os.path.expanduser("~/.cache/whisper")
    if os.path.exists(cache_dir):
        for model_path in glob.glob(os.path.join(cache_dir, "*.pt")):
            model_name = Path(model_path).stem
            print(f"- {model_name}")
    else:
        print("No models found in cache directory")
    
    print("\nDefault cache directory:", cache_dir)

if __name__ == "__main__":
    main()
