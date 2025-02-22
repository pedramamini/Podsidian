#!/bin/bash
set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies with uv
echo "Installing dependencies..."
uv pip install -e ".[dev]"

echo "Development environment setup complete!"
echo "To activate the environment, run: source .venv/bin/activate"
