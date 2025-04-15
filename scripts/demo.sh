#!/bin/bash
# Demo script to test the audio processor

# Ensure we're in the project root directory
cd "$(dirname "$0")/.." || exit 1

# Create output directory if it doesn't exist
mkdir -p output

# Generate test audio data and pipe it to the audio processor
echo "Generating test audio and processing it to output/test.wav..."
python scripts/generate_test_audio.py --duration 5 --frequency 440 | \
  poetry run python -m audio_processor.main --output output/test.wav

echo "Done! The processed audio file is at output/test.wav"
