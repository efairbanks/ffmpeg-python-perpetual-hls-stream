#!/bin/bash
# Demo script to test the audio processor

# Ensure we're in the project root directory
cd "$(dirname "$0")/.." || exit 1

# Create output directory if it doesn't exist
mkdir -p output

# Generate test audio data in real-time and pipe it to the audio processor
echo "Generating real-time test audio and processing it to output/test.m3u8..."

# Set the target segment length (in seconds)
TARGET_SEGMENT_LENGTH=2

# Run the audio generation and processing pipeline
python scripts/generate_test_audio.py \
  --target-segment-length "$TARGET_SEGMENT_LENGTH" \
  --frequency 440 | \
  poetry run python -m audio_processor.main \
  --output output/test.m3u8 \
  --target-segment-length "$TARGET_SEGMENT_LENGTH"

echo "Done! The processed HLS stream is at output/test.m3u8"
