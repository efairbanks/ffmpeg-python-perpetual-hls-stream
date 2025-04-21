# Audio Processor

A modern Python application for processing audio data in real-time using ffmpeg-python. This project demonstrates how to create an HTTP Live Streaming (HLS) pipeline with precise timing control.

## Features

- Process raw audio data from stdin in real-time
- Generate synthetic audio with precise timing and segment control
- Output to HLS format with configurable segment length
- Maintain phase continuity between audio segments
- Compensate for timing jitter to ensure accurate playback
- High-quality code with ruff and mypy integration
- Comprehensive test suite with pytest

## Requirements

- Python 3.10+
- FFmpeg installed on your system

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd audio-processor

# Install with Poetry
poetry install
```

## Usage

### Running the Demo

The easiest way to try the project is to run the included demo script:

```bash
# Run the demo (generates real-time audio and processes it to HLS)
./scripts/demo.sh
```

This will generate a continuous stream of audio data and process it into an HLS playlist and segments in the `output` directory.

### Manual Usage

#### Generating Real-time Audio

```bash
# Generate infinite real-time audio with 2-second segments
python scripts/generate_test_audio.py --target-segment-length 2

# Generate 60 seconds of real-time audio with 1-second segments
python scripts/generate_test_audio.py --target-segment-length 1 --duration 60
```

#### Processing Audio to HLS

```bash
# Process audio data from stdin and write to HLS format
cat input.raw | poetry run python -m audio_processor.main --output output/stream.m3u8

# Specify custom segment length
cat input.raw | poetry run python -m audio_processor.main --output output/stream.m3u8 --target-segment-length 5
```

#### Complete Pipeline

```bash
# Generate and process in a pipeline
python scripts/generate_test_audio.py --target-segment-length 2 | \
  poetry run python -m audio_processor.main --output output/stream.m3u8 --target-segment-length 2
```

## Development

This project uses Poetry for dependency management, ruff for linting, mypy for type checking, and pytest for testing.

```bash
# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Run type checking
poetry run mypy .
```

## Project Structure

- `audio_processor/main.py` - Main module for processing audio to HLS format
- `scripts/generate_test_audio.py` - Utility for generating real-time test audio
- `scripts/demo.sh` - Demo script to showcase the full pipeline
- `tests/` - Test suite for the project

## Implementation Details

### Real-time Audio Generation

The `generate_test_audio.py` script generates audio in real-time with precise timing control:

- Produces audio in configurable segment lengths
- Maintains phase continuity between segments
- Compensates for timing jitter to ensure accurate playback
- Provides detailed status information during generation

### HLS Output

The main processor converts raw PCM audio to HLS format:

- Creates a master playlist (.m3u8) and segment files (.ts)
- Configurable segment length via `--target-segment-length`
- Handles streaming input for continuous processing

## License

MIT
