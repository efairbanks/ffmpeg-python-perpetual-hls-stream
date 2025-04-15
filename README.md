# Audio Processor

A modern Python application for processing audio data using ffmpeg-python.

## Features

- Process raw audio data from stdin
- Continuously write to a wav file
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

```bash
# Process audio data from stdin and write to output.wav
cat input.raw | poetry run python -m audio_processor.main --output output.wav
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

## License

MIT
