#!/usr/bin/env python
"""Generate test audio data for testing the audio processor.

This script generates synthetic PCM audio data (sine wave) and writes it to stdout,
which can be piped into the audio processor for testing.
"""

import argparse
import math
import struct
import sys


def generate_sine_wave(
    duration: float = 5.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    channels: int = 2,
) -> None:
    """Generate a sine wave and write raw PCM data to stdout.

    Args:
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of the sine wave in Hz
        amplitude: Amplitude of the sine wave (0.0-1.0)
        channels: Number of audio channels
    """
    # Calculate the number of samples
    num_samples = int(duration * sample_rate)
    
    # Generate the sine wave
    for i in range(num_samples):
        # Calculate the value of the sine wave at this sample
        t = i / sample_rate
        value = int(32767 * amplitude * math.sin(2 * math.pi * frequency * t))
        
        # Convert to 16-bit PCM and write to stdout
        for _ in range(channels):
            sys.stdout.buffer.write(struct.pack("<h", value))


def main() -> None:
    """Parse command line arguments and generate audio data."""
    parser = argparse.ArgumentParser(
        description="Generate test audio data (sine wave) and write to stdout."
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Duration of the audio in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=440.0,
        help="Frequency of the sine wave in Hz (default: 440.0)"
    )
    parser.add_argument(
        "--amplitude", "-a",
        type=float,
        default=0.5,
        help="Amplitude of the sine wave (0.0-1.0) (default: 0.5)"
    )
    parser.add_argument(
        "--channels", "-c",
        type=int,
        default=2,
        help="Number of audio channels (default: 2)"
    )
    
    args = parser.parse_args()
    
    generate_sine_wave(
        duration=args.duration,
        sample_rate=args.sample_rate,
        frequency=args.frequency,
        amplitude=args.amplitude,
        channels=args.channels
    )


if __name__ == "__main__":
    main()
