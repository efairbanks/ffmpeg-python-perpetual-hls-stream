#!/usr/bin/env python
"""Generate test audio data for testing the audio processor.

This script generates synthetic PCM audio data (sine wave) and writes it to stdout,
which can be piped into the audio processor for testing. Audio is generated in real time
with precise timing control and segment handling.
"""

import argparse
import math
import struct
import sys
import time
from typing import Optional


def generate_sine_wave_segment(
    start_time: float,
    duration: float,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    channels: int = 2,
) -> None:
    """Generate a segment of sine wave and write raw PCM data to stdout.

    Args:
        start_time: Start time of this segment in seconds (used for phase continuity)
        duration: Duration of the segment in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of the sine wave in Hz
        amplitude: Amplitude of the sine wave (0.0-1.0)
        channels: Number of audio channels
    """
    # Calculate the number of samples
    num_samples = int(duration * sample_rate)
    
    # Generate the sine wave
    for i in range(num_samples):
        # Calculate the absolute time for this sample (for phase continuity)
        t = start_time + (i / sample_rate)
        value = int(32767 * amplitude * math.sin(2 * math.pi * frequency * t))
        
        # Convert to 16-bit PCM and write to stdout
        for _ in range(channels):
            sys.stdout.buffer.write(struct.pack("<h", value))
    
    # Ensure the output is flushed immediately
    sys.stdout.buffer.flush()


def generate_real_time_audio(
    total_duration: Optional[float] = None,
    target_segment_length: float = 1.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    channels: int = 2,
) -> None:
    """Generate audio in real time with precise timing.

    Args:
        total_duration: Total duration to generate in seconds (None for infinite)
        target_segment_length: Length of each segment in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of the sine wave in Hz
        amplitude: Amplitude of the sine wave (0.0-1.0)
        channels: Number of audio channels
    """
    # Calculate bytes per second (for timing)
    bytes_per_sample = 2 * channels  # 2 bytes per sample (16-bit) * number of channels
    bytes_per_second = sample_rate * bytes_per_sample
    
    # Track timing
    start_time = time.time()
    elapsed_audio_time = 0.0
    segment_count = 0
    
    try:
        while True:
            # Check if we've reached the total duration
            if total_duration is not None and elapsed_audio_time >= total_duration:
                break
            
            # Calculate the segment duration (might be shorter for the last segment)
            if total_duration is not None:
                segment_duration = min(target_segment_length, total_duration - elapsed_audio_time)
                if segment_duration <= 0:
                    break
            else:
                segment_duration = target_segment_length
            
            # Calculate when this segment should end in real time
            segment_start_real = time.time()
            segment_end_real = start_time + elapsed_audio_time + segment_duration
            
            # Generate the audio segment
            generate_sine_wave_segment(
                start_time=elapsed_audio_time,
                duration=segment_duration,
                sample_rate=sample_rate,
                frequency=frequency,
                amplitude=amplitude,
                channels=channels
            )
            
            # Update elapsed audio time
            elapsed_audio_time += segment_duration
            segment_count += 1
            
            # Calculate how long to wait before the next segment
            # This compensates for any processing time and jitter
            now = time.time()
            wait_time = segment_end_real - now
            
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                # We're falling behind, log a warning
                sys.stderr.write(f"Warning: Audio generation falling behind by {-wait_time:.4f} seconds\n")
                sys.stderr.flush()
    
    except KeyboardInterrupt:
        sys.stderr.write(f"\nAudio generation stopped after {elapsed_audio_time:.2f} seconds\n")
        sys.stderr.flush()


def main() -> None:
    """Parse command line arguments and generate audio data."""
    parser = argparse.ArgumentParser(
        description="Generate test audio data (sine wave) in real time and write to stdout."
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Total duration of the audio in seconds (default: infinite)"
    )
    parser.add_argument(
        "--target-segment-length", "-t",
        type=float,
        default=1.0,
        help="Length of each audio segment in seconds (default: 1.0)"
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
    
    # Print information about the audio generation
    if args.duration is None:
        sys.stderr.write(f"Generating infinite audio in {args.target_segment_length}s segments...\n")
    else:
        sys.stderr.write(f"Generating {args.duration}s of audio in {args.target_segment_length}s segments...\n")
    sys.stderr.write(f"Sample rate: {args.sample_rate}Hz, Frequency: {args.frequency}Hz, Channels: {args.channels}\n")
    sys.stderr.flush()
    
    generate_real_time_audio(
        total_duration=args.duration,
        target_segment_length=args.target_segment_length,
        sample_rate=args.sample_rate,
        frequency=args.frequency,
        amplitude=args.amplitude,
        channels=args.channels
    )


if __name__ == "__main__":
    main()
