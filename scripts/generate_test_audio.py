#!/usr/bin/env python
"""Generate test audio data for testing the audio processor.

This script generates synthetic PCM audio data (sine wave) and writes it to stdout,
which can be piped into the audio processor for testing. Audio is generated in real time
with precise timing control and segment handling.

It can also load WAV files and split them into equally sized buffers for processing.
"""

import argparse
import math
import os
from pathlib import Path
import random
import struct
import sys
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.io import wavfile


class WavSampleLoader:
    """Load a WAV file and split it into equally sized buffers.
    
    This class loads a WAV file, converts it to mono if necessary,
    and splits it into N equally sized buffers ordered in time.
    
    Attributes:
        sample_rate (int): Sample rate of the loaded WAV file
        num_buffers (int): Number of buffers the audio is split into
        buffer_size (int): Size of each buffer in samples
        buffers (List[np.ndarray]): List of audio buffers
        current_buffer (int): Index of the current buffer being processed
        current_sample_index (int): Index of the current sample within the current buffer
    """
    
    def __init__(self, wav_file_path: Union[str, Path], num_buffers: int) -> None:
        """Initialize the WavSampleLoader.
        
        Args:
            wav_file_path: Path to the WAV file to load
            num_buffers: Number of equally sized buffers to split the audio into
        
        Raises:
            FileNotFoundError: If the WAV file does not exist
            ValueError: If the WAV file cannot be loaded or processed
        """
        wav_file_path = Path(wav_file_path)
        if not wav_file_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file_path}")
        
        # Load the WAV file
        try:
            self.sample_rate, audio_data = wavfile.read(wav_file_path)
        except Exception as e:
            raise ValueError(f"Failed to load WAV file: {e}")
        
        # Convert to mono if stereo or multi-channel
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # Average all channels to create mono
            audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)
        
        # Ensure audio_data is 1D
        audio_data = audio_data.flatten()
        
        # Calculate buffer size (rounded down)
        total_samples = len(audio_data)
        self.num_buffers = num_buffers
        self.buffer_size = total_samples // num_buffers
        
        if self.buffer_size == 0:
            raise ValueError(f"Too many buffers ({num_buffers}) for audio length ({total_samples} samples)")
        
        # Split the audio into equally sized buffers
        self.buffers: List[np.ndarray] = []
        for i in range(num_buffers):
            start_idx = i * self.buffer_size
            end_idx = start_idx + self.buffer_size
            # Handle the last buffer which might be smaller
            if i == num_buffers - 1:
                end_idx = total_samples
            
            buffer = audio_data[start_idx:end_idx]
            self.buffers.append(buffer)
        
        # Initialize buffer pointer and sample index
        self.current_buffer = 0
        self.current_sample_index = 0
        
        print(f"Loaded WAV file: {wav_file_path}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Total samples: {total_samples}")
        print(f"  Split into {num_buffers} buffers of ~{self.buffer_size} samples each")
    
    def get_next_buffer(self) -> Tuple[np.ndarray, bool]:
        """Get the next buffer of audio samples.
        
        Returns:
            Tuple containing:
                - np.ndarray: Buffer of audio samples
                - bool: True if this is the last buffer, False otherwise
        """
        if self.current_buffer >= self.num_buffers:
            # Return empty buffer and True to indicate end of audio
            return np.array([], dtype=np.int16), True
        
        buffer = self.buffers[self.current_buffer]
        is_last = self.current_buffer == self.num_buffers - 1
        self.current_buffer += 1
        
        return buffer, is_last
    
    def reset(self) -> None:
        """Reset the buffer pointer to the beginning."""
        self.current_buffer = 0
        self.current_sample_index = 0
    
    def get_buffer_duration(self) -> float:
        """Get the duration of a single buffer in seconds.
        
        Returns:
            float: Duration of a buffer in seconds
        """
        return self.buffer_size / self.sample_rate
    
    def get_total_duration(self) -> float:
        """Get the total duration of the audio in seconds.
        
        Returns:
            float: Total duration in seconds
        """
        total_samples = sum(len(buffer) for buffer in self.buffers)
        return total_samples / self.sample_rate
    
    def get_next_sample(self) -> int:
        """Get the next sample from the current buffer.
        
        When the end of the current buffer is reached, a new buffer is randomly
        selected from the available buffers, and samples are retrieved from it.
        
        Returns:
            int: The next audio sample value
        """
        # If we've reached the end of the current buffer, select a new one randomly
        if self.current_sample_index >= len(self.buffers[self.current_buffer]):
            # Choose a random buffer (excluding the current one if possible)
            if self.num_buffers > 1:
                available_buffers = list(range(self.num_buffers))
                available_buffers.remove(self.current_buffer)
                self.current_buffer = np.random.choice(available_buffers)
            else:
                # If there's only one buffer, just reset the index
                self.current_buffer = 0
            
            # Reset the sample index
            self.current_sample_index = 0
        
        # Get the sample from the current buffer
        sample = self.buffers[self.current_buffer][self.current_sample_index]
        
        # Increment the sample index
        self.current_sample_index += 1
        
        return int(sample)



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


def generate_wav_file_audio(
    wav_file_path: Union[str, Path],
    target_segment_length: float = 1.0,
    loop: bool = False,
) -> None:
    """Generate real-time audio from a WAV file with precise timing.
    
    Args:
        wav_file_path: Path to the WAV file to load
        target_segment_length: Length of each segment in seconds
        loop: Whether to loop the audio when it reaches the end
    """
    # Estimate number of buffers based on target segment length
    try:
        sample_rate, audio_data = wavfile.read(wav_file_path)
        total_duration = len(audio_data) / sample_rate
        num_buffers = max(1, int(total_duration / target_segment_length))
        
        # Load the WAV file and split it into buffers
        loader = WavSampleLoader(wav_file_path, num_buffers)
        
        # Track timing
        start_time = time.time()
        elapsed_audio_time = 0.0
        segment_count = 0
        
        # Calculate bytes per sample (assuming 16-bit audio)
        bytes_per_sample = 2  # 16-bit = 2 bytes
        
        try:
            while True:
                # Get the next buffer
                buffer, is_last = loader.get_next_buffer()
                
                if len(buffer) == 0:
                    if loop:
                        # Reset to beginning if looping
                        loader.reset()
                        buffer, is_last = loader.get_next_buffer()
                    else:
                        # End of audio
                        break
                
                # Calculate segment duration
                segment_duration = len(buffer) / loader.sample_rate
                
                # Calculate when this segment should end in real time
                segment_start_real = time.time()
                segment_end_real = start_time + elapsed_audio_time + segment_duration
                
                # Convert buffer to bytes and write to stdout
                for sample in buffer:
                    sys.stdout.buffer.write(struct.pack("<h", int(sample)))
                sys.stdout.buffer.flush()
                
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
                
                # If we've reached the end and not looping, break
                if is_last and not loop:
                    break
        
        except KeyboardInterrupt:
            sys.stderr.write(f"\nAudio generation stopped after {elapsed_audio_time:.2f} seconds\n")
            sys.stderr.flush()
            
    except Exception as e:
        sys.stderr.write(f"Error generating audio from WAV file: {e}\n")
        sys.stderr.flush()
        sys.exit(1)


# Initialize the WavSampleLoader with the amen break sample
AMEN_BREAK_PATH = os.path.join(os.path.dirname(__file__), "cw_amen13_173.wav")
amen_loader = None


def generate_amen_break_segment(
    start_time: float,
    duration: float,
    sample_rate: int = 44100,
    frequency: float = 440.0,  # Not used but kept for API compatibility
    amplitude: float = 1.0,
    channels: int = 2,
) -> None:
    """Generate a segment of audio from the Amen break sample with random buffer selection.
    
    Args:
        start_time: Start time of this segment in seconds (used for timing)
        duration: Duration of the segment in seconds
        sample_rate: Sample rate in Hz (not used, uses the WAV file's sample rate)
        frequency: Not used, kept for API compatibility
        amplitude: Amplitude scaling factor (default: 1.0)
        channels: Number of output channels (default: 2)
    """
    global amen_loader
    
    # Initialize the loader if it hasn't been initialized yet
    if amen_loader is None:
        try:
            # Split into approximately 1-second buffers
            sample_rate, audio_data = wavfile.read(AMEN_BREAK_PATH)
            total_duration = len(audio_data) / sample_rate
            num_buffers = 8
            amen_loader = WavSampleLoader(AMEN_BREAK_PATH, num_buffers)
            sys.stderr.write(f"Loaded Amen break sample: {AMEN_BREAK_PATH}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Error loading Amen break sample: {e}\n")
            sys.stderr.flush()
            sys.exit(1)
    
    # Calculate the number of samples to generate
    num_samples = int(duration * amen_loader.sample_rate)
    
    # Generate the audio by retrieving samples one by one
    for _ in range(num_samples):
        # Get the next sample with random buffer selection
        value = amen_loader.get_next_sample()
        
        # Apply amplitude scaling
        value = int(value * amplitude)
        
        # Convert to 16-bit PCM and write to stdout for each channel
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
    use_amen_break: bool = False,
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
            if use_amen_break:
                generate_amen_break_segment(
                    start_time=elapsed_audio_time,
                    duration=segment_duration,
                    sample_rate=sample_rate,
                    frequency=frequency,
                    amplitude=amplitude,
                    channels=channels
                )
            else:
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
        description="Generate test audio data in real time and write to stdout."
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
    parser.add_argument(
        "--wav-file", "-w",
        type=str,
        default=None,
        help="Path to a WAV file to use as the audio source"
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="Loop the WAV file when it reaches the end"
    )
    parser.add_argument(
        "--use-amen-break",
        action="store_true",
        help="Use the Amen break sample with random buffer selection"
    )
    
    args = parser.parse_args()
    
    # Determine which generation method to use
    if args.use_amen_break:
        sys.stderr.write("Generating audio using Amen break sample with random buffer selection\n")
        sys.stderr.write(f"Target segment length: {args.target_segment_length}s\n")
        sys.stderr.flush()
        
        generate_real_time_audio(
            total_duration=args.duration,
            target_segment_length=args.target_segment_length,
            sample_rate=args.sample_rate,
            amplitude=args.amplitude,
            channels=args.channels,
            use_amen_break=True
        )
    elif args.wav_file:
        sys.stderr.write(f"Generating audio from WAV file: {args.wav_file}\n")
        sys.stderr.write(f"Target segment length: {args.target_segment_length}s, Loop: {args.loop}\n")
        sys.stderr.flush()
        
        generate_wav_file_audio(
            wav_file_path=args.wav_file,
            target_segment_length=args.target_segment_length,
            loop=args.loop
        )
    else:
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
