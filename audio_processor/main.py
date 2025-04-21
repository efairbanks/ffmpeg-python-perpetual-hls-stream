"""Main module for audio processing.

This module provides functionality to read raw audio data from stdin
and continuously write it to a wav file using ffmpeg-python.
"""

import argparse
import sys
from typing import Optional

import ffmpeg


def process_audio(output_file: str, sample_rate: int = 44100, channels: int = 2, target_segment_length: int = 2) -> None:
    """Process raw audio data from stdin and write to a wav file.

    Args:
        output_file: Path to the output wav file
        sample_rate: Sample rate of the audio (default: 44100)
        channels: Number of audio channels (default: 2)
        target_segment_length: Target segment length in seconds for HLS output (default: 2)
    """
    try:
        # Read raw PCM data from stdin
        process = (
            ffmpeg
            .input("pipe:0", format="s16le", acodec="pcm_s16le", 
                   ar=str(sample_rate), ac=str(channels))
            .output(output_file, format="hls", 
                    hls_time=target_segment_length, hls_list_size=5, 
                    hls_flags="delete_segments",
                    hls_delete_threshold=10)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # Read from stdin in chunks and write to ffmpeg process
        chunk_size = 4096
        while True:
            chunk = sys.stdin.buffer.read(chunk_size)
            if not chunk:
                break
            process.stdin.write(chunk)
            process.stdin.flush()

        # Close stdin pipe and wait for process to finish
        process.stdin.close()
        process.wait()
        
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        # Handle broken pipe (e.g., if ffmpeg process terminates unexpectedly)
        print("Broken pipe error. FFmpeg process may have terminated unexpectedly.", 
              file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted by user.", file=sys.stderr)
        sys.exit(0)


def main() -> None:
    """Parse command line arguments and start audio processing."""
    parser = argparse.ArgumentParser(
        description="Process raw audio data from stdin and write to an HLS stream."
    )
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="Path to the output wav file"
    )
    parser.add_argument(
        "--sample-rate", "-r", 
        type=int, 
        default=44100, 
        help="Sample rate of the audio (default: 44100)"
    )
    parser.add_argument(
        "--channels", "-c", 
        type=int, 
        default=2, 
        help="Number of audio channels (default: 2)"
    )
    parser.add_argument(
        "--target-segment-length", "-t", 
        type=int, 
        default=2, 
        help="Target segment length in seconds for HLS output (default: 2)"
    )
    
    args = parser.parse_args()
    
    process_audio(
        output_file=args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
        target_segment_length=args.target_segment_length
    )


if __name__ == "__main__":
    main()
