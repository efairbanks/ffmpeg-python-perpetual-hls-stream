"""Tests for the main module."""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from audio_processor.main import process_audio


@pytest.fixture
def mock_ffmpeg():
    """Mock ffmpeg module for testing."""
    with patch("audio_processor.main.ffmpeg") as mock_ffmpeg:
        # Create a mock process with stdin
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        
        # Setup the run_async method to return our mock process
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run_async.return_value = mock_process
        
        yield mock_ffmpeg, mock_process


def test_process_audio_with_valid_input(mock_ffmpeg, tmp_path):
    """Test processing audio with valid input data."""
    mock_ffmpeg_module, mock_process = mock_ffmpeg
    
    # Create a temporary output file
    output_file = str(tmp_path / "output.wav")
    
    # Mock stdin with some test data
    test_data = b"\x00\x01\x02\x03" * 1024
    mock_stdin = io.BytesIO(test_data)
    
    with patch("sys.stdin.buffer", mock_stdin):
        process_audio(output_file)
    
    # Verify ffmpeg was called with correct parameters
    mock_ffmpeg_module.input.assert_called_once()
    args, kwargs = mock_ffmpeg_module.input.call_args
    assert args[0] == "pipe:0"
    assert kwargs["format"] == "s16le"
    assert kwargs["acodec"] == "pcm_s16le"
    assert kwargs["ar"] == "44100"
    assert kwargs["ac"] == "2"
    
    # Verify output parameters
    mock_ffmpeg_module.input.return_value.output.assert_called_once_with(
        output_file, format="wav"
    )
    
    # Verify data was written to stdin
    mock_process.stdin.write.assert_called()
    
    # Verify process was closed properly
    mock_process.stdin.close.assert_called_once()
    mock_process.wait.assert_called_once()


@pytest.fixture
def mock_ffmpeg_error():
    """Mock ffmpeg module that raises an error."""
    with patch("audio_processor.main.ffmpeg") as mock_ffmpeg:
        # Create a mock error
        error = mock_ffmpeg.Error()
        error.stderr = b"Mock FFmpeg error"
        
        # Setup the run_async method to raise our mock error
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run_async.side_effect = error
        
        yield mock_ffmpeg


def test_process_audio_with_ffmpeg_error(mock_ffmpeg_error, tmp_path, capsys):
    """Test handling of FFmpeg errors."""
    # Create a temporary output file
    output_file = str(tmp_path / "output.wav")
    
    # Mock stdin with some test data
    test_data = b"\x00\x01\x02\x03" * 1024
    mock_stdin = io.BytesIO(test_data)
    
    with patch("sys.stdin.buffer", mock_stdin), pytest.raises(SystemExit) as excinfo:
        process_audio(output_file)
    
    # Verify exit code
    assert excinfo.value.code == 1
    
    # Verify error message
    captured = capsys.readouterr()
    assert "FFmpeg error: Mock FFmpeg error" in captured.err
