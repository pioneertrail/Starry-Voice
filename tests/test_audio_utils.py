"""
Tests for the audio utilities module.
"""
import os
import pytest
from unittest.mock import patch, mock_open
from voice_chat.audio_utils import (
    TemporaryAudioFile,
    AudioFilePool,
    process_audio_in_chunks
)

def test_temporary_audio_file_creation():
    """Test creating a temporary audio file with custom prefix and suffix."""
    with TemporaryAudioFile(prefix="test_", suffix=".mp3") as audio_path:
        # Verify file was created with correct name
        assert os.path.exists(audio_path)
        assert audio_path.endswith(".mp3")
        assert "test_" in audio_path
        
        # Write some test data
        with open(audio_path, 'wb') as f:
            f.write(b'test data')
            
        # Verify data was written
        with open(audio_path, 'rb') as f:
            assert f.read() == b'test data'
    
    # Verify file was cleaned up
    assert not os.path.exists(audio_path)

def test_temporary_audio_file_cleanup_error():
    """Test graceful handling of cleanup errors."""
    with patch('os.unlink') as mock_unlink:
        mock_unlink.side_effect = OSError("Permission denied")
        
        with TemporaryAudioFile() as audio_path:
            # Write some test data
            with open(audio_path, 'wb') as f:
                f.write(b'test data')
        
        # Verify cleanup error was handled gracefully
        assert mock_unlink.called

def test_audio_file_pool_reuse():
    """Test reusing files from the pool."""
    with AudioFilePool(max_files=2) as pool:
        # Get first file
        file1 = pool.get_file()
        assert isinstance(file1, TemporaryAudioFile)
        path1 = file1.path
        
        # Get second file
        file2 = pool.get_file()
        assert isinstance(file2, TemporaryAudioFile)
        path2 = file2.path
        
        # Get third file
        file3 = pool.get_file()
        assert isinstance(file3, TemporaryAudioFile)
        path3 = file3.path
        
        # Verify files are different
        assert path1 != path2
        assert path2 != path3

def test_audio_file_pool_cleanup():
    """Test cleanup of excess files in the pool."""
    with patch('os.unlink') as mock_unlink:
        with AudioFilePool(max_files=1) as pool:
            # Get first file
            file1 = pool.get_file()
            assert isinstance(file1, TemporaryAudioFile)
            path1 = file1.path
            
            # Get second file (should trigger cleanup)
            file2 = pool.get_file()
            assert isinstance(file2, TemporaryAudioFile)
            path2 = file2.path
            
            # Get third file (should trigger cleanup)
            file3 = pool.get_file()
            assert isinstance(file3, TemporaryAudioFile)
            path3 = file3.path
            
            # Verify cleanup was attempted
            mock_unlink.assert_any_call(path1)

def test_audio_file_pool_cleanup_error():
    """Test graceful handling of cleanup errors in the pool."""
    with patch('os.unlink') as mock_unlink:
        mock_unlink.side_effect = OSError("Permission denied")
        
        with AudioFilePool(max_files=1) as pool:
            # Get first file
            file1 = pool.get_file()
            assert isinstance(file1, TemporaryAudioFile)
            path1 = file1.path
            
            # Get second file (should attempt cleanup)
            file2 = pool.get_file()
            assert isinstance(file2, TemporaryAudioFile)
            path2 = file2.path
            
            # Get third file (should attempt cleanup)
            file3 = pool.get_file()
            assert isinstance(file3, TemporaryAudioFile)
            path3 = file3.path
            
            # Verify cleanup error was handled gracefully
            mock_unlink.assert_any_call(path1)

def test_process_audio_in_chunks():
    """Test processing audio file in chunks."""
    test_data = b'x' * (1024 * 1024 * 2)  # 2MB of test data
    
    with patch('builtins.open', mock_open(read_data=test_data)):
        chunks = list(process_audio_in_chunks('test.wav', chunk_size=1024*1024))
        assert len(chunks) == 2
        assert len(chunks[0]) == 1024 * 1024
        assert len(chunks[1]) == 1024 * 1024

def test_process_audio_in_chunks_empty_file():
    """Test processing an empty audio file."""
    with patch('builtins.open', mock_open(read_data=b'')):
        chunks = list(process_audio_in_chunks('empty.wav'))
        assert len(chunks) == 0

def test_process_audio_in_chunks_file_error():
    """Test handling of file errors during chunk processing."""
    with patch('builtins.open') as mock_open:
        mock_open.side_effect = IOError("File not found")
        
        with pytest.raises(IOError):
            list(process_audio_in_chunks('nonexistent.wav')) 