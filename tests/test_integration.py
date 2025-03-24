"""
Integration tests for the voice chat application.
"""
import pytest
import logging
from unittest.mock import patch, mock_open, MagicMock
from voice_chat import (
    record_audio,
    save_audio,
    transcribe_audio,
    speak,
    get_ai_response,
    main,
    OpenAIClient,
    AVAILABLE_VOICES
)
from voice_chat.audio_utils import AudioFilePool, TemporaryAudioFile

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

class MockAudioFile:
    def __init__(self, path="test_audio.wav"):
        self.path = path
    
    def __enter__(self):
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockAudioPool:
    def __init__(self, max_files=2):
        self.max_files = max_files
        self.files = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def get_file(self):
        audio_file = MockAudioFile(f"test_audio_{len(self.files)}.wav")
        self.files.append(audio_file)
        return audio_file

def test_audio_pool_integration():
    """Test AudioFilePool integration with main flow"""
    with patch('builtins.input', side_effect=['', '']):  # Voice selection and recording prompt
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            with patch('voice_chat.voice_chat.record_audio') as mock_record:
                mock_record.return_value = MagicMock()  # Simulated audio
                with patch('voice_chat.voice_chat.transcribe_audio', return_value="Hello"):
                    with patch('voice_chat.voice_chat.get_ai_response', return_value="Hi there"):
                        with patch('voice_chat.voice_chat.speak') as mock_speak:
                            with patch('voice_chat.voice_chat.AudioFilePool', return_value=MockAudioPool(max_files=2)):
                                # Patch save_audio to use our pool
                                with patch('voice_chat.voice_chat.save_audio') as mock_save:
                                    mock_save.side_effect = lambda data, path: None
                                    iterations = main(max_iterations=1)
                                    assert iterations == 1
                                    assert mock_save.call_count == 1
                                    assert mock_speak.call_count == 1

def test_audio_pool_with_multiple_recordings():
    """Test AudioFilePool handling multiple recordings in sequence"""
    # One input for voice selection, three for recording prompts
    with patch('builtins.input', side_effect=['', '', '', '']):
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            with patch('voice_chat.voice_chat.record_audio') as mock_record:
                mock_record.return_value = MagicMock()  # Simulated audio
                with patch('voice_chat.voice_chat.transcribe_audio', return_value="Hello"):
                    with patch('voice_chat.voice_chat.get_ai_response', return_value="Hi there"):
                        with patch('voice_chat.voice_chat.speak') as mock_speak:
                            with patch('voice_chat.voice_chat.AudioFilePool', return_value=MockAudioPool(max_files=2)):
                                # Patch save_audio to use our pool
                                with patch('voice_chat.voice_chat.save_audio') as mock_save:
                                    mock_save.side_effect = lambda data, path: None
                                    iterations = main(max_iterations=3)
                                    assert iterations == 3
                                    assert mock_save.call_count == 3
                                    assert mock_speak.call_count == 3 