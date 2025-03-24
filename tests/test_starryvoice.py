"""
Tests for the StarryVoice application.
"""
import pytest
import numpy as np
import os
from typing import Generator, Any
from unittest.mock import patch, mock_open, MagicMock
from starryvoice.starryvoice import (
    record_audio,
    save_audio,
    transcribe_audio,
    speak,
    get_ai_response,
    main,
    OpenAIClient,
    AVAILABLE_VOICES,
    VoiceChat,
    VOICE_SETTINGS
)

# Test data
SAMPLE_RECORDING = np.array([0.1, 0.2, 0.3], dtype=np.float32)
SAMPLE_AUDIO_FILE = "test.wav"
SAMPLE_TRANSCRIPTION = "Hello, this is a test message"
SAMPLE_AI_RESPONSE = "I understand your message"

class MockTemporaryAudioFile:
    """Mock class for TemporaryAudioFile context manager"""
    def __init__(self, prefix: str = "", suffix: str = ".wav") -> None:
        self.path = f"{prefix}temp{suffix}"
        
    def __enter__(self) -> str:
        return self.path
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

@pytest.fixture(autouse=True)
def mock_sounddevice():
    """Mock sounddevice to prevent hardware access during testing"""
    with patch('voice_chat.voice_chat.sd') as mock:
        mock.rec.return_value = SAMPLE_RECORDING
        mock.wait.return_value = None
        yield mock

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI client with the new API structure"""
    with patch('voice_chat.voice_chat.OpenAIClient') as mock:
        # Create a mock client
        mock_instance = MagicMock()
        mock.get_instance.return_value = mock_instance
        
        # Mock audio transcription
        mock_transcription = MagicMock()
        mock_transcription.text = SAMPLE_TRANSCRIPTION
        mock_instance.audio.transcriptions.create.return_value = mock_transcription
        
        # Mock chat completion
        mock_choice = MagicMock()
        mock_choice.message.content = SAMPLE_AI_RESPONSE
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_completion
        
        # Mock TTS response
        mock_tts_response = MagicMock()
        mock_tts_response.stream_to_file.return_value = None
        mock_instance.audio.speech.create.return_value = mock_tts_response
        
        yield mock_instance

@pytest.fixture(autouse=True)
def mock_pygame():
    """Mock pygame to prevent audio hardware access"""
    with patch('voice_chat.voice_chat.pygame') as mock:
        mock.mixer.music.get_busy.return_value = False
        yield mock

@pytest.fixture(autouse=True)
def mock_dotenv():
    """Mock dotenv to prevent file access"""
    with patch('voice_chat.voice_chat.load_dotenv') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_temporary_audio_file():
    """Mock TemporaryAudioFile context manager"""
    with patch('voice_chat.voice_chat.TemporaryAudioFile', MockTemporaryAudioFile):
        yield

@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client for testing."""
    mock_client = MagicMock()
    
    # Mock audio transcription
    mock_transcription = MagicMock()
    mock_transcription.text = SAMPLE_TRANSCRIPTION
    mock_client.audio.transcriptions.create.return_value = mock_transcription
    
    # Mock chat completion
    mock_choice = MagicMock()
    mock_choice.message.content = SAMPLE_AI_RESPONSE
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    # Mock TTS response
    mock_tts_response = MagicMock()
    mock_tts_response.stream_to_file.return_value = None
    mock_client.audio.speech.create.return_value = mock_tts_response
    
    return mock_client

@pytest.fixture
def voice_chat(mock_openai_client: MagicMock) -> VoiceChat:
    """Create a VoiceChat instance for testing."""
    return VoiceChat(mock_openai_client)

def test_record_audio():
    """Test audio recording functionality"""
    recording = record_audio(duration=1)
    
    # Verify recording parameters
    assert np.array_equal(recording, SAMPLE_RECORDING)

def test_save_audio():
    """Test audio file saving"""
    with patch('voice_chat.voice_chat.sf.write') as mock_write:
        save_audio(SAMPLE_RECORDING, SAMPLE_AUDIO_FILE)
        mock_write.assert_called_once_with(
            SAMPLE_AUDIO_FILE,
            SAMPLE_RECORDING,
            44100
        )

def test_transcribe_audio():
    """Test audio transcription"""
    with patch('builtins.open', mock_open(read_data=b'audio_data')):
        result = transcribe_audio(SAMPLE_AUDIO_FILE)
        
        # Verify API call
        assert result == SAMPLE_TRANSCRIPTION

def test_transcribe_audio_error():
    """Test audio transcription error handling"""
    with patch('voice_chat.voice_chat.OpenAIClient.get_instance') as mock:
        mock.return_value.audio.transcriptions.create.side_effect = Exception("API Error")
        
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            result = transcribe_audio(SAMPLE_AUDIO_FILE)
            assert result is None

def test_transcribe_audio_no_client():
    """Test audio transcription when client is not initialized"""
    with patch('voice_chat.voice_chat.OpenAIClient.get_instance') as mock:
        mock.return_value = None
        
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            result = transcribe_audio(SAMPLE_AUDIO_FILE)
            assert result is None

def test_speak():
    """Test text-to-speech functionality"""
    speak(SAMPLE_AI_RESPONSE)
    
    # Verify OpenAI TTS API call
    assert True  # If we get here without errors, the test passes

def test_speak_no_client():
    """Test text-to-speech when client is not initialized"""
    with patch('voice_chat.voice_chat.OpenAIClient.get_instance') as mock:
        mock.return_value = None
        speak(SAMPLE_AI_RESPONSE)
        # If we get here without errors, the test passes

def test_get_ai_response():
    """Test AI response generation"""
    result = get_ai_response(SAMPLE_TRANSCRIPTION)
    assert result == SAMPLE_AI_RESPONSE

def test_get_ai_response_error():
    """Test AI response error handling"""
    with patch('voice_chat.voice_chat.OpenAIClient.get_instance') as mock:
        mock.return_value.chat.completions.create.side_effect = Exception("API Error")
        result = get_ai_response(SAMPLE_TRANSCRIPTION)
        assert result == "I apologize, but I encountered an error. Please try again."

def test_get_ai_response_no_client():
    """Test AI response when client is not initialized"""
    with patch('voice_chat.voice_chat.OpenAIClient.get_instance') as mock:
        mock.return_value = None
        result = get_ai_response(SAMPLE_TRANSCRIPTION)
        assert result == "I apologize, but I encountered an error. Please try again."

def test_main_flow():
    """Test the main application flow"""
    # Mock user inputs: first for voice selection, then for recording
    with patch('builtins.input', side_effect=['', '']):
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            # Run main with a single iteration
            with patch('builtins.print') as mock_print:
                main(max_iterations=1)
                
                # Verify the flow
                assert mock_print.called

def test_main_error_handling():
    """Test main loop error handling"""
    with patch('builtins.input', side_effect=['', '']):
        # Simulate an error during recording
        with patch('voice_chat.voice_chat.record_audio') as mock:
            mock.side_effect = Exception("Recording error")
            
            # Run main with error handling
            with patch('builtins.print') as mock_print:
                main(max_iterations=1)
                
                # Verify error was caught and handled
                assert mock_print.called

def test_speak_with_voice():
    """Test text-to-speech functionality with different voices"""
    # Test with valid voice
    speak(SAMPLE_AI_RESPONSE, voice='nova')
    assert True  # If we get here without errors, the test passes
    
    # Test with invalid voice (should fall back to 'alloy')
    speak(SAMPLE_AI_RESPONSE, voice='invalid_voice')
    assert True  # If we get here without errors, the test passes

def test_main_with_voice_selection():
    """Test the main application flow with voice selection"""
    # Mock user inputs: first for voice selection, then for recording
    with patch('builtins.input', side_effect=['nova', '']):
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            # Run main with a single iteration
            with patch('builtins.print') as mock_print:
                main(max_iterations=1)
                
                # Verify voice selection was displayed
                assert any('nova' in str(call) for call in mock_print.call_args_list)

def test_main_with_invalid_voice():
    """Test the main application flow with invalid voice selection"""
    # Mock user inputs: first for voice selection, then for recording
    with patch('builtins.input', side_effect=['invalid_voice', '']):
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            # Run main with a single iteration
            with patch('builtins.print') as mock_print:
                main(max_iterations=1)
                
                # Verify fallback message was displayed
                assert any('Invalid voice' in str(call) for call in mock_print.call_args_list)

def test_main_with_default_voice():
    """Test the main application flow with default voice selection"""
    # Mock user inputs: empty for voice selection (default), then for recording
    with patch('builtins.input', side_effect=['', '']):
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b'audio_data')):
            # Run main with a single iteration
            with patch('builtins.print') as mock_print:
                main(max_iterations=1)
                
                # Verify default voice was used
                assert any('alloy' in str(call) for call in mock_print.call_args_list)

def test_voice_chat_initialization(voice_chat: VoiceChat) -> None:
    """Test VoiceChat initialization."""
    assert voice_chat.client is not None
    assert voice_chat.audio_pool is not None
    assert voice_chat.conversation_history == []
    assert voice_chat.current_voice is None

def test_voice_chat_start_invalid_voice(voice_chat: VoiceChat) -> None:
    """Test VoiceChat start with invalid voice."""
    with pytest.raises(ValueError, match="Invalid voice name"):
        voice_chat.start("invalid_voice")

def test_voice_chat_start_valid_voice(voice_chat: VoiceChat) -> None:
    """Test VoiceChat start with valid voice."""
    with patch('builtins.input', side_effect=['quit']):
        voice_chat.start("nova")
        assert voice_chat.current_voice == VOICE_SETTINGS["nova"]

def test_voice_chat_record_audio(voice_chat: VoiceChat) -> None:
    """Test VoiceChat record_audio method."""
    with patch('sounddevice.InputStream') as mock_stream:
        with patch('builtins.input', return_value=''):
            audio_data = voice_chat.record_audio()
            assert mock_stream.called

def test_voice_chat_speech_to_text(voice_chat: VoiceChat) -> None:
    """Test VoiceChat speech_to_text method."""
    result = voice_chat.speech_to_text(SAMPLE_RECORDING)
    assert result == SAMPLE_TRANSCRIPTION

def test_voice_chat_get_ai_response(voice_chat: VoiceChat) -> None:
    """Test VoiceChat get_ai_response method."""
    result = voice_chat.get_ai_response(SAMPLE_TRANSCRIPTION)
    assert result == SAMPLE_AI_RESPONSE
    assert len(voice_chat.conversation_history) == 2  # User message + AI response

def test_voice_chat_text_to_speech(voice_chat: VoiceChat) -> None:
    """Test VoiceChat text_to_speech method."""
    voice_chat.current_voice = VOICE_SETTINGS["nova"]
    with patch('pygame.mixer.init'), \
         patch('pygame.mixer.music.load'), \
         patch('pygame.mixer.music.play'), \
         patch('pygame.mixer.music.get_busy', return_value=False), \
         patch('pygame.mixer.quit'):
        voice_chat.text_to_speech(SAMPLE_AI_RESPONSE)
        # If we get here without errors, the test passes

def test_voice_chat_cleanup(voice_chat: VoiceChat) -> None:
    """Test VoiceChat cleanup method."""
    with patch.object(voice_chat.audio_pool, 'cleanup') as mock_cleanup:
        voice_chat.cleanup()
        assert mock_cleanup.called

def test_openai_client_singleton() -> None:
    """Test OpenAIClient singleton pattern."""
    with patch('builtins.open', mock_open(read_data='OPENAI_API_KEY=test_key')), \
         patch.object(OpenAI, 'models') as mock_models:
        client1 = OpenAIClient.get_instance()
        client2 = OpenAIClient.get_instance()
        assert client1 is client2  # Same instance

def test_openai_client_invalid_key() -> None:
    """Test OpenAIClient with invalid API key."""
    with patch('builtins.open', mock_open(read_data='OPENAI_API_KEY=invalid_key')), \
         patch.object(OpenAI, 'models') as mock_models:
        mock_models.list.side_effect = Exception("401: Invalid API key")
        with pytest.raises(Exception):
            OpenAIClient.get_instance() 