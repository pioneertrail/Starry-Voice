"""
Test configuration and shared fixtures.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Test data
SAMPLE_RECORDING = np.array([0.1, 0.2, 0.3], dtype=np.float32)
SAMPLE_AUDIO_FILE = "test.wav"
SAMPLE_TRANSCRIPTION = "Hello, this is a test message"
SAMPLE_AI_RESPONSE = "I understand your message"

@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for all tests."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
        yield

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch('voice_chat.OpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice."""
    with patch('voice_chat.sd') as mock:
        mock.rec.return_value = SAMPLE_RECORDING
        mock.wait.return_value = None
        yield mock

@pytest.fixture
def mock_pyttsx3():
    """Mock pyttsx3."""
    with patch('voice_chat.pyttsx3') as mock:
        engine = MagicMock()
        mock.init.return_value = engine
        yield engine

@pytest.fixture
def mock_dotenv():
    """Mock python-dotenv."""
    with patch('voice_chat.load_dotenv') as mock:
        yield mock 