"""
Audio utilities for managing temporary files and audio processing.
"""
import os
import tempfile
import logging
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class TemporaryAudioFile:
    """Context manager for handling temporary audio files."""
    
    def __init__(self, suffix='.wav'):
        """Initialize the temporary audio file."""
        self.suffix = suffix
        self.path = None
        self._temp_file = None
        self._is_entered = False

    def __enter__(self):
        """Create a temporary file when entering the context."""
        try:
            # Create a temporary file with the specified suffix
            self._temp_file = tempfile.NamedTemporaryFile(
                suffix=self.suffix,
                delete=False
            )
            self.path = self._temp_file.name
            self._is_entered = True
            logger.debug(f"Created temporary audio file: {self.path}")
            return self
        except Exception as e:
            logger.error(f"Error creating temporary file: {str(e)}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the temporary file when exiting the context."""
        try:
            if self._temp_file:
                self._temp_file.close()
                if os.path.exists(self.path):
                    os.unlink(self.path)
                    logger.debug(f"Cleaned up temporary file: {self.path}")
            self._is_entered = False
        except Exception as e:
            logger.error(f"Error cleaning up temporary file: {str(e)}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        if self._is_entered:
            self.__exit__(None, None, None)

class AudioFilePool:
    """Manages a pool of temporary audio files."""
    
    def __init__(self, max_files=10):
        """Initialize the audio file pool."""
        self.max_files = max_files
        self.files = []
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_chat_audio"
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized audio file pool in {self.temp_dir}")

    def create_file(self, suffix='.wav'):
        """Create a new temporary audio file."""
        try:
            if len(self.files) >= self.max_files:
                self.cleanup_oldest()
            
            temp_file = TemporaryAudioFile(suffix)
            self.files.append(temp_file)
            logger.debug(f"Created new audio file in pool. Total files: {len(self.files)}")
            return temp_file
        except Exception as e:
            logger.error(f"Error creating file in pool: {str(e)}")
            raise

    def cleanup_oldest(self):
        """Remove the oldest file from the pool."""
        try:
            if self.files:
                oldest_file = self.files.pop(0)
                if os.path.exists(oldest_file.path):
                    os.unlink(oldest_file.path)
                    logger.debug(f"Cleaned up oldest file: {oldest_file.path}")
        except Exception as e:
            logger.error(f"Error cleaning up oldest file: {str(e)}")

    def cleanup(self):
        """Clean up all files in the pool."""
        try:
            for file in self.files:
                if os.path.exists(file.path):
                    os.unlink(file.path)
                    logger.debug(f"Cleaned up file: {file.path}")
            self.files.clear()
            logger.info("Cleaned up all files in audio pool")
        except Exception as e:
            logger.error(f"Error cleaning up audio pool: {str(e)}")

def process_audio_in_chunks(audio_data, chunk_size=16000):
    """Process audio data in chunks for better memory management."""
    try:
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            yield chunk
    except Exception as e:
        logger.error(f"Error processing audio chunks: {str(e)}")
        raise 