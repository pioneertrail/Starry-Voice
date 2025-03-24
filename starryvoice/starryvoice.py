"""
Voice chat application using OpenAI's APIs for speech-to-text and text-to-speech.

This module provides a VoiceChat class that enables natural conversation with AI using
OpenAI's latest models for speech recognition, text generation, and speech synthesis.
"""
import os
import sys
import logging
from typing import Optional, List, Dict, Any
import sounddevice as sd
import soundfile as sf
import numpy as np
import pygame
import wave
import json
import pyttsx3
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from .audio_utils import TemporaryAudioFile, AudioFilePool, process_audio_in_chunks
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Vosk model
model_path = Path(__file__).parent / "models" / "vosk-model-small-en-us-0.15"
if not model_path.exists():
    raise ValueError(f"Vosk model not found at {model_path}")
model = Model(str(model_path))
recognizer = KaldiRecognizer(model, 16000)

# Configure voice settings
VOICE_SETTINGS = {
    "nova": {
        "name": "nova",
        "rate": 150,
        "volume": 1.0,
        "pitch": 1.2,
        "description": "Bright and energetic voice"
    },
    "alloy": {
        "name": "alloy",
        "rate": 130,
        "volume": 0.9,
        "pitch": 1.0,
        "description": "Balanced and clear voice"
    },
    "echo": {
        "name": "echo",
        "rate": 140,
        "volume": 1.0,
        "pitch": 0.9,
        "description": "Deep and resonant voice"
    },
    "fable": {
        "name": "fable",
        "rate": 145,
        "volume": 0.95,
        "pitch": 1.1,
        "description": "Warm and engaging voice"
    },
    "onyx": {
        "name": "onyx",
        "rate": 135,
        "volume": 1.0,
        "pitch": 0.95,
        "description": "Smooth and professional voice"
    },
    "shimmer": {
        "name": "shimmer",
        "rate": 155,
        "volume": 0.9,
        "pitch": 1.15,
        "description": "Light and cheerful voice"
    }
}

# Initialize pyttsx3 engine
try:
    TTS_ENGINE = pyttsx3.init()
    # Get available voices and set voice IDs
    voices = TTS_ENGINE.getProperty('voices')
    if len(voices) > 0:
        VOICE_SETTINGS['male'] = {'voice_id': voices[0].id}
    if len(voices) > 1:
        VOICE_SETTINGS['female'] = {'voice_id': voices[1].id}
    logging.info("Text-to-speech engine initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize text-to-speech engine: {e}")
    TTS_ENGINE = None

# Add after the imports
AVAILABLE_VOICES = {
    'default': 'Default system voice',
    'male': 'Male voice',
    'female': 'Female voice'
}

class OpenAIClient:
    """Singleton class for managing OpenAI client.
    
    This class ensures only one instance of the OpenAI client exists and handles
    API key management and initialization.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls) -> Optional[OpenAI]:
        """Get or create the OpenAI client instance.
        
        Returns:
            OpenAI: The initialized OpenAI client instance
            
        Raises:
            ValueError: If the .env file or API key is not found
            Exception: If client initialization fails
        """
        if cls._instance is None:
            try:
                # Load API key directly from .env file
                env_path = Path(__file__).parent.parent / '.env'
                if not env_path.exists():
                    raise ValueError(f".env file not found at {env_path}")
                with open(env_path, 'r') as f:
                    env_contents = f.read().strip()
                    api_key = env_contents.split('=', 1)[1].strip()
                
                cls._instance = OpenAI(api_key=api_key)
                cls._instance.models.list()  # Verify API key works
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                if "401" in str(e):
                    logger.error("Invalid API key. Please check your API key at https://platform.openai.com/account/api-keys")
                elif "403" in str(e):
                    logger.error("Access forbidden. Please check your API key permissions")
                raise
        return cls._instance

def track_resources(func):
    """Decorator to track resource usage of functions."""
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()
        
        # Get initial resource usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final resource usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        # Log resource usage
        logging.debug(
            f"Resource usage for {func.__name__}: "
            f"Memory: {final_memory - initial_memory:.2f}MB, "
            f"CPU: {final_cpu - initial_cpu:.1f}%"
        )
        
        return result
    return wrapper

@track_resources
def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone."""
    logging.info(f"Recording audio for {duration} seconds...")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        logging.info("Audio recording completed")
        return recording
    except Exception as e:
        logging.error(f"Error recording audio: {e}")
        return None

@track_resources
def save_audio(recording, filename, sample_rate=44100):
    """Save recorded audio to WAV file."""
    try:
        sf.write(filename, recording, sample_rate)
        logging.info(f"Audio saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving audio: {e}")

@track_resources
def transcribe_audio(audio_file, use_hd=False):
    """Transcribe audio using OpenAI's Whisper API
    
    Args:
        audio_file (str): Path to the audio file
        use_hd (bool): Whether to use the HD model for higher quality transcription
        
    Returns:
        str: Transcribed text or None if transcription fails
    """
    client = OpenAIClient.get_instance()
    if client is None:
        logging.error("OpenAI client not initialized")
        return None
        
    try:
        logging.info(f"Transcribing audio file: {audio_file}")
        with open(audio_file, "rb") as file:
            response = client.audio.transcriptions.create(
                model="whisper-1-hd" if use_hd else "whisper-1",
                file=file,
                response_format="text"
            )
        logging.info("Transcription successful")
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            logging.error("Authentication failed. Please check your API key.")
        elif "403" in error_msg:
            logging.error("Access forbidden. Please check your API key permissions.")
        elif "429" in error_msg:
            logging.error("Rate limit exceeded. Please wait before trying again.")
        elif "413" in error_msg:
            logging.error("File too large. Maximum file size is 25MB.")
        else:
            logging.error(f"Error transcribing audio: {e}")
        return None

@track_resources
def speak(text, voice="alloy", use_hd=False):
    """Convert text to speech using OpenAI's TTS API
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        use_hd (bool): Whether to use the HD model for higher quality speech
    """
    logging.info(f"Converting text to speech: {text[:50]}...")
    client = OpenAIClient.get_instance()
    if client is None:
        logging.error("OpenAI client not initialized")
        return
        
    try:
        # Generate speech using OpenAI's TTS API
        response = client.audio.speech.create(
            model="tts-1-hd" if use_hd else "tts-1",
            voice=voice,
            input=text
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            response.stream_to_file(temp_file.name)
            logging.info("TTS audio generated successfully")
            
            # Play the audio
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            os.unlink(temp_file.name)
            logging.info("Audio playback completed")
            
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            logging.error("Authentication failed. Please check your API key.")
        elif "403" in error_msg:
            logging.error("Access forbidden. Please check your API key permissions.")
        elif "429" in error_msg:
            logging.error("Rate limit exceeded. Please wait before trying again.")
        elif "400" in error_msg and "voice" in error_msg.lower():
            logging.error(f"Invalid voice selected. Available voices are: alloy, echo, fable, onyx, nova, shimmer")
        else:
            logging.error(f"Error in text-to-speech: {e}")

@track_resources
def get_ai_response(text):
    """Get AI response using OpenAI's chat completion API."""
    client = OpenAIClient.get_instance()
    if not client:
        logging.error("OpenAI client not initialized")
        return "I apologize, but I encountered an error. Please try again."
        
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using the latest model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Keep your responses concise and natural, as if speaking to a friend."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.7,  # Add some randomness to responses
            max_tokens=150    # Limit response length for voice output
        )
        logging.info("AI response generated")
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            logging.error("Authentication failed. Please check your API key.")
        elif "403" in error_msg:
            logging.error("Access forbidden. Please check your API key permissions.")
        elif "429" in error_msg:
            logging.error("Rate limit exceeded. Please wait before trying again.")
        elif "model_not_found" in error_msg.lower():
            logging.error("Model not found. Falling back to GPT-3.5-turbo")
            # Fallback to GPT-3.5-turbo
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Keep your responses concise and natural, as if speaking to a friend."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        else:
            logging.error(f"Error getting AI response: {e}")
        return "I apologize, but I encountered an error. Please try again."

class VoiceChat:
    """Main voice chat application class.
    
    This class handles the core functionality of the voice chat application,
    including audio recording, speech-to-text conversion, AI response generation,
    and text-to-speech synthesis.
    
    Attributes:
        client (OpenAI): The OpenAI client instance
        audio_pool (AudioFilePool): Pool for managing temporary audio files
        conversation_history (List[Dict[str, str]]): History of the conversation
        current_voice (Dict[str, Any]): Current voice settings
    """
    
    def __init__(self, client: OpenAI):
        """Initialize the voice chat application.
        
        Args:
            client (OpenAI): An initialized OpenAI client instance
        """
        self.client = client
        self.audio_pool = AudioFilePool()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_voice: Optional[Dict[str, Any]] = None
        logger.info("Voice chat initialized successfully")

    def start(self, voice_name: str) -> None:
        """Start the voice chat session.
        
        Args:
            voice_name (str): Name of the voice to use for text-to-speech
            
        Raises:
            ValueError: If the voice name is invalid
            Exception: If an error occurs during the chat session
        """
        try:
            # Set up voice settings
            if voice_name not in VOICE_SETTINGS:
                raise ValueError(f"Invalid voice name: {voice_name}")
            
            self.current_voice = VOICE_SETTINGS[voice_name]
            logger.info(f"Selected voice: {voice_name}")

            print(f"\nVoice chat started with {voice_name}!")
            print("Press Enter to start recording, or 'quit' to exit.")
            print("=" * 50)

            while True:
                user_input = input("\nPress Enter to speak (or 'quit' to exit): ")
                if user_input.lower() == 'quit':
                    break

                # Record and process audio
                audio_data = self.record_audio()
                if audio_data is None:
                    continue

                text = self.speech_to_text(audio_data)
                if not text:
                    print("No speech detected. Please try again.")
                    continue

                print(f"\nYou said: {text}")

                response = self.get_ai_response(text)
                print(f"\nAI: {response}")

                self.text_to_speech(response)

        except KeyboardInterrupt:
            print("\nVoice chat terminated by user.")
        except Exception as e:
            logger.error(f"Error in voice chat session: {str(e)}")
            raise
        finally:
            self.cleanup()

    def record_audio(self):
        """Record audio from microphone."""
        try:
            # Audio recording parameters
            sample_rate = 16000
            channels = 1
            dtype = np.float32

            # Initialize recording
            recording = []
            is_recording = True

            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Recording status: {status}")
                if is_recording:
                    recording.append(indata.copy())

            # Start recording
            with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype, callback=callback):
                input("Press Enter to stop recording...")
                is_recording = False

            # Concatenate recorded chunks
            if recording:
                audio_data = np.concatenate(recording, axis=0)
                logger.info(f"Recorded {len(audio_data)} samples")
                return audio_data
            return None

        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return None

    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper API."""
        try:
            # Save audio data to temporary file
            with TemporaryAudioFile() as temp_file:
                sf.write(temp_file.path, audio_data, 16000)
                
                # Transcribe audio using Whisper
                with open(temp_file.path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    return response.text.strip()

        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {str(e)}")
            return None

    def get_ai_response(self, text):
        """Get AI response using GPT-4."""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})

            # Get AI response
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=self.conversation_history,
                max_tokens=150
            )

            # Extract and store AI response
            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})

            return ai_response

        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

    def text_to_speech(self, text):
        """Convert text to speech using TTS API."""
        try:
            # Generate speech using TTS API
            response = self.client.audio.speech.create(
                model="tts-1-hd",
                voice=self.current_voice["name"],
                input=text
            )

            # Create a temporary file with .mp3 extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = os.path.abspath(temp_file.name)
            temp_file.close()

            try:
                # Save the audio to the temporary file
                response.stream_to_file(temp_path)
                
                # Ensure the file exists before trying to play it
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"Temporary file not created at {temp_path}")
                
                # Play the audio
                pygame.mixer.init()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            print("Error generating speech. Please try again.")

    def play_audio(self, file_path):
        """Play audio file using pygame."""
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()

        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            print("Error playing audio. Please try again.")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.audio_pool.cleanup()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main(max_iterations=None):
    """Main function to run the voice chat application."""
    print("\nWelcome to Voice Chat!")
    print("\nAvailable voices:")
    
    for voice, description in AVAILABLE_VOICES.items():
        print(f"- {voice}: {description}")
    
    # Let user choose a voice
    voice = input("\nChoose a voice (press Enter for default 'default'): ").strip().lower()
    if not voice:
        voice = 'default'
    elif voice not in AVAILABLE_VOICES:
        print(f"Invalid voice '{voice}', using 'default' instead")
        voice = 'default'
    
    print(f"\nUsing voice: {voice} - {AVAILABLE_VOICES[voice]}")
    print("\nPress Enter to start recording, Ctrl+C to exit")
    
    iteration = 0
    try:
        with AudioFilePool(max_files=2) as pool:
            while max_iterations is None or iteration < max_iterations:
                try:
                    input("\nPress Enter to start recording...")
                    logging.debug(f"Starting iteration {iteration}")
                    
                    # Record audio
                    recording = record_audio()
                    if recording is None:
                        print("Recording failed, please try again")
                        continue
                    
                    logging.debug(f"Recording completed in iteration {iteration}")
                    
                    # Get a file from the pool for recording
                    audio_file = pool.get_file()
                    logging.debug(f"Got audio file from pool: {audio_file.path}")
                    
                    with audio_file as audio_path:
                        # Save audio to temporary file
                        save_audio(recording, audio_path)
                        logging.debug(f"Saved audio to {audio_path}")
                        
                        # Transcribe audio
                        transcription = transcribe_audio(audio_path)
                        if not transcription:
                            print("Transcription failed, please try again")
                            continue
                        
                        logging.debug(f"Transcription completed: {transcription}")
                        print(f"\nYou said: {transcription}")
                        
                        # Get AI response
                        response = get_ai_response(transcription)
                        if not response:
                            print("Failed to get AI response, please try again")
                            continue
                        
                        logging.debug(f"Got AI response: {response}")
                        print(f"\nAI: {response}")
                        
                        # Convert response to speech
                        speak(response, voice=voice)
                        logging.debug("Speech completed")
                    
                    iteration += 1
                    logging.debug(f"Completed iteration {iteration}")
                    
                except Exception as e:
                    logging.error(f"Error in iteration {iteration}: {str(e)}", exc_info=True)
                    print(f"An error occurred in iteration {iteration}: {str(e)}")
                    if max_iterations is not None:
                        iteration += 1
                    continue
                    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
        print("An unexpected error occurred. Please try again.")
    
    return iteration  # Return the number of completed iterations for testing

if __name__ == "__main__":
    main() 