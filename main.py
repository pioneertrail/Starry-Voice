"""
StarryVoice - A Voice Chat AI Inspired by Van Gogh's Artistry

This application enables natural conversation with AI using OpenAI's latest models,
bringing the artistic spirit of Van Gogh's masterpieces to life through voice.
"""

import logging
from starryvoice.starryvoice import VoiceChat
from starryvoice.starryvoice import OpenAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for StarryVoice."""
    try:
        # Initialize OpenAI client
        client = OpenAIClient.get_instance()
        logger.info("OpenAI client initialized successfully")

        # Create voice chat instance
        voice_chat = VoiceChat(client)
        logger.info("Voice chat instance created successfully")

        # Display available voices
        print("\nAvailable Voices:")
        print("-" * 50)
        for voice_name, settings in voice_chat.VOICE_SETTINGS.items():
            description = settings.get('description', 'No description available')
            print(f"{voice_name}: {description}")
        print("-" * 50)

        # Get voice selection with proper input handling
        while True:
            voice_name = input("\nSelect a voice (or 'quit' to exit): ").strip().lower()
            if voice_name == 'quit':
                return
            if voice_name in voice_chat.VOICE_SETTINGS:
                break
            print(f"Invalid voice name. Please choose from: {', '.join(voice_chat.VOICE_SETTINGS.keys())}")

        # Start voice chat
        voice_chat.start(voice_name)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main()) 