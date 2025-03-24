"""
Test script for voice chat with different voices.
"""
import sys
import os
import time
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from voice_chat.voice_chat import speak, VOICE_CONFIGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_voice(voice, test_phrase):
    """Test a specific voice with a test phrase."""
    config = VOICE_CONFIGS[voice]
    print(f"\nTesting {voice} voice...")
    print(f"Properties: rate={config['rate']} wpm, volume={config['volume']}")
    print(f"Test phrase: {test_phrase}")
    speak(test_phrase, voice=voice)
    time.sleep(2)  # Pause between tests

def main():
    """Run voice tests."""
    print("\n=== Voice Chat Voice Test Suite ===")
    print("\nAvailable voices and their properties:")
    for voice, config in VOICE_CONFIGS.items():
        print(f"- {voice}: {config['description']}")
        print(f"  Rate: {config['rate']} wpm")
        print(f"  Volume: {config['volume']}")
    
    # Test phrases for each voice
    test_phrases = {
        'default': "In the swirling night sky of Arles, stars dance like golden fireflies, casting their ethereal glow upon the sleeping town below.",
        'male': "The sunflowers stand tall in the Provençal fields, their golden faces turned towards the sun, like a chorus of nature's most vibrant voices.",
        'female': "Through the window of the yellow house, the morning light streams in, painting the walls with the warm colors of a new day in Arles."
    }
    
    print("\nStarting voice tests...")
    for voice in VOICE_CONFIGS:
        test_voice(voice, test_phrases[voice])
    
    print("\nVoice tests completed!")

if __name__ == "__main__":
    main() 