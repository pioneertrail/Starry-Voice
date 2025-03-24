import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    return recording

def save_audio(recording, filename, sample_rate=44100):
    """Save recorded audio to WAV file."""
    sf.write(filename, recording, sample_rate)
    print(f"Saved to {filename}")

def transcribe_audio(client, audio_file):
    """Transcribe audio using OpenAI's Whisper API."""
    with open(audio_file, "rb") as file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format="text"
        )
    return response

def get_ai_response(client, text):
    """Get AI response using OpenAI's chat completion."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content

def text_to_speech(client, text):
    """Convert text to speech using OpenAI's TTS API."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file("response.mp3")

def play_audio(filename):
    """Play audio file."""
    data, samplerate = sf.read(filename)
    sd.play(data, samplerate)
    sd.wait()

def main():
    try:
        # Initialize OpenAI client
        client = OpenAI()
        
        # Record audio
        recording = record_audio(duration=5)
        save_audio(recording, "input.wav")
        
        # Transcribe audio
        print("\nTranscribing audio...")
        text = transcribe_audio(client, "input.wav")
        print(f"You said: {text}")
        
        # Get AI response
        print("\nGetting AI response...")
        response = get_ai_response(client, text)
        print(f"AI response: {response}")
        
        # Convert response to speech
        print("\nConverting response to speech...")
        text_to_speech(client, response)
        
        # Play response
        print("\nPlaying response...")
        play_audio("response.mp3")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 