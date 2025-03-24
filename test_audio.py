import sounddevice as sd
import numpy as np
import soundfile as sf
import time

def test_audio():
    # Test parameters
    duration = 3  # seconds
    sample_rate = 44100
    channels = 1
    
    print("Testing audio recording...")
    print(f"Recording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=np.float32
    )
    
    # Wait for the recording to complete
    sd.wait()
    
    print("Recording complete!")
    
    # Save the recording
    filename = "test_recording.wav"
    sf.write(filename, recording, sample_rate)
    print(f"Saved recording to {filename}")
    
    # Play back the recording
    print("\nPlaying back the recording...")
    sd.play(recording, sample_rate)
    sd.wait()
    print("Playback complete!")

if __name__ == "__main__":
    try:
        test_audio()
    except Exception as e:
        print(f"Error: {str(e)}") 