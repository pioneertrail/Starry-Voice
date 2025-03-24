"""
Script to download the Vosk model.
"""
import os
import sys
import urllib.request
import zipfile
import shutil

def download_vosk_model():
    """Download and extract the Vosk model."""
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_chat", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URL and path
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    model_zip = os.path.join(models_dir, "vosk-model-small-en-us-0.15.zip")
    model_dir = os.path.join(models_dir, "vosk-model-small-en-us-0.15")
    
    # Check if model already exists
    if os.path.exists(model_dir):
        print(f"Model already exists at {model_dir}")
        return
    
    print(f"Downloading Vosk model from {model_url}")
    try:
        # Download the model
        urllib.request.urlretrieve(model_url, model_zip)
        
        # Extract the model
        print("Extracting model...")
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Clean up zip file
        os.remove(model_zip)
        
        print(f"Model downloaded and extracted to {model_dir}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_vosk_model() 