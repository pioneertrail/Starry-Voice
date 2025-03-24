# StarryVoice: A Voice Chat AI Inspired by Van Gogh's Artistry

A sophisticated voice chat application that enables natural conversation with AI using OpenAI's latest models. Like Van Gogh's masterpieces, StarryVoice brings together technology and artistry to create a vibrant, expressive experience.

## Features

- **Voice-Driven Artistry**: Convert spoken words to text using OpenAI's Whisper API, like capturing the essence of a moment in a painting
- **Natural Language Processing**: Generate contextual responses using GPT-4 Turbo, weaving words like brushstrokes on a canvas
- **Text-to-Speech**: Convert AI responses to natural-sounding speech using TTS-1-HD, bringing the colors of conversation to life
- **Multiple Voice Options**: Choose from six distinct voices, each with its own artistic character:
  - `nova`: Bright and energetic, like the stars in "Starry Night"
  - `alloy`: Balanced and clear, like the composition of "The Bedroom"
  - `echo`: Deep and resonant, like the cypress trees in "Wheat Field with Cypresses"
  - `fable`: Warm and engaging, like the sunflowers in "Sunflowers"
  - `onyx`: Smooth and professional, like the night sky in "Café Terrace at Night"
  - `shimmer`: Light and cheerful, like the colors in "The Yellow House"
- **Conversation History**: Maintain context across interactions, like the layers of paint in a masterpiece
- **Error Handling**: Robust error handling and logging throughout the application

## Prerequisites

- Python 3.8 or higher
- OpenAI API key with access to:
  - GPT-4 Turbo Preview
  - Whisper-1
  - TTS-1-HD
- Microphone for voice input
- Speakers for audio output

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd starryvoice
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Select a voice from the available options, each inspired by Van Gogh's artistic palette

3. Start speaking:
   - Press Enter to begin recording
   - Press Enter again to stop recording
   - Wait for the AI to process and respond
   - Type 'quit' to exit the application

## Project Structure

```
starryvoice/
├── main.py              # Main entry point
├── requirements.txt     # Project dependencies
├── .env                # Environment variables (create this)
└── starryvoice/
    ├── __init__.py
    ├── starryvoice.py  # Main application logic
    └── audio_utils.py  # Audio handling utilities
```

## Voice Settings

Each voice has unique characteristics, inspired by Van Gogh's artistic techniques:

| Voice   | Rate | Volume | Pitch | Description           |
|---------|------|---------|-------|----------------------|
| nova    | 150  | 1.0     | 1.2   | Bright and energetic |
| alloy   | 130  | 0.9     | 1.0   | Balanced and clear   |
| echo    | 140  | 1.0     | 0.9   | Deep and resonant    |
| fable   | 145  | 0.95    | 1.1   | Warm and engaging    |
| onyx    | 135  | 1.0     | 0.95  | Smooth and professional|
| shimmer | 155  | 0.9     | 1.15  | Light and cheerful   |

## Testing

The project includes comprehensive test coverage for all major components:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=starryvoice tests/

# Run specific test file
pytest tests/test_starryvoice.py
```

### Test Structure

- `tests/test_starryvoice.py`: Core functionality tests
- `tests/test_audio_utils.py`: Audio file handling tests
- `tests/test_integration.py`: End-to-end integration tests

### Test Coverage

The test suite covers:
- Voice chat initialization and cleanup
- Audio recording and playback
- Speech-to-text conversion
- AI response generation
- Text-to-speech synthesis
- Error handling and edge cases
- OpenAI client management

## Development

### Code Style

The project follows PEP 8 guidelines and includes type hints for better code maintainability.

### Adding New Features

1. Create tests first in the appropriate test file
2. Implement the feature in the corresponding module
3. Ensure all tests pass and maintain code coverage
4. Update documentation as needed

### Version Control

```bash
# Create a new feature branch
git checkout -b feature/your-feature-name

# Run tests before committing
pytest

# Commit changes
git add .
git commit -m "feat: your feature description"

# Push changes
git push origin feature/your-feature-name
```

## Recent Changes

- Renamed to StarryVoice with artistic theme
- Added comprehensive test suite with pytest
- Improved error handling and logging
- Added type hints throughout the codebase
- Updated dependencies with version ranges
- Enhanced documentation
- Fixed temporary file handling in text-to-speech
- Improved voice selection interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the powerful APIs
- Vincent van Gogh for inspiring the artistic spirit
- The open-source community for the various Python libraries used in this project 