# YarnGPT Python Wrapper Library

## Description
YarnGPT is a Python wrapper for the YarnGPT text-to-speech model, designed to synthesize natural Nigerian-accented English speech using a pure language modeling approach. This library provides a simple API to convert text into audio output, allowing users to select from various preset voices and adjust generation parameters.

## Features
- Supports 6 preset voices (idera, jude, joke, umar, osagie, onye)
- Utilizes Hugging Face's model caching for efficient model loading
- Exposes a straightforward API function: generate_speech(text, speaker, temperature, repetition_penalty, max_length)
- Allows customization of generation parameters such as temperature, repetition penalty, and maximum token length
- Includes unit tests to ensure core functionality

## Installation
1. Create and activate a virtual environment:
   - On Linux/MacOS:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
   - On Windows:
   ```bash
   python -m venv env
   env\Scripts\activate
   ```

2. Install the package:
   ```bash
   pip install yarngpt
   ```

## Usage
Basic usage to generate and save audio:
```python
from yarngpt import generate_speech
import torchaudio

# Generate speech with the default speaker (idera)
audio = generate_speech("Hello, this is a test.")

# Save the generated audio
torchaudio.save("output.wav", audio, sample_rate=24000)
```

For Jupyter Notebook users, you can also play the audio directly:
```python
from yarngpt import generate_speech
import torchaudio
from IPython.display import Audio

# Generate and save speech
audio = generate_speech("Hello, this is a test.", speaker="joke")
torchaudio.save("output.wav", audio, sample_rate=24000)

# Play the audio in the notebook
Audio("output.wav")
```

## Parameter Options
- `text`: The input string to convert to speech
- `speaker`: Choose from available speakers: idera, jude, joke, umar, osagie, onye (default is "idera")
- `temperature`: Controls the randomness of generation (default is 0.1)
- `repetition_penalty`: A factor to reduce repetitive output (default is 1.1)
- `max_length`: The maximum length of the generated output tokens (default is 4000)

## Testing
Run the unit tests to verify functionality:
```bash
python -m unittest discover -s tests
```


## License
This project is licensed under the MIT License.

## Acknowledgments
- Built as a contribution to yarngpt projects
- Utilizes Hugging Face's model caching and the transformers library
- Special thanks to the open-source community for their ongoing support

For more details and documentation, visit the GitHub repository: https://github.com/jerryola1
