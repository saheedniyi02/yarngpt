# YarnGPT 🎙️
![image/png](https://github.com/saheedniyi02/yarngpt/blob/main/notebooks%2Faudio_0c026c21-f432-4d20-a86b-899a10d9ed60.webp)
A text-to-speech model generating natural Nigerian-accented English speech. Built on pure language modeling without external adapters.

Web Url: https://yarngpt.co/

## Quick Start

```python

!git clone https://github.com/saheedniyi02/yarngpt.git

pip install outetts uroman

import os
import re
import json
import torch
import inflect
import random
import uroman as ur
import numpy as np
import torchaudio
import IPython
from transformers import AutoModelForCausalLM, AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer


!wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
!gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt


from yarngpt.audiotokenizer import AudioTokenizerV2

tokenizer_path="saheedniyi/YarnGPT2"
wav_tokenizer_config_path="/content/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
wav_tokenizer_model_path = "/content/wavtokenizer_large_speech_320_24k.ckpt"


audio_tokenizer=AudioTokenizerV2(
    tokenizer_path,wav_tokenizer_model_path,wav_tokenizer_config_path
    )


model = AutoModelForCausalLM.from_pretrained(tokenizer_path,torch_dtype="auto").to(audio_tokenizer.device)

#change the text
text="The election was won by businessman and politician, Moshood Abiola, but Babangida annulled the results, citing concerns over national security."

# change the language and voice
prompt=audio_tokenizer.create_prompt(text,lang="english",speaker_name="idera")

input_ids=audio_tokenizer.tokenize_prompt(prompt)

output  = model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000,
            #num_beams=5,# using a beam size helps for the local languages but not english
        )

codes=audio_tokenizer.get_codes(output)
audio=audio_tokenizer.get_audio(codes)
IPython.display.Audio(audio,rate=24000)
torchaudio.save(f"Sample.wav", audio, sample_rate=24000)

```

## Features

- 🗣️ 12 preset voices (6 male, 6 female)
- 🎯 Trained on 2000+ hours of Nigerian audio
- 🔊 24kHz high-quality audio output
- 🚀 Simple API for quick integration
- 📝 Support for long-form text

## Available Voices
- Female: zainab, idera, regina, chinenye, joke, remi
- Male: jude, tayo, umar, osagie, onye, emma

## Examples

Check out our [demo notebook](link-to-notebook) or listen to [sample outputs](https://huggingface.co/saheedniyi/YarnGPT/tree/main/audio).

## Model Details

- Base: [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- Training: 5 epochs on A100 GPU
- Data: Nigerian movies, podcasts, and open-source audio
- Architecture: Pure language modeling approach

## Limitations

- English to Nigerian-accented English only
- May not capture all Nigerian accent variations
- Training data includes auto-generated content

## Citation

```bibtex
@misc{yarngpt2025,
  author = {Saheed Azeez},
  title = {YarnGPT: Nigerian-Accented English Text-to-Speech Model},
  year = {2025},
  publisher = {Hugging Face}
}
```

## License
MIT

## Acknowledgments
Built with [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) and inspired by [OuteTTS](https://huggingface.co/OuteAI/OuteTTS-0.2-500M/).
