import os
import re
import json
import torch
import inflect
import random
import uroman as ur
import numpy as np
import torchaudio
from transformers import AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer
from outetts.wav_tokenizer.encoder.utils import convert_audio

class AudioTokenizer:

    def __init__(self,tokenizer_path,wav_tokenizer_model_path,wav_tokenizer_config_path,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{audio_start}\n"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.bos = "<|im_start|>"
        self.eos = "<|im_end|>"
        self.input_length=0
        self.special_tokens = {
            "audio_code": "<|{}|>",
            "text_start": "<|text_start|>",
            "text_end": "<|text_end|>",
            "audio_start": "<|audio_start|>",
            "audio_end": "<|audio_end|>",
            "time": "<|t_{:.2f}|>",
            "code_start": "<|code_start|>",
            "code_end": "<|code_end|>",
            "text_sep": "<|text_sep|>"
        }
        self.lec = inflect.engine()
        #self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{audio_start}\n"
        #self.config_path = "/content/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        #self.model_path = "/content/wavtokenizer_large_speech_320_24k.ckpt"
        self.wavtokenizer = WavTokenizer.from_pretrained0802(wav_tokenizer_config_path, wav_tokenizer_model_path)
        self.wavtokenizer = self.wavtokenizer.to(self.device)
        self.BASE_DIR = os.path.dirname(__file__)
        self.DEFAULT_SPEAKERS_DIR = os.path.join(self.BASE_DIR, "default_speakers")
        self.speakers=["idera","emma","onye","jude","osagie","tayo","zainab","joke","regina","remi","umar","chinenye"]

    def get_speaker_path(self,speaker_name):
        return os.path.join(self.DEFAULT_SPEAKERS_DIR, f"{speaker_name}.json")

    def load_speaker(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def load_default_speaker(self, name: str):
        name = name.lower().strip()
        speaker_path=self.get_speaker_path(name)
        return self.load_speaker(speaker_path)


    def process_text(self, text: str):

        text = re.sub(r'\d+(\.\d+)?', lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = re.sub(r'[-_/,\.\\]', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def create_audio_prompt(self,words: list) -> str:
        prompt = []
        for i in words:
            word = i["word"]
            duration = self.special_tokens["time"].format(float(i["duration"]))
            tokens = "".join([self.special_tokens["audio_code"].format(c) for c in i["codes"]])
            prompt.append(f'{word}{duration}{self.special_tokens["code_start"]}{tokens}{self.special_tokens["code_end"]}')
        return "\n".join(prompt)

    def create_prompt(self,text,speaker_name="idera"):
        speaker=self.load_default_speaker(speaker_name)
        input_words = self.process_text(speaker["text"]) +  self.process_text(text)
        #input_words = process_text(speaker["text"]) + input_words

        inputs_words_strings = f"{self.special_tokens['text_sep']}".join([i.strip() for i in input_words])
        prompt = self.text_prompt.format(
          bos=self.bos,
          text_start=self.special_tokens['text_start'],
          words=inputs_words_strings,
          text_end=self.special_tokens['text_end'],
          audio_start=self.special_tokens['audio_start']
      )
        prompt += self.create_audio_prompt(speaker["words"])

        return prompt

    def tokenize_prompt(self, prompt):
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        self.input_length=input_ids.shape[1]
        return input_ids.to(self.device)


    def get_audio(self,discrete_code):
        discrete_code=torch.tensor([[discrete_code]]).to(self.device)
        features = self.wavtokenizer.codes_to_features(discrete_code).to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device)
        audio_out = self.wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        return audio_out.to("cpu")

    def extract_integers(self,s):
        # Match integers enclosed in vertical bars |integer|
        matches = re.findall(r'\|(-?\d+)\|', s)
        # Convert matches to integers
        return [int(match) for match in matches]

    def get_codes(self, output):
        new_output=self.tokenizer.decode(output[0][self.input_length:])
        codes=self.extract_integers(new_output)
        return codes


class AudioTokenizerForLocal(AudioTokenizer):

    def __init__(self,tokenizer_path,wav_tokenizer_model_path,wav_tokenizer_config_path,):
        super().__init__(tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path)
        self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{lang}\n{audio_start}\n"
        self.special_tokens = {
            "audio_code": "<|{}|>",
            "text_start": "<|text_start|>",
            "text_end": "<|text_end|>",
            "audio_start": "<|audio_start|>",
            "audio_end": "<|audio_end|>",
            "word_start": "<|word_start|>",
            "word_end": "<|word_end|>",
            "time": "<|t_{:.2f}|>",
            "code_start": "<|code_start|>",
            "code_end": "<|code_end|>",
            "text_sep": "<|text_sep|>",
            "hausa":"<|hausa|>",
            "igbo":"<|igbo|>",
            "yoruba":"<|yoruba|>",
        }
        self.uroman = ur.Uroman()
        self.DEFAULT_SPEAKERS_DIR = os.path.join(self.BASE_DIR, "default_speakers_local")
        self.speakers = [
            "hausa_male1", "hausa_male2","yoruba_male1", "yoruba_male2","igbo_male2" #"igbo_male1", "igbo_male2",
            "hausa_female1", "hausa_female2", "igbo_female1", "igbo_female2", "yoruba_female1", "yoruba_female2"
        ]
        
    def process_text(self, text: str):
        text = self.uroman.romanize_string(text)
        text = re.sub(r'\d+(\.\d+)?', lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = re.sub(r'[-_/,\.\\]', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def create_prompt(self,text,lang,speaker_name=None):
        assert lang in ["hausa","igbo","yoruba"], f"Invalid language: {lang}, language must be one of ['hausa','igbo','yoruba']"
        #if no speaker
        if speaker_name is None:
            if lang=="hausa":
                speaker_name=random.choice(["hausa_male1","hausa_male2","hausa_female1","hausa_female2"])
            elif lang=="igbo":
                speaker_name=random.choice(["igbo_female1","igbo_female2","igbo_male2"])#"igbo_male1"])
            else:
                speaker_name=random.choice(["yoruba_male2","yoruba_female1","yoruba_female2"])
        speaker=self.load_default_speaker(speaker_name)
        input_words = self.process_text(speaker["text"]) +  self.process_text(text)
        #input_words = process_text(speaker["text"]) + input_words

        inputs_words_strings = f"{self.special_tokens['text_sep']}".join([i.strip() for i in input_words])
        prompt = self.text_prompt.format(
          bos=self.bos,
          text_start=self.special_tokens['text_start'],
          words=inputs_words_strings,
          text_end=self.special_tokens['text_end'],
          lang=self.special_tokens[lang],
          audio_start=self.special_tokens['audio_start']
      )
        prompt += self.create_audio_prompt(speaker["words"])

        return prompt


class AudioTokenizerV2(AudioTokenizer):

    def __init__(self,tokenizer_path,wav_tokenizer_model_path,wav_tokenizer_config_path,):
        super().__init__(tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path)
        self.text_prompt = "{bos}\n{text_start}{words}{text_end}\n{lang}\n{audio_start}\n"
        self.asr_prompt="{bos}\n{code_start}{codes}{code_end}\n{asr}\n"
        self.special_tokens = {
            "audio_code": "<|{}|>",
            "text_start": "<|text_start|>",
            "text_end": "<|text_end|>",
            "audio_start": "<|audio_start|>",
            "audio_end": "<|audio_end|>",
            "word_start": "<|word_start|>",
            "word_end": "<|word_end|>",
            "time": "<|t_{:.2f}|>",
            "code_start": "<|code_start|>",
            "code_end": "<|code_end|>",
            "text_sep": "<|text_sep|>",
            "hausa":"<|hausa|>",
            "igbo":"<|igbo|>",
            "yoruba":"<|yoruba|>",
            "english":"<|english|>",#<|english|>
            "asr":"<|asr|>"
        }
        self.uroman = ur.Uroman()
        self.DEFAULT_SPEAKERS_DIR_LOCAL = os.path.join(self.BASE_DIR, "default_speakers_local")
        self.DEFAULT_SPEAKERS_ENG = os.path.join(self.BASE_DIR, "default_speakers")
        self.speakers_local = [
            "hausa_male1", "hausa_male2","yoruba_male1", "yoruba_male2","igbo_male2" #"igbo_male1", "igbo_male2",
            "hausa_female1", "hausa_female2", "igbo_female1", "igbo_female2", "yoruba_female1", "yoruba_female2"
        ]
        self.speakers_eng = ["idera","emma","onye","jude","osagie","tayo","zainab","joke","regina","remi","umar","chinenye","saheed"]
        self.changed_tokens=[('<|1836|>', '<|453|><|453|>'),
                             ('<|1837|>', '<|1836|><|1836|>'),
                             ('<|1838|>', '<|1837|><|1837|>'),
                             ('<|1840|>', '<|244|><|167|>'),
                             ('<|1841|>', '<|235|><|219|>'),
                             ('<|1844|>', '<|453|><|244|>'),
                             ('<|1845|>', '<|1838|><|1838|>')]

    def process_text(self, text: str):
        text = self.uroman.romanize_string(text)
        text = re.sub(r'\d+(\.\d+)?', lambda x: self.lec.number_to_words(x.group()), text.lower())
        text = re.sub(r'[-_/,\.\\]', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def get_speaker_path(self,speaker_name,dir):
        return os.path.join(dir, f"{speaker_name}.json")

    def load_speaker(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def load_default_speaker(self, name: str,dir: str):
        name = name.lower().strip()
        speaker_path=self.get_speaker_path(name,dir)
        return self.load_speaker(speaker_path)

    def create_prompt(self,text,lang,speaker_name=None):
        assert lang in ["hausa","igbo","yoruba","english"], f"Invalid language: {lang}, language must be one of ['hausa','igbo','yoruba','english']"
        #if no speaker
        dir=self.DEFAULT_SPEAKERS_DIR_LOCAL
        if speaker_name is None:
            if lang=="hausa":
                speaker_name=random.choice(["hausa_male1","hausa_male2","hausa_female1","hausa_female2"])
            elif lang=="igbo":
                speaker_name=random.choice(["igbo_female1","igbo_female2","igbo_male2"])#"igbo_male1"])
            elif lang=="yoruba":
                speaker_name=random.choice(["yoruba_male2","yoruba_female1","yoruba_female2"])
            else:
                speaker_name=random.choice(self.speakers_eng)
                
        if lang=="english":
            dir=self.DEFAULT_SPEAKERS_ENG
        speaker=self.load_default_speaker(speaker_name,dir)
        input_words = self.process_text(speaker["text"]) +  self.process_text(text)
        #input_words = process_text(speaker["text"]) + input_words

        inputs_words_strings = f"{self.special_tokens['text_sep']}".join([i.strip() for i in input_words])
        prompt = self.text_prompt.format(
          bos=self.bos,
          text_start=self.special_tokens['text_start'],
          words=inputs_words_strings,
          text_end=self.special_tokens['text_end'],
          lang=self.special_tokens[lang],
          audio_start=self.special_tokens['audio_start']
      )
        prompt += self.create_audio_prompt(speaker["words"])

        return prompt
    def replace_tokens(text):
      for pair in self.changed_tokens:
        text=text.replace(pair[0],pair[-1])
      return text 

    def resample(self,audio: np.ndarray, sr: int, target_sr: int):
        audio = audio.to(dtype=torch.float32)
        #.clone().detach()
        audio = audio.unsqueeze(0)
        # 1 as last arg corresponds to mono audio
        resampled = convert_audio(audio, sr, target_sr, 1)
        return resampled.to(self.device )

    def quantize_wavtokenizer(self, path):
        audio_data, sample_rate = torchaudio.load(path)
        audio_data=audio_data.squeeze()
        audio = self.resample(audio_data, sample_rate, 24000).to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device )
        _, codes = self.wavtokenizer.encode_infer(audio, bandwidth_id=bandwidth_id)
        codes = codes.squeeze(1).to(self.device)#+last_text_token
        res=""
        for code in codes[0].tolist():
            res+=f"<|{code}|>"
        return res
        
    def load_asr_prompt(self,audio_path):
        codes=self.quantize_wavtokenizer(audio_path)
        prompt = self.asr_prompt.format(
          bos=self.bos,
          code_start=self.special_tokens['code_start'],
          codes=codes,
          code_end=self.special_tokens['code_end'],
          asr=self.special_tokens["asr"],
        )
        return prompt

    def get_asr_results(self,output):
        res=""
        for text in self.tokenizer.decode(output[0]).split("<|text_start|>")[-1].split("<|text_end|>")[0].split("\n"):
            res+=text.split("<|word_start|>")[-1].split("<|word_end|>")[0]
            res+=" "
        return res.strip()