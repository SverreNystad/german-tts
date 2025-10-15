# 🇩🇪 German TTS with VoxPopuli & SpeechT5

This project fine-tunes **Microsoft’s SpeechT5** model for **German Text-to-Speech** using the **[VoxPopuli dataset](https://huggingface.co/datasets/facebook/voxpopuli)**.

The trained model is available on the Hugging Face Hub: [SverreNystad/speecht5_finetuned_voxpopuli_de](https://huggingface.co/SverreNystad/speecht5_finetuned_voxpopuli_de)

At its core, SpeechT5 is a Transformer-based encoder-decoder model that supports multiple speech and text modalities. The same architecture can be applied to a variety of tasks such as ASR, TTS, and speech translation, making it a versatile foundation for speech research and applications.

For text-to-speech, SpeechT5 takes text input, encodes it into hidden representations, and decodes it into log-mel spectrograms, which are then converted into audio waveforms using a HiFi-GAN vocoder.

By changing the speaker embeddings, the model can generate speech in different voices.

The model architecture is described in detail in the [original paper](https://arxiv.org/abs/2110.07205) and the [model card](https://huggingface.co/microsoft/speecht5_tts).

![SpeechT5 architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/speecht5/tts.jpg)

## 🚀 Overview

- **Base model:** [`microsoft/speecht5_tts`](https://huggingface.co/microsoft/speecht5_tts)
- **Dataset:** `facebook/voxpopuli (de)`
- **Embeddings:** `speechbrain/spkrec-xvect-voxceleb`
- **Monitoring:** Weights & Biases ([Training of model](https://wandb.ai/sverrenystad-ntnu/TTS-voxpopuli/runs/f6lzslrr?nw=nwusersverrenystad))

## 🗣️ Usage

```python
from transformers import SpeechT5HifiGan, SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
from IPython.display import Audio

model = SpeechT5ForTextToSpeech.from_pretrained(
    "SverreNystad/speecht5_finetuned_voxpopuli_de"
)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

speaker_embeddings = torch.zeros((1, 512))  # Use your own speaker embeddings here
text = "Das ist für die Menschen, die nur wissen, wie man Deutsch schreibt, man kann dieses Modell benutzen, um es zu sprechen."
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

Audio(speech.numpy(), rate=16000)
```

______________________________________________________________________

## 👤 Author

**Sverre Nystad**
AI Engineer @ Cogito NTNU
