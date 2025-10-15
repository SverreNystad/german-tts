# 🇩🇪 German TTS with VoxPopuli & SpeechT5

This project fine-tunes **Microsoft’s SpeechT5** model for **German Text-to-Speech** using the **[VoxPopuli dataset](https://huggingface.co/datasets/facebook/voxpopuli)**.

The trained model is available on the Hugging Face Hub: [SverreNystad/speecht5_finetuned_voxpopuli_de](https://huggingface.co/SverreNystad/speecht5_finetuned_voxpopuli_de)

## 🚀 Overview

* **Base model:** [`microsoft/speecht5_tts`](https://huggingface.co/microsoft/speecht5_tts)
* **Dataset:** `facebook/voxpopuli (de)`
* **Embeddings:** `speechbrain/spkrec-xvect-voxceleb`
* **Logging:** Weights & Biases
* **Output:** Fine-tuned German TTS model pushed to Hugging Face Hub


## 🗣️ Usage

```python
from transformers import SpeechT5HifiGan, SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
from IPython.display import Audio

model = SpeechT5ForTextToSpeech.from_pretrained("SverreNystad/speecht5_finetuned_voxpopuli_de")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

text = "Das ist für die Menschen, die nur wissen, wie man Deutsch schreibt, man kann dieses Modell benutzen, um es zu sprechen."
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], torch.zeros((1,512)), vocoder=vocoder)

Audio(speech.numpy(), rate=16000)
```

---

## 👤 Author

**Sverre Nystad**
AI Engineer @ Cogito NTNU
