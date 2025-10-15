# 🇩🇪 German TTS with VoxPopuli & SpeechT5

This project fine-tunes **Microsoft’s SpeechT5** model for **German Text-to-Speech** using the **[VoxPopuli dataset](https://huggingface.co/datasets/facebook/voxpopuli)**.

The trained model is available on the Hugging Face Hub: [SverreNystad/speecht5_finetuned_voxpopuli_de](https://huggingface.co/SverreNystad/speecht5_finetuned_voxpopuli_de)
---

## 🚀 Overview

* **Base model:** [`microsoft/speecht5_tts`](https://huggingface.co/microsoft/speecht5_tts)
* **Dataset:** `facebook/voxpopuli (de)`
* **Embeddings:** `speechbrain/spkrec-xvect-voxceleb`
* **Logging:** Weights & Biases
* **Output:** Fine-tuned German TTS model pushed to Hugging Face Hub

---

## ⚙️ Setup

```bash
pip install transformers "datasets==3.6.0" soundfile speechbrain accelerate wandb
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
```

```python
from huggingface_hub import login
import wandb
wandb.login(key=WANDB_API_KEY)
login(token=HF_TOKEN)
```

---

## 🧠 Training

```python
from datasets import load_dataset, Audio
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, Seq2SeqTrainer, Seq2SeqTrainingArguments

dataset = load_dataset("facebook/voxpopuli", "de", split="train").cast_column("audio", Audio(sampling_rate=16000))
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_voxpopuli_de",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_steps=40000,
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.push_to_hub()
```

---

## 🗣️ Inference

```python
from transformers import SpeechT5HifiGan, SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
from IPython.display import Audio

model = SpeechT5ForTextToSpeech.from_pretrained("SverreNystad/speecht5_finetuned_voxpopuli_de")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

text = "Der Nationalssozialistische deutsche Arbeiterpartei"
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], torch.zeros((1,512)), vocoder=vocoder)

Audio(speech.numpy(), rate=16000)
```

---

## 👤 Author

**Sverre Nystad**
AI Engineer @ Cogito NTNU
