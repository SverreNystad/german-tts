# %% [markdown]
# # German TTS with VoxPopuli and Audio Transformers

# %%
%pip install transformers "datasets==3.6.0" soundfile speechbrain accelerate wandb


# %%
from huggingface_hub import notebook_login, login
from google.colab import userdata
import wandb

from datasets import load_dataset, Audio
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import torch
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt

from IPython.display import Audio
from functools import partial
from transformers import SpeechT5Processor
from transformers import SpeechT5ForTextToSpeech
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan


# %%
HF_TOKEN = os.getenv('HF_TOKEN')
WANDB_API_KEY= os.getenv('WANDB_API_KEY')
LANGUAGE = "de"

if WANDB_API_KEY is None or HF_TOKEN is None:
  print("Please set the HF_TOKEN and WANDB_API_KEY environment variables.")
  

os.environ["WANDB_PROJECT"]="TTS-voxpopuli"

wandb.login(key=WANDB_API_KEY)
login(HF_TOKEN)

# %%
dataset = load_dataset("facebook/voxpopuli", LANGUAGE, split="train", trust_remote_code=True)
len(dataset)

# %%
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# %%

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)

# %%
tokenizer = processor.tokenizer

# %%
dataset[0]

# %%
def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

# %%
dataset_vocab - tokenizer_vocab

# %% [markdown]
# The vocabulary of a language might not be in the tokenizer of a TTS model. In this notebook we will use the [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) dataset to finetune a TTS model for German. Therefore one must find replacements of tokens that are not in the tokenizer.

# %%
replacements = [
    ("ß", "ss"),
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


dataset = dataset.map(cleanup_text)

# %%
speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1

# %%


plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()

# %%
def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400


dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

# %%
len(set(dataset["speaker_id"]))

# %%
len(dataset)

# %%
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

# %%
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

# %%
processed_example = prepare_dataset(dataset[0])
list(processed_example.keys())

# %%
processed_example["speaker_embeddings"].shape

# %%

plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()

# %%
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# %%
def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
len(dataset)

# %%
dataset = dataset.train_test_split(test_size=0.1)

# %%


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

# %%
data_collator = TTSDataCollatorWithPadding(processor=processor)

# %%

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

# %%


# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, use_cache=True)

# %%


training_args = Seq2SeqTrainingArguments(
    output_dir=f"speecht5_finetuned_voxpopuli_{LANGUAGE}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=40_000,
    gradient_checkpointing=True,
    # This is important to fix the backward error
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard", "wandb"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

# %%
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

# %%
trainer.train()

# %%
trainer.push_to_hub()

# %%

model = SpeechT5ForTextToSpeech.from_pretrained(
    f"SverreNystad/speecht5_finetuned_voxpopuli_{LANGUAGE}"
)

# %%
example = dataset["test"][1]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)

# %%
text = "Ich bin ein Berliner! Meine Name bin Sverre Nystad. Ich libe Deutch!"

# %%
inputs = processor(text=text, return_tensors="pt")

# %%

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# %%
Audio(speech.numpy(), rate=16000)


