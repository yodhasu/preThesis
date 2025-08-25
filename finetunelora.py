import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
from huggingface_hub import login
import math

def compute_metrics(eval_pred):
    loss = eval_pred.metrics["eval_loss"]
    perplexity = math.exp(loss)
    return {"perplexity": perplexity}


load_dotenv()
login(new_session=False, token=os.getenv("HF_TOKEN"))

local_model = snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.3",
    cache_dir="model/mistralai/Mistral-7B-v0.3/huggingface_cache",
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=False
)

torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

MODEL_NAME = local_model # "mistralai/Mistral-7B-v0.3"
MAX_LEN = 256  # reduced for T4 safety

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# data = load_dataset("json", data_files={
#     "train": "/content/train.jsonl",
#     "test": "/content/test.jsonl"
# })

# # If you only have one file:
# # data = load_dataset("json", data_files={"all": "/content/aura_130_messages.jsonl"})
# # data = data["all"].train_test_split(test_size=0.1, seed=42)

data = load_dataset("Yodhasu04/prethesis_dataset")

def merge_messages(messages):
    # Flatten into single string
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

def tokenize_function(batch):
    texts = []
    for messages in batch["messages"]:
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                conversation += f"[SYSTEM]: {content}\n"
            elif role == "user":
                conversation += f"[USER]: {content}\n"
            elif role == "assistant":
                conversation += f"[ASSISTANT]: {content}\n"
        texts.append(conversation)

    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

tokenized_dataset = data.map(tokenize_function, batched=True, remove_columns=["messages"])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # fp16 for T4
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=False,
)

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_bs = 1
grad_acc = 16
eval_bs = 1
# full_dataset = tokenized_dataset["train"].concatenate(tokenized_dataset["test"])


# args = TrainingArguments(
#     output_dir="./qwen3-8b-qlora",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     learning_rate=2e-5,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     num_train_epochs=1,
#     warmup_ratio=0.05,
#     lr_scheduler_type="cosine",
#     weight_decay=0.01,
#     fp16=False,
#     bf16=True if torch.cuda.is_available() else False,
#     max_grad_norm=0.5,
#     save_total_limit=2,
#     logging_steps=20,
#     report_to="none"
# )

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    eval_strategy="epoch",       # <-- Evaluate every epoch
    save_strategy="epoch",             # <-- Save checkpoints
    load_best_model_at_end=True,       # <-- Needed for early stopping
    metric_for_best_model="eval_loss", # <-- Tell Trainer what to monitor
    greater_is_better=False,           # <-- Lower loss = better
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

trainer.model.save_pretrained("./mistralai/Mistral-7B-v0.3/adapter")
tokenizer.save_pretrained("./mistralai/Mistral-7B-v0.3/tokenizer")
