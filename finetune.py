from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
# If you don't know how to setup secret key on g colab
from huggingface_hub import login
login(new_session=False)

class baseModel:
    """
    Base class for loading Qwen model and/or tokenizer.
    """
    def __init__(self, load_model=True):
        self.model_name = "Qwen/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = None
        if load_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=False,
                revision="main"
            )
            
# use model for training

model = baseModel().model

model.train()

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model = get_peft_model(model, config)

# trainable parameter count
model.print_trainable_parameters()

# load dataset
data = load_dataset("Yodhasu04/prethesis_dataset")

# Function to merge the "messages" field into a single string
def merge_messages(messages):
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

# Tokenization function
def tokenize_function(examples):
    texts = [merge_messages(m) for m in examples["messages"]]
    tokenizer = baseModel(load_model=False).tokenizer
    tokenizer.truncation_side = "left"
    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

# setting pad token
baseModel(load_model=False).tokenizer.pad_token = baseModel(load_model=False).tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(baseModel(load_model=False).tokenizer, mlm=False)

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "tryFineTUne",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",

)

# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)


# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# renable warnings
model.config.use_cache = True