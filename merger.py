from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(new_session=False, token=os.getenv("HF_TOKEN"))

local_model = snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.3",
    cache_dir="model/mistralai/Mistral-7B-v0.3/huggingface_cache",
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=False
)
# Paths
BASE_MODEL = local_model # r"D:\backup project\PreThesis\model"
ADAPTER_PATH = r"D:\backup project\PreThesis\mistralai\Mistral-7B-v0.3\adapter"
OUTPUT_DIR = r"D:\backup project\PreThesis\Mistral-7B-merged"

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Load LoRA adapter into model
print("Merging LoRA adapters...")
merged_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
merged_model = merged_model.merge_and_unload()  # <-- This actually merges LoRA into model weights

# Save merged model
print(f"Saving merged model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
merged_model.save_pretrained(OUTPUT_DIR)

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Merging complete! Use this folder for GGUF conversion.")
