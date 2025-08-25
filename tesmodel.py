from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("D:/backup project/PreThesis/qwen3-8b-qlora-finetuned-full")
print(sum(p.numel() for p in model.parameters()))
