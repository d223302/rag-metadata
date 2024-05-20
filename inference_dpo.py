import torch
from unsloth import FastLanguageModel
import json
from datasets import Dataset
max_seq_length = 8192 # Supports automatic RoPE Scaling, so choose any number.

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "dpo_output",
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
)

dialogue = [
    {"role": "user", "content": "Who are you?"},
]


prompt = tokenizer.apply_chat_template(dialogue, tokenize = False)
prompt += "<|assistant|>\n"
print(prompt)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, do_sample=False, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)