import torch
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel
import json
from datasets import Dataset
max_seq_length = 8192 # Supports automatic RoPE Scaling, so choose any number.

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "allenai/tulu-2-dpo-7b",
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)

training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,
)

train_dataset = json.load(open("data/dpo_train.json", 'r'))
train_dataset = Dataset.from_dict(train_dataset)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    bf16=False,
)
dpo_trainer.train()