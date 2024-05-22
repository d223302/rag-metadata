import torch
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel
import json
from datasets import Dataset
from transformers import TrainingArguments
max_seq_length = 8192 # Supports automatic RoPE Scaling, so choose any number.

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "allenai/tulu-2-dpo-13b",
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
    random_state = 42,
)

train_dataset = json.load(open("data/dpo_test.json", 'r'))
train_dataset = Dataset.from_dict(train_dataset)



dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args = DPOConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "dpo_13b_output",
    ),
    beta = 0.1,
)
dpo_trainer.train()
dpo_trainer.save_model("dpo_13b_output")
