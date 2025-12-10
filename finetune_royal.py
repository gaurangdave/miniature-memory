from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configuration for 8GB VRAM
max_seq_length = 2048 # Do not go higher than this on 8GB
dtype = None # Auto detection
load_in_4bit = True # MANDATORY for 8GB cards

# 2. Load the Model
print("Loading Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Pre-quantized 4bit model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters (The "Fine-Tuning" part)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank. 16 is standard. 8 is safer for VRAM if you OOM.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports 0 only for optimized training
    bias = "none",
    use_gradient_checkpointing = "unsloth", # MANDATORY for 8GB VRAM
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Load your dataset
# Ensure 'royal_dataset.jsonl' is in the same folder
dataset = load_dataset("json", data_files="royal_dataset.jsonl", split="train")

# 5. Formatter Function
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 6. The Trainer
print("Starting Training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can speed up training but uses more VRAM
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Keep this LOW (1 or 2)
        gradient_accumulation_steps = 4, # Increase this to simulate larger batch
        warmup_steps = 5,
        max_steps = 60, # Small step count for "Hello World" test
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # Use 8bit optimizer to save VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

print("Training finished! Saving model...")
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
print("Model saved to /lora_model")