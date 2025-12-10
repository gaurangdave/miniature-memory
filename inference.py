# Import required libraries for model loading, inference, and memory management
from unsloth import FastLanguageModel  # Unsloth library for optimized LLaMA model loading
import torch  # PyTorch for tensor operations and GPU management
import gc  # Python garbage collector for memory cleanup
from trl.trainer.sft_trainer import SFTTrainer  # Supervised Fine-Tuning trainer (imported but not used in inference)
from transformers import TrainingArguments  # Training configuration (imported but not used in inference)
from datasets import load_dataset  # Dataset loading utilities (imported but not used in inference)

def start_inference_loop():
    """
    Main function to start an interactive inference loop with the fine-tuned model.
    Loads the model, sets up the prompt template, and handles user interactions.
    """
    # 1. Configuration for Inference
    # Load the fine-tuned model from the "lora_model" folder created during training
    # Unsloth automatically finds the base model from the adapter configuration
    max_seq_length = 1024  # Maximum sequence length for inference (reduced for faster processing)
    dtype = None  # Data type for model weights - None enables automatic detection based on hardware
    load_in_4bit = True  # Enable 4-bit quantization to reduce memory usage during inference

    print("Loading your Royal Model...")
    # FastLanguageModel.from_pretrained() loads the fine-tuned model with LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./notebooks/lora_model",  # Relative path to the saved LoRA model directory
        max_seq_length = max_seq_length,  # Maximum tokens the model can process in one sequence
        dtype = dtype,  # Data type for model weights (auto-detected based on hardware capabilities)
        load_in_4bit = load_in_4bit,  # Enable 4-bit quantization to reduce VRAM usage
        device_map = "cuda",  # Force model placement on GPU to prevent CPU offloading
    )

    # 2. Enable Native Inference (2x Faster)
    # FastLanguageModel.for_inference() optimizes the model for inference by:
    # - Disabling gradient computation
    # - Removing training-specific components
    # - Enabling faster attention mechanisms
    FastLanguageModel.for_inference(model) 

    # 3. The Prompt Template
    # This template MUST be identical to the one used during training for consistent results
    # The Alpaca format provides structured instruction-following capability
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}    # Placeholder for user's instruction/question

    ### Input:
    {}    # Placeholder for additional context (often empty for chat)

    ### Response:
    """   # Response section where model generates its answer
    
    print("\n--- The Royal Court is in Session ---")
    print("(Type 'exit' to abdicate the throne)\n")

    # Start interactive conversation loop
    while True:
        # Get user input from command line
        user_input = input("You: ")
        
        # Check for exit commands to break the loop
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Tokenize the formatted prompt for model input
        inputs = tokenizer(
            [alpaca_prompt.format(user_input, "", "")],  # Format: instruction, input (empty), response (empty)
            return_tensors="pt"  # Return PyTorch tensors instead of lists
        ).to("cuda")  # Move input tensors to GPU for processing

        # Generate response using the fine-tuned model
        outputs = model.generate(
            **inputs,  # Unpack tokenized inputs (input_ids, attention_mask, etc.)
            max_new_tokens=128,  # Maximum number of new tokens to generate
            use_cache=True,  # Enable key-value caching for faster generation
            temperature=0.7,  # Controls randomness (0.1=deterministic, 1.0=creative)
        )

        # Process and extract the generated response
        # tokenizer.batch_decode() converts token IDs back to human-readable text
        full_response = tokenizer.batch_decode(
            outputs,  # Generated token sequences from the model
            skip_special_tokens=True  # Remove special tokens like <eos>, <pad>, etc.
        )[0]  # Get the first (and only) response from the batch
        
        # Extract the original prompt text to separate it from the generated response
        prompt_text = tokenizer.batch_decode(
            inputs.input_ids,  # Original input token sequence
            skip_special_tokens=True  # Remove special tokens for clean text
        )[0]
            
        # Extract only the newly generated text by removing the prompt
        generated_text = full_response[len(prompt_text):].strip()
        
        # Display the model's response
        print(f"Royal Assistant: {generated_text}")
        
        # Memory cleanup to prevent GPU memory accumulation
        del inputs, outputs  # Delete tensor objects
        gc.collect()  # Run Python garbage collection
        torch.cuda.empty_cache()  # Clear GPU memory cache    
    

# Main execution block - only runs when script is executed directly (not imported)
if __name__ == "__main__":
    start_inference_loop()  # Start the interactive inference session
