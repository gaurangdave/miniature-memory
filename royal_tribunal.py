# Unsloth library for optimized LLaMA model loading
from unsloth import FastLanguageModel
import torch  # PyTorch for tensor operations and GPU management
import gc  # Python garbage collector for memory cleanup
# Supervised Fine-Tuning trainer (imported but not used in inference)
from trl.trainer.sft_trainer import SFTTrainer
# Training configuration (imported but not used in inference)
from transformers import TrainingArguments
# Dataset loading utilities (imported but not used in inference)
from datasets import load_dataset
import json


class RoyalTribunal:
    def __init__(self, model, tokenizer, questions):
        self.model = model
        self.tokenizer = tokenizer
        self.questions = questions
        self.alpaca_prompt = """
        Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.

        ### Instruction:
        {}    # Placeholder for user's instruction/question

        ### Input:
        {}    # Placeholder for additional context (often empty for chat)

        ### Response:
        """

        pass

    def __generate_response(self):
        responses = []
        for question in self.questions:
            inputs = self.tokenizer([self.alpaca_prompt.format(question, "", "")],  # Format: instruction, input (empty), response (empty)
                                    return_tensors="pt"  # Return PyTorch tensors instead of lists
                                    # Move input tensors to GPU for processing
                                    ).to("cuda")

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # # Process and extract the generated response
            # # tokenizer.batch_decode() converts token IDs back to human-readable text
            # full_response = tokenizer.batch_decode(
            #     outputs,  # Generated token sequences from the model
            #     # Remove special tokens like <eos>, <pad>, etc.
            #     skip_special_tokens=True
            # )[0]  # Get the first (and only) response from the batch

            # # Extract the original prompt text to separate it from the generated response
            # prompt_text = tokenizer.batch_decode(
            #     inputs.input_ids,  # Original input token sequence
            #     skip_special_tokens=True  # Remove special tokens for clean text
            # )[0]

            # # Extract only the newly generated text by removing the prompt
            # generated_text = full_response[len(prompt_text):].strip()
            # generated_text = generated_text.replace("<|end_of_text|>", "")

            # Decode
            raw_text = self.tokenizer.batch_decode(outputs)[0]
            print(raw_text)
            # Parse out just the response (simple string splitting)
            response_text = raw_text.split("### Response:")[-1].strip()
            response_text = response_text.replace("<|end_of_text|>", "")

            print("\n======\n")
            print(f"Question : {question}")
            print(f"Response : {response_text}")
            print("\n======\n")
            responses.append({"question": question, "response": response_text})

            # Memory cleanup to prevent GPU memory accumulation
            del inputs, outputs  # Delete tensor objects
            gc.collect()  # Run Python garbage collection
            torch.cuda.empty_cache()  # Clear GPU memory cache

        return responses

    def run_court_sessions(self):
        print("\n--- The Royal Tribunal is in Session ---")
        responses = self.__generate_response()
        resp_obj = {
            "responses": responses
        }

        with open("./data/test_responses.json", "w") as f:
            json.dump(resp_obj, f, indent=4)
        print("successfully wrote responses to json file")


if __name__ == "__main__":
    # Maximum sequence length for inference (reduced for faster processing)
    max_seq_length = 1024
    dtype = None  # Data type for model weights - None enables automatic detection based on hardware
    load_in_4bit = True  # Enable 4-bit quantization to reduce memory usage during inference

    print("Loading your Royal Model...")
    # # FastLanguageModel.from_pretrained() loads the fine-tuned model with LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Relative path to the saved LoRA model directory
        model_name="./notebooks/lora_model_v3",
        # Maximum tokens the model can process in one sequence
        max_seq_length=max_seq_length,
        # Data type for model weights (auto-detected based on hardware capabilities)
        dtype=dtype,
        load_in_4bit=load_in_4bit,  # Enable 4-bit quantization to reduce VRAM usage
        device_map="cuda",  # Force model placement on GPU to prevent CPU offloading
    )
    FastLanguageModel.for_inference(model)
    # step 1: read test data
    with open("./data/test_questions.json", "r") as f:
        test_data_dict = json.load(f)
        questions = test_data_dict["questions"]

    rt = RoyalTribunal(model=model, tokenizer=tokenizer, questions=questions)
    rt.run_court_sessions()
