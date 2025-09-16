import json
import os
import time
from collections import deque
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer
from datasets import Dataset
import torch

# --- Configuration ---
LOG_FILE = os.getenv("CUSTOM_AGENT_LOG_FILE", "expt-logs/custom_agent_dataset.jsonl")
CHECKPOINT_DIR = "./dpo_checkpoints"
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
LORA_ADAPTER_NAME = "dpo_adapter"
BUFFER_SIZE = 500  # Number of recent turns to keep in memory
BATCH_SIZE = 8
TRAINING_INTERVAL = 30 # seconds

# --- DPO and Model Setup (runs once at startup) ---
print("Loading base model and tokenizer...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Initializing or loading LoRA adapter...")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"])
if os.path.exists(os.path.join(CHECKPOINT_DIR, LORA_ADAPTER_NAME)):
    model = PeftModel.from_pretrained(base_model, os.path.join(CHECKPOINT_DIR, LORA_ADAPTER_NAME))
else:
    model = get_peft_model(base_model, lora_config)

# --- The Continuous DPO Loop ---
experience_buffer = deque(maxlen=BUFFER_SIZE)
last_read_position = 0

def create_dpo_dataset(data):
    """
    Converts a list of log data into a Hugging Face Dataset for DPO training.
    """
    dpo_data = []
    for turn_data in data:
        # Check for valid data
        if not all(k in turn_data for k in ['chosen_response', 'rejected_response', 'game_state_raw', 'belief_state_intuitive', 'belief_state_critical']):
            continue

        prompt = f"""### Input Context\n{json.dumps(turn_data['game_state_raw'])}\n### Dual-Process Analysis\n**Intuitive Lens (Facts):** {json.dumps(turn_data['belief_state_intuitive'])}\n**Critical Lens (Subtext):** {json.dumps(turn_data['belief_state_critical'])}\n"""
        
        chosen = json.dumps(turn_data['chosen_response'])
        rejected = json.dumps(turn_data['rejected_response'])
        
        dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        
    return Dataset.from_list(dpo_data)

if __name__ == "__main__":
    print("Starting continuous DPO training service...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    while True:
        try:
            with open(LOG_FILE, "r") as f:
                f.seek(last_read_position)
                new_lines = f.readlines()
                for line in new_lines:
                    experience_buffer.append(json.loads(line))
                last_read_position = f.tell()
        except FileNotFoundError:
            print(f"Log file {LOG_FILE} not found. Waiting for agent to create it...")
            time.sleep(TRAINING_INTERVAL)
            continue

        if len(experience_buffer) >= BATCH_SIZE:
            print(f"Training on a new mini-batch of size {BATCH_SIZE}...")
            
            mini_batch_data = list(experience_buffer)
            dpo_dataset = create_dpo_dataset(mini_batch_data)
            
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                beta=0.1,
                train_dataset=dpo_dataset,
                tokenizer=tokenizer,
                args=TrainingArguments(
                    output_dir=CHECKPOINT_DIR,
                    per_device_train_batch_size=BATCH_SIZE,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    save_strategy="epoch",
                    logging_steps=1,
                    fp16=True,
                    disable_tqdm=True,
                )
            )
            
            dpo_trainer.train()
            dpo_trainer.save_model(os.path.join(CHECKPOINT_DIR, LORA_ADAPTER_NAME))
            print("Updated weights saved.")
            
            # Clear the buffer after successful training to ensure we're always using fresh data
            experience_buffer.clear()

        time.sleep(TRAINING_INTERVAL)
