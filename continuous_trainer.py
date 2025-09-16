import json
import os
import time
import sys
from collections import deque
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from packaging.version import Version
import transformers as _hf_tfm
try:
    # TRL >= 0.9 uses DPOConfig with DPOTrainer
    from trl import DPOTrainer, DPOConfig  # type: ignore
    HAS_DPO_CONFIG = True
except Exception:
    from trl import DPOTrainer  # type: ignore
    DPOConfig = None  # type: ignore
    HAS_DPO_CONFIG = False
from datasets import Dataset
import torch

# --- Configuration ---
LOG_FILE = os.getenv("CUSTOM_AGENT_LOG_FILE", "expt-logs/custom_agent_dataset.jsonl")
CHECKPOINT_DIR = os.getenv("DPO_CHECKPOINT_DIR", "./dpo_checkpoints")
# Prefer a local model path to avoid remote downloads. If not set and ALLOW_HF_DOWNLOAD is not true, training waits.
MODEL_PATH = os.getenv("DPO_BASE_MODEL_PATH")
MODEL_NAME = os.getenv("DPO_BASE_MODEL_NAME", "Qwen/Qwen1.5-7B-Chat")
ALLOW_HF_DOWNLOAD = os.getenv("ALLOW_HF_DOWNLOAD", "true").lower() in {"1", "true", "yes"}
LORA_ADAPTER_NAME = os.getenv("DPO_LORA_ADAPTER_NAME", "dpo_adapter")
BUFFER_SIZE = int(os.getenv("DPO_BUFFER_SIZE", "500"))  # Number of recent turns to keep in memory
BATCH_SIZE = int(os.getenv("DPO_BATCH_SIZE", "8"))
TRAINING_INTERVAL = int(os.getenv("DPO_TRAINING_INTERVAL", "30")) # seconds

# --- DPO and Model Setup (runs once at startup) ---
def load_or_wait_for_model():
    print("Preparing base model and tokenizer for DPO training...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

    model_source = None
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model_source = MODEL_PATH
    elif ALLOW_HF_DOWNLOAD:
        model_source = MODEL_NAME
    else:
        print("No local model path provided (DPO_BASE_MODEL_PATH), and ALLOW_HF_DOWNLOAD is not enabled.")
        print("Continuous trainer will wait. Set DPO_BASE_MODEL_PATH to a local model directory or export ALLOW_HF_DOWNLOAD=true.")
        return None, None

    print(f"Loading model from: {model_source}")
    base_model = AutoModelForCausalLM.from_pretrained(model_source, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing or loading LoRA adapter...")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"])
    adapter_path = os.path.join(CHECKPOINT_DIR, LORA_ADAPTER_NAME)
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = get_peft_model(base_model, lora_config)

    return model, tokenizer

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
    # Ensure compatible Transformers version for current TRL (processing_class support)
    try:
        if Version(_hf_tfm.__version__) < Version("4.46.0"):
            print(
                f"Transformers {_hf_tfm.__version__} is too old for current TRL; please install transformers>=4.46.0.\n"
                "Run: pip install -r requirements.txt --upgrade --no-cache-dir"
            )
            sys.exit(1)
    except Exception:
        pass
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    model, tokenizer = load_or_wait_for_model()
    
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

        # If model isn't ready, retry load each interval
        if model is None or tokenizer is None:
            model, tokenizer = load_or_wait_for_model()
            time.sleep(TRAINING_INTERVAL)
            continue

        if len(experience_buffer) >= BATCH_SIZE:
            print(f"Training on a new mini-batch of size {BATCH_SIZE}...")
            
            mini_batch_data = list(experience_buffer)
            dpo_dataset = create_dpo_dataset(mini_batch_data)
            
            if HAS_DPO_CONFIG and DPOConfig is not None:
                # Use the modern TRL config API; pass only broadly-supported fields
                dpo_args = DPOConfig(
                    output_dir=CHECKPOINT_DIR,
                    per_device_train_batch_size=BATCH_SIZE,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    logging_steps=1,
                    fp16=True,
                    beta=0.1,
                    report_to=None,
                )
                # Set optional fields only if present on this TRL version
                optional_fields = {
                    "padding_value": -100,
                    "truncation_side": "right",
                    "max_length": 1024,
                    "max_prompt_length": 512,
                    "max_target_length": 512,
                }
                for k, v in optional_fields.items():
                    if hasattr(dpo_args, k):
                        try:
                            setattr(dpo_args, k, v)
                        except Exception:
                            pass
                dpo_trainer = DPOTrainer(
                    model=model,
                    ref_model=None,
                    train_dataset=dpo_dataset,
                    processing_class=tokenizer,
                    args=dpo_args,
                )
            else:
                # Fallback for older TRL versions expecting TrainingArguments
                training_args = TrainingArguments(
                    output_dir=CHECKPOINT_DIR,
                    per_device_train_batch_size=BATCH_SIZE,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    save_strategy="epoch",
                    logging_steps=1,
                    fp16=True,
                    disable_tqdm=True,
                )
                try:
                    dpo_trainer = DPOTrainer(
                        model=model,
                        ref_model=None,
                        train_dataset=dpo_dataset,
                        tokenizer=tokenizer,
                        args=training_args,
                    )
                except TypeError:
                    dpo_trainer = DPOTrainer(
                        model=model,
                        ref_model=None,
                        train_dataset=dpo_dataset,
                        args=training_args,
                    )
            
            dpo_trainer.train()
            dpo_trainer.save_model(os.path.join(CHECKPOINT_DIR, LORA_ADAPTER_NAME))
            print("Updated weights saved.")
            
            # Clear the buffer after successful training to ensure we're always using fresh data
            experience_buffer.clear()

        time.sleep(TRAINING_INTERVAL)
