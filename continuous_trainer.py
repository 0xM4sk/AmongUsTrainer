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
    base_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure consistent padding/truncation sides for chat templates
    try:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        # Cap tokenizer max length to model capacity
        model_max_len = getattr(base_model.config, "max_position_embeddings", None)
        if model_max_len is not None:
            tokenizer.model_max_length = int(model_max_len)
    except Exception:
        pass

    # Ensure model embeddings cover tokenizer vocab (avoid OOB indices on CUDA)
    try:
        model_vocab = base_model.get_input_embeddings().weight.shape[0]
        tok_vocab = len(tokenizer)
        if tok_vocab > model_vocab:
            print(f"Resizing token embeddings: model_vocab={model_vocab} -> tok_vocab={tok_vocab}")
            base_model.resize_token_embeddings(tok_vocab)
    except Exception as e:
        print(f"Warning: could not verify/resize token embeddings: {e}")

    # Sanity check tokenizer/model vocab agreement
    try:
        model_vocab = getattr(base_model.config, "vocab_size", None)
        tok_vocab = getattr(tokenizer, "vocab_size", None)
        if model_vocab is not None and tok_vocab is not None and tok_vocab > model_vocab:
            print(f"Warning: tokenizer vocab ({tok_vocab}) > model vocab ({model_vocab}). This can cause index errors.")
    except Exception:
        pass

    print("Initializing or loading LoRA adapter...")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"])
    adapter_path = os.path.join(CHECKPOINT_DIR, LORA_ADAPTER_NAME)
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = get_peft_model(base_model, lora_config)

    class _SafeForward(torch.nn.Module):
        def __init__(self, inner_model):
            super().__init__()
            self.inner = inner_model
            try:
                self.max_token_id = self.inner.get_input_embeddings().weight.shape[0] - 1
            except Exception:
                self.max_token_id = None

        # Proxy attribute access (e.g., .config, .generate) to the inner model
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
            try:
                if input_ids is not None and self.max_token_id is not None:
                    with torch.no_grad():
                        max_id = torch.max(input_ids).item()
                        min_id = torch.min(input_ids).item()
                        if max_id > self.max_token_id or min_id < 0:
                            print(f"[SafeForward] Clamping token ids (min={min_id}, max={max_id}, limit={self.max_token_id})")
                        input_ids = torch.clamp(input_ids, 0, self.max_token_id)
                if attention_mask is not None:
                    attention_mask = (attention_mask > 0).to(dtype=attention_mask.dtype)
                if position_ids is not None and input_ids is not None:
                    seq_len = input_ids.shape[-1]
                    position_ids = torch.clamp(position_ids, 0, max(seq_len - 1, 0))
            except Exception as e:
                print(f"[SafeForward] Warning during input sanitation: {e}")
            return self.inner(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    # Wrap model to guard against OOB indices causing CUDA asserts
    model = _SafeForward(model)

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
    # Optional debug: enable CUDA_LAUNCH_BLOCKING to get precise stacktraces on CUDA errors
    if os.getenv("DPO_DEBUG", "0") in {"1", "true", "True"}:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
                safe_mode = os.getenv("DPO_SAFE_MODE", "0") in {"1", "true", "True"}
                per_device_bs = 1 if safe_mode else BATCH_SIZE
                use_fp16 = False if safe_mode else True
                # Derive conservative lengths from model capacity
                model_max_len = getattr(model.config, "max_position_embeddings", 2048)
                desired_max_len = min(1024, int(model_max_len))
                desired_prompt_len = min(512, desired_max_len // 2)
                desired_target_len = desired_max_len - desired_prompt_len
                dpo_args = DPOConfig(
                    output_dir=CHECKPOINT_DIR,
                    per_device_train_batch_size=per_device_bs,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    logging_steps=1,
                    fp16=use_fp16,
                    beta=0.1,
                    report_to=None,
                )
                # Set optional fields only if present on this TRL version
                optional_fields = {
                    "padding_value": -100,
                    "label_pad_token_id": -100,
                    "truncation_side": "right",
                    "max_length": desired_max_len,
                    "max_prompt_length": desired_prompt_len,
                    "max_target_length": desired_target_len,
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
                safe_mode = os.getenv("DPO_SAFE_MODE", "0") in {"1", "true", "True"}
                per_device_bs = 1 if safe_mode else BATCH_SIZE
                use_fp16 = False if safe_mode else True
                training_args = TrainingArguments(
                    output_dir=CHECKPOINT_DIR,
                    per_device_train_batch_size=per_device_bs,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    save_strategy="epoch",
                    logging_steps=1,
                    fp16=use_fp16,
                    disable_tqdm=True,
                )
                # Avoid column pruning which can interfere with TRL processors
                if not hasattr(training_args, "remove_unused_columns"):
                    try:
                        setattr(training_args, "remove_unused_columns", False)
                    except Exception:
                        pass
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
