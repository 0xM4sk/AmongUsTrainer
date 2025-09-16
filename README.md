# *AmongUs*: A Sandbox for Agentic Deception

This project introduces the game "Among Us" as a model organism for lying and deception and studies how AI agents learn to express lying and deception, while evaluating the effectiveness of AI safety techniques to detect and control out-of-distribution deception.

## Overview

The aim is to simulate the popular multiplayer game "Among Us" using AI agents and analyze their behavior, particularly their ability to deceive and lie, which is central to the game's mechanics.

<img src="https://static.wikia.nocookie.net/among-us-wiki/images/f/f5/Among_Us_space_key_art_redesign.png" alt="Among Us" width="400"/>

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/0xM4sk/AmongUsTrainer
   cd AmongUsTrainer
   ```

2. Set up the environment:
   ```bash
   conda create -n amongus python=3.10
   conda activate amongus
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run Games

To run the sandbox and log games of various LLMs playing against each other, run:

```
main.py
```
You will need to add a `.env` file with an [OpenRouter](https://openrouter.ai/) API key.

Alternatively, you can download 400 full-game logs (for `Phi-4-15b` and `Llama-3.3-70b-instruct`) and 810 game summaries from the [HuggingFace](https://huggingface.co/datasets/7vik/AmongUs) dataset to reproduce the results in the paper (and evaluate your own techniques!).

## LiteLLM + Qwen (Default)

This repo uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM calls. By default, both Crewmate and Impostor agents use `ollama/qwen:7b-chat`.

### Option A: Direct to Ollama
- Install and start Ollama, then pull the model:
  - `ollama pull qwen:7b-chat`
- Optional env vars for the game:
  - `LITELLM_MODEL` (default: `ollama/qwen:7b-chat`)
  - `LITELLM_API_BASE` (Ollama default: `http://localhost:11434`)
  - `LITELLM_API_KEY` (not required for Ollama)

### Option B: Via LiteLLM Proxy
Run a local OpenAI-compatible proxy with LiteLLM and route to Ollama.

1) Ensure Ollama is running and the model is available:
   - `ollama pull qwen:7b-chat`

2) Launch the LiteLLM proxy using the provided configuration:
   - `litellm --config config.yaml --port 4000`

3) Point the game to the proxy (OpenAI-compatible endpoint):
   - `export LITELLM_API_BASE=http://localhost:4000`
   - `export LITELLM_MODEL=ollama/qwen:7b-chat`

You can still override models per run using CLI args `--crewmate_llm` and `--impostor_llm` with any LiteLLM-supported `provider/model` string.

## Custom Agent + Continuous Trainer

This repo includes a simple preference data generator (`custom_agent.py`) and a continuous DPO trainer (`continuous_trainer.py`) to fine-tune models based on agent gameplay-style preferences.

### 1) Configure the LLM backend for `custom_agent.py`
- Option A: vLLM (default in `custom_agent.py`)
  - Start vLLM with a chat model, e.g.: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --port 8000`
  - Env vars (optional):
    - `LITELLM_API_BASE` (default: `http://localhost:8000/v1`)
    - `LITELLM_MODEL` (default: `qwen/qwen1.5-7b-chat`)
    - `LITELLM_API_KEY` if your backend requires it
- Option B: Ollama
  - `ollama pull qwen:7b-chat`
  - Set env vars: `export LITELLM_API_BASE=http://localhost:11434` and `export LITELLM_MODEL=ollama/qwen:7b-chat`

### 2) Generate preference data with `custom_agent.py`
`custom_agent.agent_play_turn(game_state)` expects a `game_state` dict like:
```
game_state = {
  "players_alive": ["A", "B", "C"],
  "game_events": [{"event_type": "kill", "player_id": "B"}],
  "discussion_log": [
    {"player_id": "A", "message": "I was in Electrical."},
    {"player_id": "B", "message": "idk"}
  ]
}
```
You can call it from a script/notebook to append JSONL rows to `expt-logs/custom_agent_dataset.jsonl`:
```
from custom_agent import agent_play_turn
message, vote = agent_play_turn(game_state)
```
The file is created automatically if missing.

### 3) Start the continuous DPO trainer
`continuous_trainer.py` tails `expt-logs/custom_agent_dataset.jsonl` and periodically runs DPO fine-tuning on the most recent samples.

- Install training deps (already in `requirements.txt`): `transformers`, `trl`, `peft`, `datasets`, `bitsandbytes`, `accelerate`.
- Ensure GPU drivers and CUDA are available if training on GPU.
- Run: `python continuous_trainer.py`

Environment/config knobs:
- `CUSTOM_AGENT_LOG_FILE` to override the log path (default: `expt-logs/custom_agent_dataset.jsonl`)
- `MODEL_NAME` inside `continuous_trainer.py` to change the base model for training
- Checkpoints saved under `./dpo_checkpoints/dpo_adapter`

Integration note: The custom agent and trainer are decoupled from the AmongUs gameplay loop. Use `custom_agent.agent_play_turn()` within your own loop or data collection process to generate preference pairs, then run the trainer in parallel to adapt the model continuously.

## Deception ELO

After running (or downloading) the games, to reproduce our Deception ELO results, run the following notebook:

```
reports/2025_02_26_deception_ELO_v3_ci.ipynb
```

The other report files can be used to reproduce the respective results.

## Caching Activations

Once the (full) game logs are in place, use the following command to cache the activations of the LLMs:

```
python linear-probes/cache_activations.py --dataset <dataset_name>
```

This loads up the HuggingFace models and caches the activations of the specified layers for each game action step. This step is computationally expensive, so it is recommended to run this using GPUs.

Use `configs.py` to specify the model and layer to cache, and other configuration options.

## LLM-based Evaluation (for Lying, Awareness, Deception, and Planning)

To evaluate the game actions by passing agent outputs to an LLM, run:

```
bash evaluations/run_evals.sh
```
You will need to add a `.env` file with an OpenAI API key.

Alternatively, you can download the ground truth labels from the [HuggingFace](https://huggingface.co/datasets/7vik/AmongUs).

(TODO)

## Training Linear Probes

Once the activations are cached, training linear probes is easy. Just run:

```
python linear-probes/train_all_probes.py
```
You can choose which datasets to train probes on - by default, it will train on all datasets.

## Evaluating Linear Probes

To evaluate the linear probes, run:

```
python linear-probes/eval_all_probes.py
```
You can choose which datasets to evaluate probes on - by default, it will evaluate on all datasets.

It will store the results in `linear-probes/results/`, which are used to generate the plots in the paper.

## Sparse Autoencoders (SAEs)

We use the [Goodfire API](https://goodfire.ai/) to evaluate SAE features on the game logs. To do this, run the notebook:

```
reports/2025_02_27_sparse_autoencoders.ipynb
```
You will need to add a `.env` file with a Goodfire API key.

## Project Structure

```plaintext
.
├── CONTRIBUTING.md         # Contribution guidelines
├── Dockerfile               # Docker setup for project environment
├── LICENSE                  # License information
├── README.md                # Project documentation (this file)
├── among-agents             # Main code for the Among Us agents
│   ├── README.md            # Documentation for agent implementation
│   ├── amongagents          # Core agent and environment modules
│   ├── envs                 # Game environment and configurations
│   ├── evaluation           # Evaluation scripts for agent performance
│   ├── notebooks            # Jupyter notebooks for running experiments
│   ├── requirements.txt     # Python dependencies for agents
│   └── setup.py             # Setup script for agent package
├── expt-logs                # Experiment logs
├── k8s                      # Kubernetes configurations for deployment
├── main.py                  # Main entry point for running the game
├── notebooks                # Additional notebooks (not part of the main project)
├── reports                  # Experiment reports
├── requirements.txt         # Python dependencies for main project
├── tests                    # Unit tests for project functionality
└── utils.py                 # Utility functions
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under CC0 1.0 Universal - see [LICENSE](LICENSE).

## Acknowledgments

- Our game logic uses a bunch of code from [AmongAgents](https://github.com/cyzus/among-agents).

- Forked from Satvik Golechha (7vik) (https://github.com/7vik/AmongUs)
