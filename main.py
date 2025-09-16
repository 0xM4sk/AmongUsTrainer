# !/usr/bin/env python3
# usage: main.py [-h] [--name NAME]

import os
import sys
import asyncio
import random

from typing import Optional, List

sys.path.append(os.path.join(os.path.abspath("."), "among-agents"))

import argparse
import datetime
import subprocess

from amongagents.envs.configs.agent_config import ALL_LLM
from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME, FIVE_MEMBER_GAME
from amongagents.envs.configs.map_config import map_coords
from amongagents.envs.game import AmongUs
from amongagents.UI.MapUI import MapUI
from dotenv import load_dotenv

from utils import setup_experiment

ROOT_PATH = os.path.abspath(".")
LOGS_PATH = os.path.join(ROOT_PATH, "expt-logs")
ASSETS_PATH = os.path.join(ROOT_PATH, "among-agents", "amongagents", "assets")
BLANK_MAP_IMAGE = os.path.join(ASSETS_PATH, "blankmap.png")

load_dotenv()

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
COMMIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

# Default to LiteLLM with qwen:7b-chat via Ollama; can be overridden via env
DEFAULT_LITELLM_MODEL = os.getenv("LITELLM_MODEL", "ollama/qwen:7b-chat")

BIG_LIST_OF_MODELS: List[str] = [
    DEFAULT_LITELLM_MODEL,
    # Additional models supported by LiteLLM (require proper API keys)
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-3.7-sonnet",
    "openai/o3",
    "openai/o4-mini-high",
    "openai/gpt-4o",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-001",
    "google/gemma-3-4b-it",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "x-ai/grok-3-beta",
    "microsoft/phi-4",
]

ARGS = {
    "game_config": SEVEN_MEMBER_GAME,
    "include_human": False,
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        # Default to qwen:7b-chat via LiteLLM; override with CLI or env
        "IMPOSTOR_LLM_CHOICES": [DEFAULT_LITELLM_MODEL],
        "CREWMATE_LLM_CHOICES": [DEFAULT_LITELLM_MODEL],
    },
    "UI": False,
}

async def multiple_games(experiment_name=None, num_games=1, rate_limit=50):
    experiment_name = setup_experiment(experiment_name, LOGS_PATH, DATE, COMMIT_HASH, ARGS)
    ui = MapUI(BLANK_MAP_IMAGE, map_coords, debug=False) if ARGS["UI"] else None
    with open(os.path.join(os.environ["EXPERIMENT_PATH"], "experiment-details.txt"), "a") as experiment_file:
        experiment_file.write(f"\nExperiment args: {ARGS}\n")

    semaphore = asyncio.Semaphore(rate_limit)

    async def run_limited_game(game_index):
        async with semaphore:
            if ARGS.get("tournament_style") == "1on1":
                # Randomly select one model for each role for this specific game
                game_config = ARGS["agent_config"].copy()
                game_config["CREWMATE_LLM_CHOICES"] = [random.choice(BIG_LIST_OF_MODELS)]
                game_config["IMPOSTOR_LLM_CHOICES"] = [random.choice(BIG_LIST_OF_MODELS)]
            else:
                game_config = ARGS["agent_config"]
                
            game = AmongUs(
                game_config=ARGS["game_config"],
                include_human=ARGS["include_human"],
                test=ARGS["test"],
                personality=ARGS["personality"],
                agent_config=game_config,
                UI=ui,
                game_index=game_index,
            )
            await game.run_game()

    tasks = [run_limited_game(i) for i in range(1, num_games+1)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AmongUs experiment.")
    parser.add_argument("--name", type=str, default=None, help="Optional name for the experiment.")
    parser.add_argument("--num_games", type=int, default=2, help="Number of games to run.")
    parser.add_argument("--display_ui", type=bool, default=False, help="Display UI.")
    parser.add_argument("--crewmate_llm", type=str, default=None, help="Crewmate LLM model.")
    parser.add_argument("--impostor_llm", type=str, default=None, help="Impostor LLM model.")
    parser.add_argument("--streamlit", type=bool, default=False, help="Streamlit.")
    parser.add_argument("--tournament_style", type=str, default="random", help="random or 1on1.")
    args = parser.parse_args()
    if args.num_games > 1 or args.display_ui == False:
        ARGS["UI"] = False
    if args.crewmate_llm:
        ARGS["agent_config"]["CREWMATE_LLM_CHOICES"] = [args.crewmate_llm]
    if args.impostor_llm:
        ARGS["agent_config"]["IMPOSTOR_LLM_CHOICES"] = [args.impostor_llm]
    ARGS["tournament_style"] = args.tournament_style
    # Keep Streamlit setting available for utils/setup and other modules
    ARGS["Streamlit"] = bool(args.streamlit)
    asyncio.run(multiple_games(experiment_name=args.name, num_games=args.num_games))
