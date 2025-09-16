import json
import os
import random
import litellm
from copy import deepcopy

# --- Configuration ---
# Backend server + model configuration (env-driven with sensible defaults)
# Default integrates with repo's LiteLLM proxy config.yaml at port 4000
# Option A: LiteLLM proxy (default) -> api_base=http://localhost:4000, model=ollama/qwen:7b-chat
# Option B: vLLM -> api_base=http://localhost:8000/v1, model=qwen/qwen1.5-7b-chat
# Option C: Ollama direct -> api_base=http://localhost:11434, model=ollama/qwen:7b-chat
api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
model = os.getenv("LITELLM_MODEL", "ollama/qwen:7b-chat")
api_key = os.getenv("LITELLM_API_KEY")
client = litellm.completion

# Path to the log file for the continuous trainer
LOG_FILE_PATH = os.path.join(os.getcwd(), "expt-logs", "custom_agent_dataset.jsonl")
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# --- Heuristic Logic (As per previous discussions) ---
def create_intuitive_belief_state(game_state):
    """
    Parses the game state to create a simple, factual belief state.
    """
    intuitive_state = {
        "players_alive": game_state["players_alive"],
        "game_events": game_state["game_events"],
        "discussion_summary": " ".join([d["message"] for d in game_state["discussion_log"]]),
    }
    return intuitive_state

def create_critical_belief_state(game_state, intuitive_state):
    """
    Performs heuristic-based reasoning to find contradictions and
    subtext, creating a 'slow thinking' belief state.
    """
    critical_state = {
        "contradictions_found": [],
        "suspicion_scores": {}
    }
    
    for player_id in intuitive_state["players_alive"]:
        critical_state["suspicion_scores"][player_id] = 0.0

    for event in intuitive_state["game_events"]:
        if event["event_type"] == "kill":
            critical_state["suspicion_scores"][event["player_id"]] += 0.5
    
    # Example rule for a 'rejected' action
    for discussion in game_state["discussion_log"]:
        if len(discussion["message"]) < 20 and ("shutup" in discussion["message"].lower() or "idk" in discussion["message"].lower()):
            critical_state["suspicion_scores"][discussion["player_id"]] += 0.3

    total_suspicion = sum(critical_state["suspicion_scores"].values())
    if total_suspicion > 0:
        for player_id in critical_state["suspicion_scores"]:
            critical_state["suspicion_scores"][player_id] /= total_suspicion

    return critical_state

# --- New Prompt for Preference Pair Generation ---
def generate_preference_pair(game_state, intuitive_state, critical_state):
    """
    Prompts the LLM to generate both a strategically chosen and a rejected response.
    """
    prompt = f"""
    ### Role and Mission
    You are an advanced AI agent playing a social deduction game. Your goal is to win. You must generate a persuasive message and a strategic vote.

    ### Input Context
    <game_state>
    {json.dumps(game_state, indent=2)}
    </game_state>

    ### Dual-Process Analysis
    Your task is to produce a two-part analysis before generating your final action.

    <analysis>
    **Intuitive Lens (Facts):** {json.dumps(intuitive_state)}
    **Critical Lens (Subtext):** {json.dumps(critical_state)}
    </analysis>

    ### Output Format
    Your final output must be a single JSON object containing two responses.
    The `chosen` response should be the most strategically effective message and vote based on your analysis.
    The `rejected` response should be a plausible but strategically weak alternative. It should be realistic but perhaps too aggressive, too timid, or a truthful statement at a time when a lie is needed.
    
    {{
        "chosen": {{
            "message": "Your strategically effective message.",
            "vote": "The Player ID of who you are voting for."
        }},
        "rejected": {{
            "message": "Your strategically weak message.",
            "vote": "The Player ID of an alternative vote."
        }}
    }}
    """
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "api_base": api_base,
            "response_format": {"type": "json_object"},
        }
        if api_key:
            kwargs["api_key"] = api_key
        response = client(**kwargs)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating response: {e}")
        # Return a fallback for a failed API call
        return {
            "chosen": {"message": "I'm thinking...", "vote": ""},
            "rejected": {"message": "I don't know.", "vote": ""}
        }

# --- Main Agent Logic ---
def agent_play_turn(game_state):
    intuitive_state = create_intuitive_belief_state(game_state)
    critical_state = create_critical_belief_state(game_state, intuitive_state)
    
    preference_pair = generate_preference_pair(game_state, intuitive_state, critical_state)
    
    # Log the full triplet for training
    with open(LOG_FILE_PATH, "a") as f:
        turn_data = {
            "game_state_raw": game_state,
            "belief_state_intuitive": intuitive_state,
            "belief_state_critical": critical_state,
            "chosen_response": preference_pair.get("chosen"),
            "rejected_response": preference_pair.get("rejected")
        }
        f.write(json.dumps(turn_data) + "\n")
        
    message = preference_pair["chosen"]["message"]
    vote = preference_pair["chosen"]["vote"]
    
    return message, vote
