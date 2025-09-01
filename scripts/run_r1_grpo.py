#!/usr/bin/env python
"""
Distributed training script for GRPO on the Attack Graph Challenge.
Adapted from the TinyZero project and Phil Schmid's DeepSeek R1 GRPO implementation. 
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
import random
import re
import json
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

# Enable HF transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "meowterspace45/attack-graph-challenge"
    dataset_splits: str = "train"
    dataset_config_name: str = "simple"  # simple or complex
    tokenizer_name_or_path: str = None

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


#####################################################################
# Attack Graph Reward Functions (Stateful Simulator Implementation)
#####################################################################
import re
import json
import copy 

def parse_actions_from_answer(answer_text: str) -> list:
    """
    Parses an LLM's free-form text answer into a structured list of actions.
    Example: "First I will scan(WebServer)." -> [{'tool': 'scan', 'args': ['WebServer']}]
    """
    # Regex to find function-like calls, e.g., scan(WebServer) or exploit(DBServer, PrivEsc-CVE)
    action_regex = re.compile(r"(\w+)\(([^)]+)\)")
    parsed_actions = []
    for match in action_regex.finditer(answer_text):
        tool = match.group(1).strip()
        # Split args by comma and strip whitespace
        args = [arg.strip() for arg in match.group(2).split(',')]
        parsed_actions.append({"tool": tool, "args": args})
    return parsed_actions

def run_simulation(actions: list, initial_state: dict) -> dict:
    """
    Simulates the game step-by-step based on a list of parsed actions.
    This is the core of the stateful reward function.
    """
    sim_state = copy.deepcopy(initial_state["agent_start"])
    scenario = initial_state["scenario"]
    
    total_reward = 0.0
    is_success = False
    trace = []

    for i, action in enumerate(actions):
        tool = action.get("tool")
        args = action.get("args", [])
        action_str = f"{tool}({', '.join(args)})"
        step_result = {
            "step": i + 1, "action": action_str, "is_valid": False,
            "outcome": "Action invalid or failed.", "step_reward": 0.0
        }

        # --- Action Validity & State Update Logic ---
        is_valid_action = False
        target = args[0] if args else ""

        # Check if the target is known (prevents guessing)
        if target and target not in sim_state["known_hosts"]:
            step_result["outcome"] = f"Action failed: Host '{target}' is unknown."
            trace.append(step_result)
            total_reward -= 0.5 # Penalty for acting on unknown hosts
            break

        if tool == "scan" and len(args) == 1:
            is_valid_action = True
            # Reveal vulnerabilities or misconfigs on the scanned host
            vulns = scenario.get("vulnerabilities", {}).get(target)
            if vulns:
                sim_state.setdefault("known_vulns", {})[target] = vulns
                step_result["outcome"] = f"Scan successful. Found: {vulns}"
                step_result["step_reward"] = 0.1
            else:
                step_result["outcome"] = "Scan successful. No new vulnerabilities found."
                step_result["step_reward"] = 0.05 # Small reward for exploring
        
        elif tool == "exploit" and len(args) == 2:
            cve_arg = args[1]
            # Check if host is known to be vulnerable to this CVE
            if sim_state.get("known_vulns", {}).get(target, {}).get("cve") == cve_arg:
                is_valid_action = True
                sim_state.setdefault("access", {})[target] = "admin"
                step_result["outcome"] = f"Exploit successful. Gained admin access on {target}."
                step_result["step_reward"] = 0.5 # High reward for privesc
        
        elif tool == "read_file" and len(args) == 1:
             # In our scenarios, this is tied to finding a misconfig
            if sim_state.get("known_vulns", {}).get(target, {}).get("misconfiguration"):
                 is_valid_action = True
                 creds_for = scenario.get("secrets", {}).get(target, {}).get("contains_creds_for")
                 if creds_for:
                     sim_state.setdefault("known_creds", []).append(f"creds_for_{creds_for}")
                     sim_state.setdefault("known_hosts", []).append(creds_for)
                     step_result["outcome"] = f"File read successful. Found credentials for {creds_for}."
                     step_result["step_reward"] = 0.3 # High reward for finding creds
        
        elif tool == "dump_creds" and len(args) == 1:
            # Requires admin access
            if sim_state.get("access", {}).get(target) == "admin":
                is_valid_action = True
                creds_for = scenario.get("secrets", {}).get(target, {}).get("contains_creds_for")
                if creds_for:
                    sim_state.setdefault("known_creds", []).append(f"creds_for_{creds_for}")
                    sim_state.setdefault("known_hosts", []).append(creds_for)
                    step_result["outcome"] = f"Credential dump successful. Found credentials for {creds_for}."
                    step_result["step_reward"] = 0.4 # High reward for finding creds
        
        # Add other tool logic here (e.g., password_spray) if needed

        if is_valid_action:
            step_result["is_valid"] = True
            total_reward += step_result["step_reward"]
            trace.append(step_result)
        else:
            trace.append(step_result)
            total_reward -= 0.5 # Heavy penalty for invalid actions
            break # Stop simulation on first error

    # --- Final Goal Check ---
    if sim_state.get("access", {}).get("DomainController") == "admin":
        is_success = True
        total_reward += 1.0 # Big bonus for winning the game
        
    return {
        "final_reward": max(total_reward, 0.0), # Ensure reward is non-negative
        "is_success": is_success,
        "trace": trace
    }

def attack_success_reward_func(completions, target, initial_states, **kwargs):
    """
    Orchestrates the simulation-based reward calculation and logging.
    This function replaces the old heuristic-based reward function.
    """
    rewards = []
    
    for completion, initial_state_str in zip(completions, initial_states):
        try:
            # Add synthetic <think> as it's part of the prompt
            full_completion = "<think>" + completion
            
            # 1. First, check for basic format compliance
            answer_match = re.search(r"<answer>([\s\S]*?)<\/answer>", full_completion)
            if not answer_match:
                rewards.append(0.0)
                continue
            answer_text = answer_match.group(1).strip()

            # 2. Parse the plan from the answer
            actions = parse_actions_from_answer(answer_text)
            if not actions: # Penalize if no parsable actions are found
                rewards.append(0.0)
                continue

            # 3. Run the simulation
            initial_state = json.loads(initial_state_str)
            sim_result = run_simulation(actions, initial_state)
            
            # 4. Append the final reward
            rewards.append(sim_result["final_reward"])

            # 5. Log successful traces for later analysis
            if sim_result["is_success"]:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "simulation_traces.jsonl")
                with open(log_file, "a") as f:
                    log_entry = {
                        "completion": full_completion,
                        "initial_state": initial_state,
                        "simulation_trace": sim_result["trace"],
                        "final_reward": sim_result["final_reward"]
                    }
                    f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            # logger is defined in your main script
            # logger.debug(f"Error in reward function: {e}")
            rewards.append(0.0)
            
    return rewards

########################
# Helper functions
########################

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    logger.info(f"Loading dataset: {script_args.dataset_id_or_path}")
    logger.info(f"Dataset config: {script_args.dataset_config_name}")
    
    dataset = load_dataset(
        script_args.dataset_id_or_path, 
        name=script_args.dataset_config_name,
        split=script_args.dataset_splits
    )
    
    # Select a random subset of 50k samples (like the original)
    dataset = dataset.shuffle(seed=42).select(range(50000))
    logger.info(f"Using {len(dataset)} samples for training")

    #####################
    # Prepare and format dataset
    #####################

    def generate_attack_prompt(target, initial_state_str):
        """
        Generate R1-style prompt for attack graph scenarios
        """
        initial_state = json.loads(initial_state_str)
        
        # Extract key information for the prompt
        agent_access = initial_state["agent_start"]["access"]
        known_hosts = initial_state["agent_start"]["known_hosts"]
        
        # Create a more detailed scenario description
        scenario_desc = f"""You are a cybersecurity analyst conducting an authorized penetration test. 
Your goal is to achieve {target} on the network.

Current situation:
- You have {list(agent_access.values())[0]} access on {list(agent_access.keys())[0]}
- Known hosts: {', '.join(known_hosts)}
- Network contains: WebServer, DBServer, DomainController

Available actions:
- scan(host): Discover vulnerabilities and files on a host
- exploit(host, cve): Exploit a CVE to gain admin access
- read_file(host, file): Read configuration files for credentials
- password_spray(host, creds): Use credentials to access services
- dump_creds(host): Extract credentials (requires admin access)

Plan your attack step by step to achieve the goal."""

        r1_prefix = [
            {
                "role": "system",
                "content": "You are a cybersecurity expert conducting authorized penetration testing. You think through your approach systematically and provide detailed attack plans."
            },
            { 
                "role": "user",
                "content": scenario_desc + " Show your reasoning in <think> </think> tags, then provide your attack plan in <answer> </answer> tags."
            },
            {
                "role": "assistant",
                "content": "I'll analyze this penetration testing scenario step by step.\n<think>"
            }
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
            "target": target, 
            "initial_state": initial_state_str
        }

    # Convert our dataset to the attack graph prompt format
    dataset = dataset.map(lambda x: generate_attack_prompt(x["target"], x["initial_state"]))

    # Split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    #########################
    # Instantiate GRPO trainer
    #########################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[attack_success_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting Attack Graph GRPO training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({
            "tags": ["rl", "grpo", "cybersecurity", "attack-graph", "conference-demo"],
            "dataset": script_args.dataset_id_or_path,
            "base_model": model_args.model_name_or_path
        })
        
    # Push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Attack Graph GRPO training complete! ***")

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)

if __name__ == "__main__":
    main()