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

########################
# Attack Graph Reward Functions
########################

def format_reward_func(completions, target, **kwargs):
    """
    Rewards completions that follow the correct format:
    <think>...</think><answer>...</answer>
    
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected targets (always "Domain Admin Access")
      
    Returns:
        list[float]: Reward scores (1.0 for correct format, 0.0 otherwise)
    """
    rewards = []

    for completion, gt in zip(completions, target):
        try:
            # Add synthetic <think> as it's already part of the prompt and prefilled
            completion = "<think>" + completion
            
            if random.random() < 0.01:  # 1% chance to log samples
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
            
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL) 
            
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def attack_success_reward_func(completions, target, initial_states, **kwargs):
    """
    Rewards completions that successfully achieve Domain Admin Access.
    
    This is a simplified reward function that checks if the completion:
    1. Follows the correct action sequence format
    2. Contains the necessary steps to achieve the goal
    3. Ends with gaining Domain Admin Access
    
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected targets (always "Domain Admin Access")
        initial_states (list[str]): JSON strings of initial game states
    
    Returns:
        list[float]: Reward scores (1.0 for success, 0.0 for failure)
    """
    rewards = []
    
    for completion, gt, initial_state_str in zip(completions, target, initial_states):
        try:
            # Add synthetic <think> as it's already part of the prompt
            completion = "<think>" + completion
            
            # Parse the initial state
            initial_state = json.loads(initial_state_str)
            solution_path = initial_state.get("metadata", {}).get("solution_path", "")
            
            # Extract the answer section
            answer_match = re.search(r"<answer>([\s\S]*?)<\/answer>", completion)
            if not answer_match:
                rewards.append(0.0)
                continue
                
            answer = answer_match.group(1).strip()
            
            # Simple heuristic-based reward function
            # In a full implementation, you'd simulate the environment
            reward = calculate_attack_reward(answer, solution_path, initial_state)
            rewards.append(reward)
            
            if reward > 0.8 and random.random() < 0.10:  # 10% chance to log successful attempts
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n============== SUCCESS (Reward: {reward}) ==============\n")
                    f.write(f"Solution Path: {solution_path}\n")
                    f.write(completion)
                    
        except Exception as e:
            logger.debug(f"Error in attack_success_reward_func: {e}")
            rewards.append(0.0)
            
    return rewards

def calculate_attack_reward(answer: str, solution_path: str, initial_state: dict) -> float:
    """
    Calculate reward based on the quality of the attack sequence.
    
    This is a simplified heuristic function. In a full implementation,
    you would simulate the environment step by step.
    """
    answer_lower = answer.lower()
    
    # Base reward for attempting actions
    reward = 0.0
    
    # Check for key cybersecurity actions
    actions_found = 0
    required_actions = ["scan", "exploit", "read_file", "dump_creds"]
    
    for action in required_actions:
        if action in answer_lower:
            actions_found += 1
    
    reward += actions_found * 0.15  # 0.15 per action found
    
    # Bonus for following a logical sequence based on solution path
    if solution_path == "exploit_path":
        if "exploit" in answer_lower and "sqli" in answer_lower:
            reward += 0.2
        if "privesc" in answer_lower:
            reward += 0.2
    elif solution_path == "misconfig_path":
        if "config" in answer_lower and "readable" in answer_lower:
            reward += 0.2
        if "password" in answer_lower and "weak" in answer_lower:
            reward += 0.2
    
    # Bonus for mentioning the goal
    if "domain admin" in answer_lower or "domaincontroller" in answer_lower:
        reward += 0.2
    
    # Bonus for sequential thinking (multiple steps)
    steps = answer_lower.count("step") + answer_lower.count("then") + answer_lower.count("next")
    reward += min(steps * 0.05, 0.15)  # Up to 0.15 bonus for showing steps
    
    return min(reward, 1.0)  # Cap at 1.0

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
        reward_funcs=[format_reward_func, attack_success_reward_func],
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