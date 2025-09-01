#!/usr/bin/env python
"""
Distributed training script for GRPO on the Attack Graph Challenge.
"""

import copy
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import (GRPOConfig, GRPOTrainer, ModelConfig, TrlParser,
                 get_peft_config)

# Enable HF Hub fast downloads BEFORE imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

#####################################################################
# 1. Script Configuration & Dataclasses
#####################################################################

@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "meowterspace45/attack-graph-challenge"
    dataset_splits: str = "train"
    dataset_config_name: str = "simple"
    tokenizer_name_or_path: str = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
logger.propagate = False

#####################################################################
# 2. Stateful Simulator & Reward Functions
#####################################################################

def parse_actions_from_answer(answer_text: str) -> list:
    """Extract structured actions like tool(arg1, arg2)."""
    action_regex = re.compile(r"(\w+)\(([^)]+)\)")
    parsed_actions = []
    for match in action_regex.finditer(answer_text):
        tool, args = match.group(1).strip(), [
            arg.strip() for arg in match.group(2).split(",")
        ]
        parsed_actions.append({"tool": tool, "args": args})
    return parsed_actions

def run_simulation(actions: list, initial_state: dict, target: str) -> dict:
    """
    Simulation:
    - scan(host): enumerate vulns, discover host if in scenario
    - exploit(host, cve): admin if vuln matches
    - read_file(host, file): requires known host and admin or misconfig
    - dump_creds(host): requires admin, yields creds
    - use_creds(host, creds): requires known host and known creds
    Success = admin on `target`
    """

    sim_state = copy.deepcopy(initial_state["agent_start"])
    scenario = initial_state["scenario"]
    total_reward, is_success, trace = 0.0, False, []

    for i, action in enumerate(actions):
        tool, args = action.get("tool"), action.get("args", [])
        action_str = f"{tool}({', '.join(args)})"
        step_result = {
            "step": i + 1,
            "action": action_str,
            "is_valid": False,
            "outcome": "Action invalid or failed.",
            "step_reward": -0.1,
        }

        is_valid_action = False
        target_host = args[0] if args else ""

        # --- Action handlers ---
        if tool == "scan" and len(args) == 1:
            if target_host in sim_state.get("known_hosts", []):
                is_valid_action = True
                vulns = scenario.get("vulnerabilities", {}).get(target_host)
                if vulns:
                    sim_state.setdefault("known_vulns", {})[target_host] = vulns
                    step_result.update(
                        outcome=f"Scan successful. Found: {vulns}", step_reward=0.1
                    )
                else:
                    step_result.update(
                        outcome="Scan successful. No new vulnerabilities found.",
                        step_reward=0.05,
                    )
                # Discover linked hosts with a 20% chance
                links = scenario.get("network_links", {}).get(target_host, [])
                if links:
                    for new_host in links:
                        if new_host not in sim_state.get("known_hosts", []):
                            if random.random() < 0.2:
                                sim_state.setdefault("known_hosts", []).append(new_host)
                                step_result.update(
                                    outcome=f"{step_result['outcome']} Also discovered host {new_host}.",
                                    step_reward=step_result["step_reward"] + 0.2,
                                )

        elif tool == "exploit" and len(args) == 2:
            cve = args[1]
            if (
                sim_state.get("known_vulns", {})
                .get(target_host, {})
                .get("cve") == cve
            ):
                is_valid_action = True
                sim_state.setdefault("access", {})[target_host] = "admin"
                step_result.update(
                    outcome=f"Exploit successful. Admin access on {target_host}.",
                    step_reward=0.5,
                )

        elif tool == "read_file" and len(args) == 2:
            file_path = args[1]
            has_admin = sim_state.get("access", {}).get(target_host) == "admin"
            has_misconfig = (
                sim_state.get("known_vulns", {})
                .get(target_host, {})
                .get("misconfiguration")
            )
            if has_admin or has_misconfig:
                is_valid_action = True
                creds_for = (
                    scenario.get("secrets", {})
                    .get(target_host, {})
                    .get("contains_creds_for")
                )
                if creds_for:
                    sim_state.setdefault("known_creds", []).append(
                        f"creds_for_{creds_for}"
                    )
                    sim_state.setdefault("known_hosts", []).append(creds_for)
                    step_result.update(
                        outcome=f"Read {file_path} successful. Found credentials for {creds_for}.",
                        step_reward=0.3,
                    )
                else:
                    step_result.update(
                        outcome=f"Read {file_path} successful, but no useful credentials found.",
                        step_reward=0.1,
                    )

        elif tool == "dump_creds" and len(args) == 1:
            if sim_state.get("access", {}).get(target_host) == "admin":
                is_valid_action = True
                creds_for = (
                    scenario.get("secrets", {})
                    .get(target_host, {})
                    .get("contains_creds_for")
                )
                if creds_for:
                    sim_state.setdefault("known_creds", []).append(
                        f"creds_for_{creds_for}"
                    )
                    sim_state.setdefault("known_hosts", []).append(creds_for)
                    step_result.update(
                        outcome=f"Dumped creds for {creds_for}.",
                        step_reward=0.4,
                    )
                else:
                    step_result.update(
                        outcome="Dumped creds, but nothing useful found.",
                        step_reward=0.1,
                    )

        elif tool == "use_creds" and len(args) == 2:
            creds = args[1]
            if creds in sim_state.get("known_creds", []):
                is_valid_action = True
                sim_state.setdefault("access", {})[target_host] = "user"
                step_result.update(
                    outcome=f"Used {creds}. Gained user access on {target_host}.",
                    step_reward=0.2,
                )

        # --- Apply result ---
        if is_valid_action:
            step_result["is_valid"] = True
            total_reward += step_result["step_reward"]
        else:
            total_reward += step_result["step_reward"]

        trace.append(step_result)

        # Check for success
        if sim_state.get("access", {}).get(target) == "admin":
            is_success = True
            total_reward += 1.0
            break

    return {"final_reward": max(total_reward, 0.0), "is_success": is_success, "trace": trace}


_step_counter = 0
_win_loss_stats = {"wins": 0, "losses": 0, "total": 0}


def attack_success_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Simulation-based reward function."""
    global _step_counter, _win_loss_stats
    prompts = kwargs.get("prompts") or [""] * len(completions)
    rewards = []
    batch_wins, batch_total = 0, len(completions)

    for i in range(batch_total):
        completion = completions[i]
        prompt = prompts[i]
        try:
            prompt_parts = prompt.split("---END_OF_PROMPT---")
            if len(prompt_parts) < 2:
                rewards.append(0.0)
                continue
            initial_state = json.loads(prompt_parts[1])

            full_completion = "<think>" + completion
            answer_match = re.search(r"<answer>([\s\S]*?)<\/answer>", full_completion)
            if not answer_match:
                rewards.append(0.0)
                continue

            actions = parse_actions_from_answer(answer_match.group(1).strip())
            if not actions:
                rewards.append(0.0)
                continue

            sim_result = run_simulation(actions, initial_state, target="DomainController")
            rewards.append(sim_result["final_reward"])

            if sim_result["is_success"]:
                batch_wins += 1
                if os.environ.get("RANK", "0") == "0" and random.random() < 0.1:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "simulation_traces.jsonl")
                    with open(log_file, "a") as f:
                        log_entry = {
                            "completion": full_completion,
                            "initial_state": initial_state,
                            "simulation_trace": sim_result["trace"],
                            "final_reward": sim_result["final_reward"],
                            "final_state": sim_result["final_state"],
                        }
                        f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.debug(f"Error in reward function: {e}")
            rewards.append(0.0)

    _win_loss_stats["wins"] += batch_wins
    _win_loss_stats["losses"] += batch_total - batch_wins
    _win_loss_stats["total"] += batch_total
    _step_counter += 1
    return rewards


#####################################################################
# 3. Custom Callback for Logging Metrics
#####################################################################

class AttackGraphMetricsCallback(TrainerCallback):
    """Custom callback to log win/loss statistics during GRPO training."""
    def __init__(self):
        self.win_loss_stats = {"wins": 0, "losses": 0, "total": 0}

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        global _win_loss_stats
        current_total = _win_loss_stats["total"]
        if current_total > self.win_loss_stats["total"]:
            new_total = current_total - self.win_loss_stats["total"]
            new_wins = _win_loss_stats["wins"] - self.win_loss_stats["wins"]
            if new_total > 0 and logs is not None:
                overall_win_rate = _win_loss_stats["wins"] / current_total
                recent_win_rate = new_wins / new_total
                logs.update({
                    "attack_success/overall_win_rate": overall_win_rate,
                    "attack_success/recent_win_rate": recent_win_rate,
                    "attack_success/total_wins": _win_loss_stats["wins"],
                    "attack_success/total_attempts": current_total,
                })
                logger.info(
                    f"Overall Win Rate: {overall_win_rate:.1%} | "
                    f"Recent: {recent_win_rate:.1%} | "
                    f"Total Wins: {_win_loss_stats['wins']}/{current_total}"
                )
            self.win_loss_stats = _win_loss_stats.copy()


#####################################################################
# 4. Main Training Orchestration
#####################################################################

def get_checkpoint(training_args: GRPOConfig):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def grpo_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        script_args.dataset_id_or_path,
        name=script_args.dataset_config_name,
        split=script_args.dataset_splits,
    )
    n = min(len(dataset), 50_000)
    dataset = dataset.shuffle(seed=42).select(range(n))

    def generate_attack_prompt(target, initial_state_str):
        initial_state = json.loads(initial_state_str)
        agent_access, known_hosts = (
            initial_state["agent_start"]["access"],
            initial_state["agent_start"]["known_hosts"],
        )
        start_host, start_access = next(iter(agent_access.items()))
        scenario_desc = f"""You are a cybersecurity analyst conducting an authorized penetration test. 
Your goal is to achieve admin access on the DomainController.

Current situation:
- You have {start_access} access on {start_host}
- Known hosts: {', '.join(known_hosts)}
- Network contains: WebServer, DBServer, DomainController

Available actions:
- scan(host)
- exploit(host, cve)
- read_file(host, file)
- dump_creds(host)
- use_creds(host, creds)

Plan your attack step by step to achieve the goal."""
        r1_prefix = [
            {
                "role": "system",
                "content": "You are a cybersecurity expert. You think systematically and provide detailed attack plans.",
            },
            {
                "role": "user",
                "content": scenario_desc
                + " Show your reasoning in <think> </think> tags, then provide your attack plan in <answer> </answer> tags.",
            },
            {
                "role": "assistant",
                "content": "I'll analyze this penetration testing scenario step by step.\n<think>",
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, add_generation_prompt=True
        )
        combined_prompt = f"{prompt_text}---END_OF_PROMPT---{initial_state_str}"
        return {"prompt": combined_prompt}

    dataset = dataset.map(
        lambda x: generate_attack_prompt(x["target"], x["initial_state"]),
        remove_columns=["target", "initial_state"],
    )

    split = dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = split["train"], split["test"]

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[attack_success_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[AttackGraphMetricsCallback()],
    )

    last_checkpoint = get_checkpoint(training_args)
    logger.info(
        f'*** Starting Attack Graph GRPO training at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ***'
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("*** Training complete, saving model... ***")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.create_model_card()
    if training_args.push_to_hub:
        trainer.push_to_hub()
    logger.info("*** Attack Graph GRPO training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
