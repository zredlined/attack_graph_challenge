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
    action_regex = re.compile(r"(\w+)\(([^)]*)\)")  # Updated regex for no-arg tools
    parsed_actions = []
    for match in action_regex.finditer(answer_text):
        tool, args_str = match.group(1).strip(), match.group(2).strip()
        args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
        parsed_actions.append({"tool": tool, "args": args})
    return parsed_actions


def run_simulation(actions: list, initial_state: dict, target: str) -> dict:
    """
    Stateful simulation for attack graph, designed to handle multi-step chains.
    Rewards:
    - scan(host): first-time discovery
    - exploit(host, cve): grants admin if scan discovered CVE
    - read_file(host, file) / dump_creds(host): grants creds (dump requires admin)
    - use_creds(host, creds): grants user access
    Success = admin on `target`
    """
    import copy, json

    sim_state = copy.deepcopy(initial_state["agent_start"])
    scenario = initial_state["scenario"]

    # canonicalize lists -> sets for membership checks
    sim_state.setdefault("known_hosts", list(sim_state.get("known_hosts", [])))
    sim_state.setdefault("known_creds", list(sim_state.get("known_creds", [])))
    sim_state.setdefault("access", dict(sim_state.get("access", {})))

    known_hosts = set(sim_state["known_hosts"])
    known_creds = set(sim_state["known_creds"])
    access = dict(sim_state["access"])  # host -> access
    discovered_cves = {host: set() for host in scenario.get("vulnerabilities", {})}

    total_reward = 0.0
    trace = []
    is_success = False

    MAX_STEPS = 200
    STEP_PENALTY = 0.01
    SUCCESS_BONUS = 5.0
    LOOP_PENALTY = -0.5
    rewarded_events = set()
    seen_state_hashes = set()

    def state_hash():
        state_snapshot = {
            "known_hosts": sorted(list(known_hosts)),
            "known_creds": sorted(list(known_creds)),
            "access": {k: access[k] for k in sorted(access.keys())},
            "discovered_cves": {k: sorted(list(v)) for k, v in discovered_cves.items()},
        }
        return json.dumps(state_snapshot, sort_keys=True)

    for i, action in enumerate(actions):
        if i >= MAX_STEPS:
            trace.append({
                "step": i + 1, "action": str(action), "is_valid": False,
                "outcome": "Max steps reached. Ending.", "step_reward": -0.5
            })
            total_reward += -0.5
            break

        tool, args = action.get("tool"), action.get("args", [])
        action_str = f"{tool}({', '.join(args)})"
        step_reward = -STEP_PENALTY
        is_valid_action = False
        target_host = args[0] if args else ""
        outcome = "Action invalid or failed."

        def make_event_id(kind, payload):
            return f"{kind}::{json.dumps(payload, sort_keys=True)}"

        if tool == "scan" and len(args) == 1:
            vulns_in_scenario = scenario.get("vulnerabilities", {}).get(target_host)
            payload = {"host": target_host, "vulns": vulns_in_scenario}
            ev = make_event_id("scan", payload)
            is_valid_action = True
            if ev in rewarded_events:
                step_reward -= 0.05
                outcome = "Scan repeated; no reward."
            else:
                rewarded_events.add(ev)
                if vulns_in_scenario:
                    step_reward += 0.1
                    outcome = f"Scan successful. Found: {json.dumps(vulns_in_scenario)}"
                    # track discovered CVEs for exploit eligibility
                    if "cve" in vulns_in_scenario:
                        discovered_cves[target_host].add(vulns_in_scenario["cve"])
                else:
                    step_reward += 0.05
                    outcome = "Scan successful. No new vulnerabilities found."

        elif tool == "exploit" and len(args) == 2:
            cve = args[1]
            if cve in discovered_cves.get(target_host, set()):
                is_valid_action = True
                if access.get(target_host) == "admin":
                    step_reward -= 0.05
                    outcome = f"Already admin on {target_host}."
                else:
                    ev = make_event_id("exploit", {"host": target_host, "cve": cve})
                    if ev in rewarded_events:
                        step_reward -= 0.05
                        outcome = "Exploit replayed; no reward."
                    else:
                        rewarded_events.add(ev)
                        access[target_host] = "admin"
                        step_reward += 0.5
                        outcome = f"Exploit successful. Admin on {target_host}."
            else:
                is_valid_action = False
                outcome = "CVE not discovered yet; exploit failed."
                step_reward -= 0.1

        elif tool == "read_file" and len(args) == 2:
            file_path = args[1]
            known_vulns = scenario.get("vulnerabilities", {}).get(target_host, {})
            if isinstance(known_vulns, dict) and known_vulns.get("file") == file_path:
                is_valid_action = True
                creds_for = scenario.get("secrets", {}).get(target_host, {}).get("contains_creds_for")
                payload = {"tool": tool, "host": target_host, "creds_for": creds_for}
                ev = make_event_id(tool, payload)
                if creds_for:
                    new_cred = f"creds_for_{creds_for}"
                    if new_cred in known_creds or ev in rewarded_events:
                        step_reward -= 0.05
                        outcome = f"Already knew creds for {creds_for} or repeated event."
                    else:
                        rewarded_events.add(ev)
                        known_creds.add(new_cred)
                        if creds_for not in known_hosts:
                            known_hosts.add(creds_for)
                        step_reward += 0.3
                        outcome = f"Found credentials for {creds_for}."
                else:
                    if ev in rewarded_events:
                        step_reward -= 0.05
                        outcome = "Repeat: no useful credentials."
                    else:
                        rewarded_events.add(ev)
                        step_reward += 0.1
                        outcome = "Action successful, but no useful credentials found."

        elif tool == "dump_creds" and len(args) == 1:
            if access.get(target_host) == "admin":
                is_valid_action = True
                creds_for = scenario.get("secrets", {}).get(target_host, {}).get("contains_creds_for")
                payload = {"tool": tool, "host": target_host, "creds_for": creds_for}
                ev = make_event_id(tool, payload)
                if creds_for:
                    new_cred = f"creds_for_{creds_for}"
                    if new_cred in known_creds or ev in rewarded_events:
                        step_reward -= 0.05
                        outcome = f"Already knew creds for {creds_for} or repeated event."
                    else:
                        rewarded_events.add(ev)
                        known_creds.add(new_cred)
                        if creds_for not in known_hosts:
                            known_hosts.add(creds_for)
                        step_reward += 0.4
                        outcome = f"Dumped credentials for {creds_for}."
                else:
                    if ev in rewarded_events:
                        step_reward -= 0.05
                        outcome = "Repeat: no useful credentials."
                    else:
                        rewarded_events.add(ev)
                        step_reward += 0.1
                        outcome = "Action successful, but no useful credentials found."
            else:
                is_valid_action = False
                step_reward -= 0.1
                outcome = "Admin access required for dump_creds."

        elif tool == "use_creds" and len(args) == 2:
            creds = args[1]
            if creds in known_creds:
                is_valid_action = True
                current_access = access.get(target_host)
                ev = make_event_id("use_creds", {"host": target_host, "creds": creds})
                if current_access in ["user", "admin"]:
                    step_reward -= 0.05
                    outcome = f"Already have access on {target_host}."
                else:
                    if ev in rewarded_events:
                        step_reward -= 0.05
                        outcome = "Use_creds repeated; no reward."
                    else:
                        rewarded_events.add(ev)
                        access[target_host] = "user"
                        step_reward += 0.2
                        outcome = f"Used {creds}. Gained user access on {target_host}."
            else:
                is_valid_action = False
                step_reward -= 0.1
                outcome = "Unknown creds; use_creds failed."

        total_reward += step_reward
        trace.append({
            "step": i + 1,
            "action": action_str,
            "is_valid": is_valid_action,
            "outcome": outcome,
            "step_reward": step_reward,
        })

        sim_state["known_hosts"] = sorted(list(known_hosts))
        sim_state["known_creds"] = sorted(list(known_creds))
        sim_state["access"] = dict(access)

        if access.get(target) == "admin":
            is_success = True
            total_reward += SUCCESS_BONUS
            trace.append({
                "step": i + 1,
                "action": "SUCCESS",
                "is_valid": True,
                "outcome": f"Admin on {target}. Success bonus applied.",
                "step_reward": SUCCESS_BONUS
            })
            break

        # loop detection
        h = state_hash()
        if h in seen_state_hashes:
            total_reward += LOOP_PENALTY
            trace.append({
                "step": i + 1,
                "action": "LOOP_DETECTED",
                "is_valid": False,
                "outcome": "Repeated simulator state detected. Ending episode.",
                "step_reward": LOOP_PENALTY
            })
            break
        seen_state_hashes.add(h)

    return {"final_reward": total_reward, "is_success": is_success, "trace": trace}


_step_counter = 0
_win_loss_stats = {"wins": 0, "losses": 0, "total": 0}


def attack_success_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Simulation-based reward function with enhanced debugging."""
    global _step_counter, _win_loss_stats
    prompts = kwargs.get("prompts") or [""] * len(completions)
    rewards = []
    batch_wins, batch_total = 0, len(completions)
    sim_result = {}  # Define sim_result here to ensure it's in scope for logging

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

            # --- START: ADDED FOR DEBUGGING ---
            # Verbose print statements for one process to avoid log spam
            if os.environ.get("RANK", "0") == "0":
                print("\n--- DEBUG: RUNNING SIMULATION ---")
                if not actions:
                    print("!!! FAILED TO PARSE ACTIONS !!!")
                    print(f"COMPLETION: {full_completion}")
                else:
                    print(f"ACTIONS: {actions}")
            # --- END: ADDED FOR DEBUGGING ---

            if not actions:
                rewards.append(0.0)
                continue

            sim_result = run_simulation(
                actions, initial_state, target="DomainController"
            )

            # --- START: ADDED FOR DEBUGGING ---
            if os.environ.get("RANK", "0") == "0":
                print(f"TRACE: {json.dumps(sim_result['trace'], indent=2)}")
                print(f"FINAL REWARD: {sim_result['final_reward']}")
                print("--- END DEBUG ---")
            # --- END: ADDED FOR DEBUGGING ---

            rewards.append(sim_result["final_reward"])

            if sim_result["is_success"]:
                batch_wins += 1
                if os.environ.get("RANK", "0") == "0" and random.random() < 0.1:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "simulation_traces.jsonl"
                    )
                    with open(log_file, "a") as f:
                        log_entry = {
                            "completion": full_completion,
                            "initial_state": initial_state,
                            "simulation_trace": sim_result["trace"],
                            "final_reward": sim_result["final_reward"],
                        }
                        f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.debug(f"Error in reward function: {e}")
            rewards.append(0.0)

        # --- START: ADDED FOR DEBUGGING ---
        # Log all attempts (successful or not) to a file for later analysis
        if os.environ.get("RANK", "0") == "0":
            os.makedirs("completion_samples", exist_ok=True)
            debug_log_file = os.path.join("completion_samples", "all_completions.jsonl")
            with open(debug_log_file, "a") as f:
                log_entry = {
                    "step": _step_counter,
                    "completion": full_completion,
                    "reward": sim_result.get("final_reward", 0.0),
                    "is_success": sim_result.get("is_success", False),
                }
                f.write(json.dumps(log_entry) + "\n")
        # --- END: ADDED FOR DEBUGGING ---

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
                logs.update(
                    {
                        "attack_success/overall_win_rate": overall_win_rate,
                        "attack_success/recent_win_rate": recent_win_rate,
                        "attack_success/total_wins": _win_loss_stats["wins"],
                        "attack_success/total_attempts": current_total,
                    }
                )
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


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
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
