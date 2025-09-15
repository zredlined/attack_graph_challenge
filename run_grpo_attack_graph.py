# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
import pprint
import random
import re
from collections import defaultdict
from typing import Any, Optional

import ray
import torch
from datasets import load_dataset
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (DatumSpec, LLMMessageLogType,
                                     TaskDataProcessFnCallable, TaskDataSpec)
from nemo_rl.distributed.ray_actor_environment_registry import \
    get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

# Enable HF Hub fast downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                         Attack Graph Environment
# ===============================================================================

# Global statistics for tensorboard logging
_step_counter = 0
_win_loss_stats = {"wins": 0, "losses": 0, "total": 0}


def parse_actions_from_answer(answer_text: str) -> list:
    """Extract structured actions like tool(arg1, arg2)."""
    action_regex = re.compile(r"(\w+)\(([^)]*)\)")
    parsed_actions = []
    for match in action_regex.finditer(answer_text):
        tool, args_str = match.group(1).strip(), match.group(2).strip()
        args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
        parsed_actions.append({"tool": tool, "args": args})
    return parsed_actions


def run_simulation(actions: list, initial_state: dict, target: str) -> dict:
    """
    Stateful simulation that understands placeholders to link tool outputs to inputs.
    """
    sim_state = copy.deepcopy(initial_state["agent_start"])
    executed_actions = set()
    scenario = initial_state["scenario"]
    total_reward, is_success, trace = 0.0, False, []

    # A dictionary to store the results of each step for placeholder substitution
    step_outputs = {}

    for i, action in enumerate(actions):
        tool = action.get("tool", "").lower()
        args = action.get("args", [])

        # --- Placeholder Substitution Logic ---
        processed_args = list(args)
        for j, arg in enumerate(processed_args):
            if isinstance(arg, str) and arg.startswith("@step"):
                match = re.match(r"@step(\d+)_(\w+)", arg)
                if match:
                    step_num, key = int(match.group(1)), match.group(2)
                    if step_num in step_outputs and key in step_outputs[step_num]:
                        processed_args[j] = step_outputs[step_num][key]

        action_str = f"{tool}({', '.join(processed_args)})"
        step_result = {
            "step": i + 1,
            "action": action_str,
            "is_valid": False,
            "outcome": "Action invalid or failed.",
            "step_reward": -0.1,
        }

        if action_str in executed_actions:
            step_result.update(
                is_valid=True,
                outcome="Action repeated. No new information gained.",
                step_reward=0.0,
            )
            total_reward += step_result["step_reward"]
            trace.append(step_result)
            continue

        is_valid_action = False
        target_host = processed_args[0] if processed_args else ""

        # --- Action handlers ---
        if tool == "scan" and len(processed_args) == 1:
            if target_host in sim_state.get("known_hosts", []):
                is_valid_action = True
                vulns = scenario.get("vulnerabilities", {}).get(target_host)
                if vulns:
                    sim_state.setdefault("known_vulns", {})[target_host] = vulns
                    step_outputs[i + 1] = vulns
                    step_result.update(
                        outcome=f"Scan successful. Found: {json.dumps(vulns)}",
                        step_reward=0.1,
                    )
                else:
                    step_result.update(
                        outcome="Scan successful. No new vulnerabilities found.",
                        step_reward=0.05,
                    )

        elif tool == "exploit" and len(processed_args) == 2:
            cve_or_vuln_name = processed_args[1]
            known_vulns_on_host = sim_state.get("known_vulns", {}).get(target_host, {})

            cve_match = known_vulns_on_host.get("cve") == cve_or_vuln_name
            misconfig_match = (
                known_vulns_on_host.get("misconfiguration") == cve_or_vuln_name
                and cve_or_vuln_name == "Weak-Service-Password"
            )

            if cve_match or misconfig_match:
                is_valid_action = True
                sim_state.setdefault("access", {})[target_host] = "admin"
                step_result.update(
                    outcome=f"Exploit successful. Admin access on {target_host}.",
                    step_reward=0.5,
                )

        elif tool == "read_file" and len(processed_args) == 2:
            file_path = processed_args[1]
            has_admin = sim_state.get("access", {}).get(target_host) == "admin"
            known_vulns = sim_state.get("known_vulns", {}).get(target_host, {})
            is_readable = (
                known_vulns.get("misconfiguration")
                and known_vulns.get("file") == file_path
            )

            if has_admin or is_readable:
                is_valid_action = True
                creds_for = (
                    scenario.get("secrets", {})
                    .get(target_host, {})
                    .get("contains_creds_for")
                )
                if creds_for:
                    credential_string = f"creds_for_{creds_for}"
                    sim_state.setdefault("known_creds", []).append(credential_string)
                    sim_state.setdefault("known_hosts", []).append(creds_for)
                    step_outputs[i + 1] = {"creds": credential_string}
                    step_result.update(
                        outcome=f"Read {file_path} successful. Found creds: {credential_string}.",
                        step_reward=0.3,
                    )
                else:
                    step_result.update(
                        outcome=f"Read {file_path} successful, but no useful credentials found.",
                        step_reward=0.1,
                    )

        elif tool == "dump_creds" and len(processed_args) == 1:
            if sim_state.get("access", {}).get(target_host) == "admin":
                is_valid_action = True
                creds_for = (
                    scenario.get("secrets", {})
                    .get(target_host, {})
                    .get("contains_creds_for")
                )
                if creds_for:
                    credential_string = f"creds_for_{creds_for}"
                    sim_state.setdefault("known_creds", []).append(credential_string)
                    sim_state.setdefault("known_hosts", []).append(creds_for)
                    step_outputs[i + 1] = {"creds": credential_string}
                    step_result.update(
                        outcome=f"Dumped creds. Found: {credential_string}.",
                        step_reward=0.4,
                    )
                else:
                    step_result.update(
                        outcome="Dumped creds, but nothing useful found.",
                        step_reward=0.1,
                    )

        elif tool == "use_creds" and len(processed_args) == 2:
            creds = processed_args[1]
            if creds in sim_state.get("known_creds", []):
                is_valid_action = True
                if target_host == target:
                    sim_state.setdefault("access", {})[target_host] = "admin"
                    outcome_message = (
                        f"Used {creds}. Gained ADMIN access on {target_host}."
                    )
                else:
                    sim_state.setdefault("access", {})[target_host] = "user"
                    outcome_message = (
                        f"Used {creds}. Gained user access on {target_host}."
                    )
                step_result.update(
                    outcome=outcome_message,
                    step_reward=0.2,
                )

        # --- Apply result ---
        if is_valid_action:
            step_result["is_valid"] = True
            total_reward += step_result["step_reward"]
            if step_result["step_reward"] > 0:
                executed_actions.add(action_str)
        else:
            total_reward += step_result["step_reward"]

        trace.append(step_result)

        # Check for success
        if sim_state.get("access", {}).get(target) == "admin":
            is_success = True
            total_reward += 1.0
            break

    return {
        "final_reward": max(total_reward, 0.0),
        "is_success": is_success,
        "trace": trace,
    }


@ray.remote
class AttackGraphEnvironment:
    """Attack Graph Environment as a Ray actor."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = None

    def set_logger(self, logger):
        """Set the logger for tensorboard logging."""
        self.logger = logger

    def step(self, message_log_batch: list, env_info_batch: list):
        """
        Processes a BATCH of prompts and returns a BATCH of results.
        """
        # Lists to collect results for the entire batch
        batch_observations = []
        batch_rewards = []
        batch_terminateds = []
        batch_answers = []

        # Iterate over each item in the received batch
        for messages, env_info in zip(message_log_batch, env_info_batch):
            completion = ""
            if messages:
                last_message = messages[-1]
                if (
                    isinstance(last_message, list)
                    and len(last_message) > 0
                    and isinstance(last_message[0], dict)
                ):
                    completion = last_message[0].get("content", "")
                elif isinstance(last_message, dict):
                    completion = last_message.get("content", "")

            if not completion:
                reward_val = 0.0
                is_success = True  # Terminate on failure
                trace = []
                observation_content = "Error: No completion generated."
            else:
                reward_result = self.reward_function(completion, env_info)
                reward_val = reward_result.get("reward", 0.0)
                is_success = bool(reward_result.get("is_success", False))
                trace = reward_result.get("trace", [])
                observation_content = f"Simulation complete. Success: {is_success}"

            # Append results for this single item to the batch lists
            batch_observations.append(
                {"role": "environment", "content": observation_content}
            )
            batch_rewards.append(reward_val)
            batch_terminateds.append(is_success)
            batch_answers.append(trace)

        # Return the 6 required items, now correctly batched
        final_observations = batch_observations
        final_metadata = env_info_batch
        final_next_stop_strings = [None] * len(message_log_batch)
        final_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        final_terminateds = torch.tensor(batch_terminateds, dtype=torch.bool)
        final_answers = batch_answers

        return (
            final_observations,
            final_metadata,
            final_next_stop_strings,
            final_rewards,
            final_terminateds,
            final_answers,
        )

    def reward_function(self, completion: str, extra_env_info: dict) -> dict:
        """Compute reward for a completion using attack graph simulation."""
        global _step_counter, _win_loss_stats

        try:
            initial_state = extra_env_info.get("initial_state", {})

            if not initial_state:
                return {"reward": 0.0, "is_success": False, "trace": []}

            if "<|im_start|>assistant" in completion:
                parts = completion.split("<|im_start|>assistant")
                actual_completion = parts[-1].strip().replace("<|im_end|>", "").strip()
            else:
                actual_completion = completion

            if not actual_completion.startswith("<think>"):
                full_completion = "<think>" + actual_completion
            else:
                full_completion = actual_completion

            answer_match = re.search(r"<answer>([\s\S]*?)<\/answer>", full_completion)
            if not answer_match:
                return {"reward": 0.0, "is_success": False, "trace": []}

            actions = parse_actions_from_answer(answer_match.group(1).strip())

            if not actions:
                return {"reward": 0.0, "is_success": False, "trace": []}

            sim_result = run_simulation(
                actions, initial_state, target="DomainController"
            )

            _step_counter += 1
            _win_loss_stats["total"] += 1
            if sim_result["is_success"]:
                _win_loss_stats["wins"] += 1
            else:
                _win_loss_stats["losses"] += 1

            if self.logger and _win_loss_stats["total"] > 0:
                win_rate = _win_loss_stats["wins"] / _win_loss_stats["total"]

                # Print to terminal every 20 steps to avoid clutter
                if _step_counter % 20 == 0:
                    print(
                        f"--> Step {_step_counter}: Overall Win Rate: {win_rate:.2%} ({_win_loss_stats['wins']}/{_win_loss_stats['total']})"
                    )

                self.logger.log_scalar(
                    "attack_success/overall_win_rate", win_rate, _step_counter
                )
                self.logger.log_scalar(
                    "attack_success/total_wins", _win_loss_stats["wins"], _step_counter
                )
                self.logger.log_scalar(
                    "attack_success/total_attempts",
                    _win_loss_stats["total"],
                    _step_counter,
                )

                if _step_counter % 10 == 0:
                    recent_total = min(100, _win_loss_stats["total"])
                    recent_wins = (
                        _win_loss_stats["wins"]
                        if recent_total == _win_loss_stats["total"]
                        else min(_win_loss_stats["wins"], recent_total)
                    )
                    if recent_total > 0:
                        recent_win_rate = recent_wins / recent_total
                        self.logger.log_scalar(
                            "attack_success/recent_win_rate",
                            recent_win_rate,
                            _step_counter,
                        )

            if sim_result["is_success"] and random.random() < 0.1:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "simulation_traces.jsonl")
                with open(log_file, "a") as f:
                    log_entry = {
                        "completion": full_completion,
                        "initial_state": initial_state,
                        "simulation_trace": sim_result["trace"],
                        "final_reward": sim_result["final_reward"],
                    }
                    f.write(json.dumps(log_entry) + "\n")

            if _step_counter % 10 == 0:
                os.makedirs("completion_samples", exist_ok=True)
                debug_log_file = os.path.join(
                    "completion_samples", "example_completions.jsonl"
                )
                with open(debug_log_file, "a") as f:
                    log_entry = {
                        "step": _step_counter,
                        "completion": full_completion,
                        "reward": sim_result["final_reward"],
                        "is_success": sim_result["is_success"],
                    }
                    f.write(json.dumps(log_entry) + "\n")

            return {
                "reward": sim_result["final_reward"],
                "is_success": sim_result["is_success"],
                "trace": sim_result["trace"],
            }

        except Exception as e:
            print(f"Error in attack graph reward function: {e}")
            import traceback

            traceback.print_exc()
            return {"reward": 0.0, "is_success": False, "trace": []}


# ===============================================================================
#                       Attack Graph Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


def attack_graph_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process attack graph data with an inline prompt."""

    initial_state_str = datum_dict["initial_state"]
    initial_state = json.loads(initial_state_str)
    agent_access, known_hosts = (
        initial_state["agent_start"]["access"],
        initial_state["agent_start"]["known_hosts"],
    )
    start_host, start_access = next(iter(agent_access.items()))

    prompt_template = f"""You are a cybersecurity analyst conducting an authorized penetration test.
Your goal is to achieve admin access on the DomainController.

Current situation:
- You have {start_access} access on {start_host}
- Known hosts: {', '.join(known_hosts)}
- Network contains: WebServer, DBServer, DomainController

Available actions and their outputs:
- scan(host) -> outputs: 'cve', 'file'
- exploit(host, cve)
- read_file(host, file) -> outputs: 'creds'
- dump_creds(host) -> outputs: 'creds'
- use_creds(host, creds)

IMPORTANT: To use information from a previous step, use the placeholder format '@step<number>_<key>'.
Example of a logical chain:
1. scan(WebServer)
2. exploit(WebServer, @step1_cve)
3. dump_creds(WebServer)
4. use_creds(DBServer, @step3_creds)

Plan your attack step by step to achieve the goal. Show your reasoning in <think> </think> tags, then provide your attack plan in <answer> </answer> tags."""

    extra_env_info = {"initial_state": initial_state, "target": "DomainController"}

    user_message = {
        "role": "user",
        "content": prompt_template,
    }

    message: str = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message

    message_log: LLMMessageLogType = [user_message]
    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": "attack_graph",
    }
    return output


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data...")
    attack_graph_task_spec = TaskDataSpec(
        task_name="attack_graph",
        system_prompt_file=data_config.get("system_prompt_file"),
    )

    print("Loading attack graph challenge dataset for training and validation")
    dataset = load_dataset(
        "meowterspace45/attack_graph_challenge",
        name="simple",
        split="train",
    )

    def add_task_name(example):
        example["task_name"] = "attack_graph"
        return example

    dataset = dataset.map(add_task_name)

    n = min(len(dataset), 50_000)
    dataset = dataset.shuffle(seed=seed).select(range(n))

    split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_data, val_data = split["train"], split["test"]

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (attack_graph_task_spec, attack_graph_data_processor))
    )
    task_data_processors["attack_graph"] = (
        attack_graph_task_spec,
        attack_graph_data_processor,
    )

    # Simple environment creation without runtime_env specification
    attack_graph_env = AttackGraphEnvironment.remote(env_configs["attack_graph"])

    train_dataset = AllTaskProcessedDataset(
        train_data,
        tokenizer,
        attack_graph_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if val_data:
        val_dataset = AllTaskProcessedDataset(
            val_data,
            tokenizer,
            attack_graph_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: attack_graph_env)
    task_to_env["attack_graph"] = attack_graph_env
    return train_dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_attack_graph_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert (
        config["policy"]["generation"] is not None
    ), "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
