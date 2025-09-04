Attack Graph Challenge: An RL Scenario for Autonomous Intrusion
This project provides a dataset of simulated cybersecurity scenarios designed for training autonomous AI agents using Reinforcement Learning. The core task is a descriptive, tool-based intrusion challenge where an agent must learn to traverse a network's attack graph to achieve a final objective.

The environment emphasizes exploration, sequential tool usage, and state management under conditions of incomplete information, mirroring real-world security challenges.

How the Challenge Works
The agent's objective is to autonomously navigate a simplified corporate network and gain administrative control. Each scenario starts from a randomized state, presenting a unique attack graph for the agent to solve.

The Goal
The single, unambiguous goal for every scenario is to achieve administrative access on the network's DomainController.

The Environment
The simulated network consists of three interconnected hosts:

WebServer: The public-facing entry point.
DBServer: A backend database server.
DomainController: The central authentication server and the final target.
A key feature of the challenge is the "fog of war." The agent begins with incomplete information about the network. It does not know which vulnerabilities or secrets exist and must actively explore the environment to discover the viable attack path.

The Agent's Toolkit (Actions)
The agent has a predefined set of tools it can use to interact with the environment:

scan(host): Probes a host to discover open ports, services, and potential vulnerabilities or readable files.
exploit(host, cve): Attempts to exploit a specific CVE on a target host to gain administrative privileges on that machine.
read_file(host, file): Reads the contents of a file, which may contain credentials or other useful information.
password_spray(host, creds): Attempts to use a known set of credentials to log into a service on a target host.
dump_creds(host): If the agent has admin access on a host, this action extracts any credentials stored on it.
The Core Learning Task
This challenge is designed to be non-trivial. An agent cannot simply guess the solution; it must learn a sophisticated policy to succeed. Specifically, the agent must learn to:

Explore and Discover: Effectively use tools like scan to overcome the "fog of war," map the attack surface, and uncover the hidden state of the environment.
Execute Dependent Sequences: Construct and follow multi-step "kill chains" where the success of each action is dependent on the outcome of previous ones (e.g., must gain admin access before using dump_creds).
Manage State and Knowledge: Build and maintain an internal representation of the network graph, including credentials found, access levels gained, and known vulnerabilities, to inform future decisions.
Filter Signal from Noise: Learn to identify and ignore irrelevant information ("red herrings"), such as useless files or dead-end network ports, and focus only on the clues that advance its position on the attack graph.
Attack Paths
The challenge includes multiple solution paths that the agent must learn to identify and execute:

1. Exploit Path
Uses CVE-based vulnerabilities to gain system access:

WebServer: Exploit a randomly generated CVE (e.g., CVE-2024-Nexus-12345)
DBServer: Use a privilege escalation vulnerability (e.g., CVE-2025-CoreAPI-PrivEsc-67890)
2. Misconfiguration Path
Leverages security misconfigurations:

WebServer: Read exposed configuration files containing credentials
DBServer: Exploit weak service passwords
3. Hybrid Path (Complex Difficulty Only)
Combines exploitation and misconfiguration techniques:

WebServer: CVE exploitation
DBServer: Misconfiguration abuse
The Dataset
The dataset contains scenarios with dynamically generated vulnerabilities, ensuring each instance presents unique challenges while maintaining consistent structural patterns for learning.

Key Features
Dynamic CVE Generation: Realistic-looking CVE identifiers are randomly generated using products like "Nexus," "DataFlow," "AuthServ," etc.
Randomized Configuration Files: Misconfiguration vulnerabilities reference plausible config files like config.json, settings.ini, db.conf, etc.
Adaptive Noise: Irrelevant information (red herrings) that varies by difficulty level to test the agent's ability to focus on relevant signals.
Generating the Dataset
Use the generate_attack_dataset.py script to create datasets:

Basic Usage
bash
# Generate simple difficulty dataset
python generate_attack_dataset.py --difficulty simple --num_samples 50000

# Generate complex difficulty dataset  
python generate_attack_dataset.py --difficulty complex --num_samples 50000
Advanced Options
bash
# Generate and upload to Hugging Face
python generate_attack_dataset.py \
    --num_samples 500000 \
    --difficulty complex \
    --output_file my_dataset.jsonl \
    --upload_to_hf \
    --hf_username your_username \
    --dataset_name attack_graph_challenge
Command Line Parameters
--num_samples: Number of scenarios to generate (default: 50000)
--difficulty: Difficulty level - simple or complex (default: simple)
--output_file: Output filename (auto-generated if not specified)
--upload_to_hf: Upload dataset to Hugging Face Hub
--hf_username: Your Hugging Face username
--dataset_name: Dataset name for HF Hub (default: attack_graph_challenge)
Difficulty Levels
Simple Difficulty
Attack Paths: 2 paths (exploit_path, misconfig_path)
Noise: Static, predictable distractors
Purpose: Foundational learning and validation of core mechanics
Complex Difficulty
Attack Paths: 3 paths (adds hybrid_path)
Noise: Randomized distractors from pools of useless ports and files
Purpose: Tests generalization and adaptability to unpredictable environments
Recommended Training Methodology
To effectively train an agent while managing complexity, we recommend a two-stage curriculum learning approach:

Stage 1: Foundational Learning (Baseline)
Dataset: Use the simple difficulty setting
Goal: Train the agent on this baseline dataset first to learn core mechanics
Validation: Ensures the agent can overcome "fog of war" and execute basic multi-step attacks
Stage 2: Generalization & Adaptability
Dataset: Use the complex difficulty setting
Goal: Fine-tune the Stage 1 agent on more difficult scenarios
Purpose: Tests ability to generalize to new attack patterns and adapt to randomized environments
Record Structure
The dataset is a JSON Lines (.jsonl) file where each line contains:

target (string): The goal - always "DomainController"
initial_state (JSON string): Serialized JSON object describing the complete scenario state
The initial_state Object
agent_start: Agent's initial conditions (access levels, known hosts, credentials)
scenario: Puzzle layout including vulnerabilities, secrets, and noise
metadata: Supplementary information like the solution_path
Example Record
json
{
  "agent_start": {
    "access": {
      "WebServer": "user"
    },
    "known_hosts": ["WebServer"],
    "known_creds": []
  },
  "scenario": {
    "vulnerabilities": {
      "WebServer": {
        "cve": "CVE-2024-DataFlow-45612"
      },
      "DBServer": {
        "cve": "CVE-2025-SecureStream-PrivEsc-78934"
      }
    },
    "secrets": {
      "WebServer": {
        "contains_creds_for": "DBServer"
      },
      "DBServer": {
        "contains_creds_for": "DomainController"
      }
    },
    "noise": {
      "WebServer": {
        "files": ["temp_notes.txt"]
      },
      "DBServer": {
        "open_ports": [9001]
      }
    }
  },
  "metadata": {
    "solution_path": "exploit_path"
  }
}
Dataset Statistics
After generation, the script provides statistics showing the distribution of solution paths across the dataset. When using --path_bias, the output will reflect your specified distribution:

Distribution of solution paths:
misconfig_path    0.800000
exploit_path      0.100000  
hybrid_path       0.100000
This helps verify that your bias settings are working correctly and enables you to create datasets tailored for specific research objectives.

