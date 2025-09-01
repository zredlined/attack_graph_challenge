# Attack Graph Challenge: An RL Scenario for Autonomous Intrusion

This project provides a dataset of simulated cybersecurity scenarios designed for training autonomous AI agents using Reinforcement Learning. The core task is a descriptive, tool-based intrusion challenge where an agent must learn to traverse a network's attack graph to achieve a final objective.

The environment emphasizes exploration, sequential tool usage, and state management under conditions of incomplete information, mirroring real-world security challenges.

## How the Challenge Works

The agent's objective is to autonomously navigate a simplified corporate network and gain administrative control. Each scenario starts from a randomized state, presenting a unique attack graph for the agent to solve.

### The Goal

The single, unambiguous goal for every scenario is to achieve **`Domain Admin Access`** on the network's `DomainController`.

### The Environment

The simulated network consists of three interconnected hosts:

1.  `WebServer`: The public-facing entry point.
2.  `DBServer`: A backend database server.
3.  `DomainController`: The central authentication server and the final target.

A key feature of the challenge is the **"fog of war."** The agent begins with incomplete information about the network. It does not know which vulnerabilities or secrets exist and must actively explore the environment to discover the viable attack path.

### The Agent's Toolkit (Actions)

The agent has a predefined set of tools it can use to interact with the environment:

  * `scan(host)`: Probes a host to discover open ports, services, and potential vulnerabilities or readable files.
  * `exploit(host, cve)`: Attempts to exploit a specific CVE on a target host to gain administrative privileges on that machine.
  * `read_file(host, file)`: Reads the contents of a file, which may contain credentials or other useful information.
  * `password_spray(host, creds)`: Attempts to use a known set of credentials to log into a service on a target host.
  * `dump_creds(host)`: If the agent has `admin` access on a host, this action extracts any credentials stored on it.

## The Core Learning Task

This challenge is designed to be non-trivial. An agent cannot simply guess the solution; it must learn a sophisticated policy to succeed. Specifically, the agent must learn to:

  * **Explore and Discover:** Effectively use tools like `scan` to overcome the "fog of war," map the attack surface, and uncover the hidden state of the environment.
  * **Execute Dependent Sequences:** Construct and follow multi-step "kill chains" where the success of each action is dependent on the outcome of previous ones (e.g., must gain `admin` access *before* using `dump_creds`).
  * **Manage State and Knowledge:** Build and maintain an internal representation of the network graph, including credentials found, access levels gained, and known vulnerabilities, to inform future decisions.
  * **Filter Signal from Noise:** Learn to identify and ignore irrelevant information ("red herrings"), such as useless files or dead-end network ports, and focus only on the clues that advance its position on the attack graph.

## The Dataset

The dataset contains hundreds of thousands of unique, solvable game scenarios, which can be generated at two different difficulty levels to support a staged training approach.

### Generating the Dataset

The `attack_graph_generator.py` script is used to create the dataset. The key parameter is `--difficulty`:

  * **Simple Difficulty:** Creates a baseline challenge with 2 attack paths and predictable, static "noise."
    ```bash
    python attack_graph_generator.py --difficulty simple --num_samples 500000
    ```
  * **Complex Difficulty:** Creates an advanced challenge with 3 attack paths (including a hybrid path) and randomized "noise" for each scenario.
    ```bash
    python attack_graph_generator.py --difficulty complex --num_samples 500000
    ```

## Recommended Training Methodology

To effectively train an agent while managing complexity, we recommend a two-stage **curriculum learning** approach. This method de-risks the training process and creates a powerful demonstration of the agent's ability to adapt.

### Stage 1: Foundational Learning (Baseline)

  * **Dataset:** Use the **`simple`** difficulty setting.
  * **Goal:** Train the agent on this baseline dataset first. This ensures the agent can learn the core mechanics of the challenge, such as overcoming the "fog of war" and executing basic multi-step attacks, in a more controlled environment. This stage validates the fundamental learnability of the task.

### Stage 2: Generalization & Adaptability

  * **Dataset:** Use the **`complex`** difficulty setting.
  * **Goal:** Take the agent successfully trained in Stage 1 and continue fine-tuning it on this more difficult dataset.
  * **Purpose:** This stage directly tests the agent's ability to **generalize** its strategy to new attack patterns (the hybrid path) and **adapt** to a more unpredictable environment with randomized distractions. Success at this stage is a strong indicator of a robust and adaptable policy.

## Record Structure

The dataset is a **JSON Lines (`.jsonl`)** file where each line is a complete JSON object with two top-level keys:

  * `target` (string): The goal of the challenge. This is always `"Domain Admin Access"`.
  * `initial_state` (JSON string): A string containing a serialized JSON object that describes the complete starting state and puzzle layout for the scenario.

#### The `initial_state` Object

This object contains all the information needed to define a single game instance.

  * `agent_start`: Defines the agent's initial conditions (e.g., `user` access on `WebServer`).
  * `scenario`: Defines the specific puzzle layout for this game, including active `vulnerabilities`, `secrets` to be found, and distracting `noise`.
  * `metadata`: Contains supplementary information, such as the intended `solution_path` (`exploit_path` or `misconfig_path`), which is useful for analysis.

### Example Record

Here is a formatted example of a single record's `initial_state`:

```json
{
  "agent_start": {
    "access": {
      "WebServer": "user"
    },
    "known_hosts": [
      "WebServer"
    ],
    "known_creds": []
  },
  "scenario": {
    "vulnerabilities": {
      "WebServer": {
        "misconfiguration": "Readable-Config-File"
      },
      "DBServer": {
        "misconfiguration": "Weak-Service-Password"
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
      "DBServer": {
        "open_ports": [
          3306,
          8080
        ]
      },
      "WebServer": {
        "files": [
          "index.html",
          "app.js",
          "config.txt.bak"
        ]
      }
    }
  },
  "metadata": {
    "solution_path": "misconfig_path"
  }
}
```
