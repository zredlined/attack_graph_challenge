# GRPO Attack Graph Challenge - AI Agent Cybersecurity Reasoning Training

Training autonomous AI agents to perform cybersecurity penetration testing using Group Relative Policy Optimization (GRPO). This project teaches language models cybersecurity reasoning and tool usage that requires adaptive reasoning and planning to succeed through reinforcement learning.

## Overview

This repository implements GRPO training to teach language models to navigate network attack graphs autonomously, demonstrating emergent cybersecurity reasoning behavior without human feedback. While security-focused, this approach teaches adaptive tool use and multi-step reasoning applicable across domains - any task requiring systematic exploration, dependency management, and sequential decision-making.

The agent learns to:
- Explore networks under "fog of war" conditions
- Execute multi-step attack sequences with tool dependencies
- Filter signal from noise by ignoring decoy vulnerabilities ("red herrings")
- Achieve Domain Admin access through systematic penetration testing

## Dataset Generation

Training data is generated using **NeMo Data Designer** for efficient, scalable synthetic scenario creation. We'll use Data Designer's templating, statistical sampling, and structured generation capabilities to generate high quality training scenarios for our RL agent. 

### Quick Start

The script below creates `attack_graph_complex_50000.jsonl` with 50,000 synthetic training scenarios for the RL agent to train with in ~90 seconds.

```bash
python scripts/dataset_generation/design_attack_dataset.py
```



### Requirements

- **NeMo Data Designer**: Follow the [deployment guide](https://docs.nvidia.com/nemo/microservices/latest/generate-synthetic-data/index.html)
- **Dependencies**: `pip install nemo-microservices[data-designer] pandas`

### Dataset Structure

Each scenario contains:
- **Target**: Goal system (DomainController)
- **Initial State**: Agent starting position, network topology, vulnerabilities, and secrets
- **Solution Path**: Optimal attack sequence (exploit_path, misconfig_path, hybrid_path)

The generator creates scenarios with realistic noise and red herrings to train agents on decision-making under uncertainty.

For detailed configuration options, see [`scripts/dataset_generation/README.md`](scripts/dataset_generation/README.md).

## How Success is Measured

An agent's generated attack plan is evaluated by a stateful simulator. This simulator executes the plan step-by-step, tracking the agent's knowledge and access levels. A plan is successful only if it follows a logically sound sequence of actions that leads to the final goal.

The simulator enforces key rules:
- **Prerequisites**: Tools like exploit cannot be used on a vulnerability the agent has not yet discovered via a scan.
- **Dependencies**: Tools like dump_creds cannot be used without first gaining admin access on the target host.
- **Accuracy**: The agent must use the specific information (e.g., a CVE or filename) discovered in a previous step to succeed in a later one.

## The Core Challenge: Learning to Reason

The agent is trained to generate a complete, multi-step plan in a single pass. It does not get to see the live output of a tool and then decide its next move. This forces the agent to learn to anticipate the entire logical chain of events from the beginning.

### The Placeholder System

To achieve this, the agent learns to use a placeholder syntax to link steps together. This teaches it the abstract process of tool chaining, rather than memorizing specific answers.

Example of a Successful Plan:
```
<think>
I will scan the WebServer first. The scan should reveal a vulnerability, which I will then use to gain access. After that, I will dump credentials to pivot to the next server.
</think>
<answer>
1. scan(WebServer)
2. exploit(WebServer, @step1_cve)
3. dump_creds(WebServer)
4. use_creds(DBServer, @step3_creds)
</answer>
```

### The "Aha!" Moment: Ignoring Red Herrings

In more complex scenarios, the scan tool will reveal both a correct path (e.g., a readable config file) and a decoy (a fake, non-functional CVE). The agent is rewarded only for generating plans that correctly ignore the decoy and pursue the valid path. This is the core test of its reasoning ability.

## The Experiment: Learning Cybersecurity Reasoning

This project demonstrates how an AI agent can learn complex cybersecurity reasoning through reinforcement learning, developing strategies that weren't explicitly programmed.

### Training Results

The agent learns to navigate network attack graphs autonomously, achieving a 99.2% average win rate in the final 20 training steps. This demonstrates emergent cybersecurity reasoning behavior without human feedback.

Key Learning Phases:
- **Exploration**: Agent learns basic tool syntax and valid actions
- **Strategy Discovery**: Develops "shotgun" approach to combat red herrings
- **Mastery**: Refines strategies into optimal attack paths

### Training Progress & Results

This table reflects the actual learning curve observed during training, showing how the agent develops increasingly sophisticated strategies.

| Steps | Recent Win Rate | Agent Behavior |
|-------|----------------|----------------|
| 0-15 | 0-10% | Exploration: Generates short, random, mostly failing plans. |
| 15-50 | 10-60% | "Aha!" Moment: Discovers that long, chained plans yield high rewards. Win rate rapidly increases. |
| 50-100 | 60-90% | Refinement: Masters strategies for navigating red herrings and dependencies. |
| 100-160 | 90-99.2% | Mastery: Achieves 99.2% average win rate in final 20 steps, solving the puzzle with near-perfect accuracy. |

### Training Metrics Visualization

![Tensorboard Training Results](path/to/tensorboard_image.png)

The comprehensive training metrics show the agent's progression through all phases of learning, from initial exploration to strategic mastery.

![Reward and Completion Length](path/to/reward_curve.png)

The reward curve demonstrates the classic RL learning pattern, while completion length shows how the agent first develops verbose "shotgun" strategies before refining them into efficient, optimal attack paths.

## Training Infrastructure & Requirements

### Model & Hardware Specifications

- **Base Model**: Qwen 2.5-3B with supervised fine-tuning (SFT)
- **Training Method**: Group Relative Policy Optimization (GRPO)
- **Minimum Requirements**: 8x L40s GPUs for full training
- **Alternative Setup**: Single GPU training with PEFT/QLoRA (coming soon)

### Computational Considerations

- **Training Scale**: 168 training steps across 10,752 attack attempts
- **Memory Usage**: Full fine-tuning requires substantial VRAM
- **Roadmap**: PEFT/QLoRA implementation for resource-constrained environments

## Resources & Further Reading

### Project Resources

- [Attack Graph Challenge GitHub Repository](https://github.com/NVIDIA/grpo-attack-graph-challenge)
- [Hugging Face Dataset](https://huggingface.co/datasets/nvidia/attack-graph-scenarios)

### NeMo Data Designer

- [Data Designer Documentation](https://docs.nvidia.com/nemo/microservices/latest/generate-synthetic-data/index.html)
- [Data Designer Examples](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/nemo/NeMo-Data-Designer/intro-tutorials)

### Key Papers & References

- [NVIDIA NeMo RL Framework](https://github.com/NVIDIA-NeMo/RL)
- Group Relative Policy Optimization for reinforcement learning
- [Minimal GRPO implementation](https://github.com/jiayi-pan/minimal-grpo) by Jiayi Pan
- [Hugging Face Trainer implementation of GRPO](https://huggingface.co/blog/grpo) by Phil Schmid

## Acknowledgments

This project builds upon recent advances in reinforcement learning for reasoning tasks, adapting these approaches to the cybersecurity domain. Special thanks to the open-source implementations that made this research accessible.
