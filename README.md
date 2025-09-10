# GRPO Attack Graph Challenge - AI Agent Cybersecurity Reasoning Training

Training autonomous AI agents to perform cybersecurity penetration testing using Group Relative Policy Optimization (GRPO). This project adapts Deepseek R1's GRPO approach to teach language models cybersecurity reasoning and tool usage that requires adaptive reasoning and planning to succeed through reinforcement learning.

## Overview

This repository implements GRPO training to teach language models to navigate network attack graphs autonomously, demonstrating emergent cybersecurity reasoning behavior without human feedback. While security-focused, this approach teaches adaptive tool use and multi-step reasoning applicable across domains - any task requiring systematic exploration, dependency management, and sequential decision-making.

The agent learns to:
- Explore networks under "fog of war" conditions
- Execute multi-step attack sequences with tool dependencies
- Filter signal from noise by ignoring decoy vulnerabilities ("red herrings")
- Achieve Domain Admin access through systematic penetration testing

## How Success is Measured

An agent's generated attack plan is evaluated by a stateful simulator. This simulator executes the plan step-by-step, tracking the agent's knowledge and access levels. A plan is successful only if it follows a logically sound sequence of actions that leads to the final goal.

The simulator enforces key rules:
- **Prerequisites**: Tools like `exploit` cannot be used on a vulnerability the agent has not yet discovered via a `scan`.
- **Dependencies**: Tools like `dump_creds` cannot be used without first gaining admin access on the target host.
- **Accuracy**: The agent must use the specific information (e.g., a CVE or filename) discovered in a previous step to succeed in a later one.

## The Core Challenge: Learning to Reason

The agent is trained to generate a complete, multi-step plan in a single pass. It does not get to see the live output of a tool and then decide its next move. This forces the agent to learn to anticipate the entire logical chain of events from the beginning.

### The Placeholder System

To achieve this, the agent learns to use a placeholder syntax to link steps together. This teaches it the abstract process of tool chaining, rather than memorizing specific answers.

**Example of a Successful Plan:**
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

**Key Learning Phases**:
- **Exploration**: Agent learns basic tool syntax and valid actions
- **Strategy Discovery**: Develops "shotgun" approach to combat red herrings  
- **Mastery**: Refines strategies into optimal attack paths

## Training Progress & Results

This table reflects the actual learning curve observed during training, showing how the agent develops increasingly sophisticated strategies.

| Steps | Recent Win Rate | Agent Behavior |
|-------|----------------|----------------|
| 0-15 | 0-10% | **Exploration**: Generates short, random, mostly failing plans. |
| 15-50 | 10-60% | **"Aha!" Moment**: Discovers that long, chained plans yield high rewards. Win rate rapidly increases. |
| 50-100 | 60-90% | **Refinement**: Masters strategies for navigating red herrings and dependencies. |
| 100-160 | 90-99.2% | **Mastery**: Achieves 99.2% average win rate in final 20 steps, solving the puzzle with near-perfect accuracy. |

### Training Metrics Visualization

![Tensorboard Training Results](images/tensorboard_results.png)

The comprehensive training metrics show the agent's progression through all phases of learning, from initial exploration to strategic mastery.

![Reward and Completion Length](images/tensorboard_reward.png)

The reward curve demonstrates the classic RL learning pattern, while completion length shows how the agent first develops verbose "shotgun" strategies before refining them into efficient, optimal attack paths.

## Training Infrastructure & Requirements

### Model & Hardware Specifications
- **Base Model**: Qwen 2.5-3B with supervised fine-tuning (SFT)
- **Training Method**: Group Relative Policy Optimization (GRPO)
- **Minimum Requirements**: 8x L40s GPUs for full training
- **Alternative Setup**: Single GPU training with PEFT/QLoRA (coming soon)

### Why 3B Parameters?
Recent research suggests that models around 3B parameters represent the minimum threshold for complex reasoning tasks like multi-step cybersecurity planning. Smaller models struggle with the dependency management and strategic thinking required for this challenge.

### Next Release: Scaling Up with NVIDIA NeMo
The next version will leverage more powerful infrastructure:
- **Model**: [NVIDIA Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) for enhanced reasoning capabilities
- **Framework**: [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) for optimized reinforcement learning
- **Accessibility**: QLoRA with PEFT enabling single GPU fine-tuning
- **Performance**: Expected improvements in strategic reasoning and adaptation

### Computational Considerations
- **Training Scale**: 168 training steps across 10,752 attack attempts
- **Memory Usage**: Full fine-tuning requires substantial VRAM
- **Roadmap**: PEFT/QLoRA implementation for resource-constrained environments

## Generating Datasets for Training

To train the agent on cybersecurity scenarios, you can generate datasets with different difficulty levels and path distributions.

```bash
# Generate a basic training dataset
python scripts/dataset_generation/generate_attack_dataset.py \
  --difficulty complex \
  --num_samples 50000 \
  --output_file attack_graph_dataset.jsonl

# Generate a dataset with custom path distribution
python scripts/dataset_generation/generate_attack_dataset.py \
  --difficulty complex \
  --path_bias 0.3 0.4 0.3 \
  --num_samples 50000 \
  --output_file balanced_dataset.jsonl
```

## Resources & Further Reading

### Project Resources
- [GitHub Repository](https://github.com/yourusername/grpo-attack-graph-challenge)
- [Hugging Face Dataset](https://huggingface.co/datasets/meowterspace45/attack-graph-challenge)

### Key Papers & References
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Minimal GRPO implementation of DeepSeek by Jiayi Pan](https://github.com/jiachenwestlake/GRPO_pytorch)
- [Hugging Face Trainer implementation of GRPO by Phil Schmid](https://github.com/philschmid/grpo-trainer)

### Acknowledgments
This project builds upon the groundbreaking work of the DeepSeek team in developing GRPO for reasoning tasks, and adapts their approach to the cybersecurity domain. Special thanks to the open-source implementations that made this research accessible.
