# GRPO Attack Graph Challenge - AI Agent Cybersecurity Training

Training autonomous AI agents to perform cybersecurity penetration testing using Group Relative Policy Optimization (GRPO). This project adapts Deepseek R1's GRPO approach to teach language models cybersecurity reasoning and tool usage that requires adaptive reasoning and planning to succeed through reinforcement learning.

## Overview

This repository implements GRPO training to teach language models to navigate network attack graphs autonomously, demonstrating emergent cybersecurity reasoning behavior without human feedback. **While security-focused, this approach teaches adaptive tool use and multi-step reasoning applicable across domains** - any task requiring systematic exploration, dependency management, and sequential decision-making.

The agent learns to:

- **Explore networks** under "fog of war" conditions
- **Execute multi-step attack sequences** with tool dependencies  
- **Filter signal from noise** to identify viable attack paths
- **Achieve Domain Admin access** through systematic penetration testing

## How Success is Measured

The agent's outputs are evaluated using two reward functions:

1. **Format Reward**: Does the response follow the required `<think>...</think><answer>...</answer>` structure?

2. **Attack Success Reward**: Does the attack plan demonstrate sound cybersecurity reasoning? This heuristic function awards points for:
   - Using reconnaissance tools (scan) before exploitation
   - Respecting tool dependencies (admin access required for credential dumping)
   - Following logical attack sequences based on discovered vulnerabilities
   - Mentioning the correct goal (Domain Admin Access)

Responses scoring >0.8 are considered "wins." The reward function is intentionally heuristic rather than simulating a full environment - it checks for reasonable cybersecurity thinking patterns rather than executing actual attacks.

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone https://github.com/your-username/attack_graph_challenge.git
cd attack_graph_challenge

# Install dependencies
pip install torch==2.5.1 tensorboard "setuptools<71.0.0" --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn
pip install --upgrade \
  "transformers==4.48.1" \
  "datasets==3.1.0" \
  "accelerate==1.3.0" \
  "hf-transfer==0.1.9" \
  "deepspeed==0.15.4" \
  "trl==0.14.0" \
  "huggingface_hub"
pip install "vllm==0.7.0"

# Login to Hugging Face
huggingface-cli login
```

### 2. Run Training

```bash
# For 8 GPUs (7 for training, 1 for vLLM) - adjust NUM_GPUS as needed
./run_training.sh 8

# Or run directly:
accelerate launch \
  --num_processes 7 \
  --config_file configs/accelerate/deepspeed_zero3.yaml \
  scripts/run_r1_grpo.py \
  --config configs/training/grpo-attack-graph.yaml
```

**Note:** Adjust `num_processes` to (number of GPUs - 1). For 4 GPUs use 3, for 8 GPUs use 7.

## Understanding the Challenge

To understand what the agent needs to learn, here's what success and failure look like in the attack graph scenarios:

### The Task Definition

Given a network state like this:
```json
{
  "agent_start": {
    "access": {"WebServer": "user"},
    "known_hosts": ["WebServer"],
    "known_creds": []
  },
  "scenario": {
    "vulnerabilities": {
      "WebServer": {"misconfiguration": "Readable-Config-File"},
      "DBServer": {"misconfiguration": "Weak-Service-Password"}
    },
    "secrets": {
      "WebServer": {"contains_creds_for": "DBServer"},
      "DBServer": {"contains_creds_for": "DomainController"}
    }
  }
}
```

**Goal**: Generate a plan to achieve "Domain Admin Access"

### What Complete Failure Looks Like

```
<think>I need to get domain admin access.</think>
<answer>Use exploit(DomainController, admin-hack) to get access.</answer>
```

**Why this fails:**
- Attempts to attack unknown hosts directly
- Uses non-existent CVEs/tools
- Ignores the required exploration and multi-step dependencies
- No systematic approach

### What Partial Success Looks Like

```
<think>
I should scan the WebServer first to see what's available.
</think>
<answer>
1. scan(WebServer) to find vulnerabilities
2. If there are files, read them for information  
3. Try to move to other servers
4. Eventually get to DomainController
</answer>
```

**Why this is partial:**
- Shows understanding of reconnaissance
- Recognizes need for multi-step approach
- But lacks specific tool usage and dependency understanding
- Vague about actual attack sequence

### What Complete Success Looks Like

```
<think>
I have user access on WebServer. I need to find a path to DomainController.
The scenario shows WebServer has a "Readable-Config-File" misconfiguration 
and contains credentials for DBServer. Let me plan the attack chain:
1. Scan WebServer for the readable config
2. Read the config file to get DBServer credentials  
3. Use those credentials on DBServer
4. DBServer has weak passwords, so I can escalate privileges
5. Dump credentials from DBServer to access DomainController
</think>
<answer>
Step 1: scan(WebServer)
Step 2: read_file(WebServer, config.txt) # Extract DBServer credentials
Step 3: password_spray(DBServer, extracted_credentials)  
Step 4: password_spray(DBServer, weak_service_password) # Escalate to admin
Step 5: dump_creds(DBServer) # Requires admin access
Step 6: password_spray(DomainController, dumped_credentials)
Result: Domain Admin Access achieved
</answer>
```

**Why this succeeds:**
- Systematic reconnaissance approach
- Understands tool dependencies (admin access required for dump_creds)
- Follows logical attack chain based on available vulnerabilities
- Adapts plan based on discovered information

### Key Learning Challenges

The agent must learn:

1. **Tool Dependencies**: Can't use `dump_creds()` without admin access
2. **Information Flow**: Credentials found on one host unlock access to another
3. **Exploration Strategy**: Must use `scan()` to discover available attack vectors
4. **Noise Filtering**: Ignore irrelevant ports/files that don't advance the attack
5. **Sequential Reasoning**: Each step builds on previous discoveries

### Success Criteria

Our reward function considers an attempt successful if it:
- Uses tools in correct dependency order
- Shows systematic exploration (scanning before exploitation)
- Identifies the correct attack path for the given scenario
- Demonstrates understanding of the multi-step nature of network intrusion

Now we can run the training and see if the agent actually learns these behaviors!

### Network Environment
- **WebServer**: Public-facing entry point (initial user access)
- **DBServer**: Backend database server  
- **DomainController**: Final target requiring Domain Admin access

### Agent Capabilities
- `scan(host)`: Discover vulnerabilities and readable files
- `exploit(host, cve)`: Exploit CVEs to gain admin privileges
- `read_file(host, file)`: Extract credentials from configuration files
- `password_spray(host, creds)`: Authenticate with discovered credentials
- `dump_creds(host)`: Extract stored credentials (requires admin access)

### Learning Objectives
The agent must learn to overcome "fog of war" by systematically:
1. **Exploring** the network to discover attack surfaces
2. **Sequencing** dependent actions (e.g., gain admin before credential dumping)
3. **Following** multi-step attack paths to reach the Domain Controller
4. **Ignoring** noise and red herrings in the environment

## Dataset

Uses `meowterspace45/attack-graph-challenge` with 500K procedurally generated scenarios:

- **Simple difficulty**: 2 attack paths, predictable noise (recommended for initial training)
- **Complex difficulty**: 3 attack paths, randomized noise (for advanced training)

### Generating Your Own Dataset

```bash
# Generate simple dataset (recommended starting point)
python scripts/dataset_generation/generate_attack_dataset.py \
  --difficulty simple \
  --num_samples 50000 \
  --upload_to_hf \
  --hf_username YOUR_USERNAME

# Generate complex dataset (for curriculum learning)
python scripts/dataset_generation/generate_attack_dataset.py \
  --difficulty complex \
  --num_samples 50000 \
  --upload_to_hf \
  --hf_username YOUR_USERNAME
```

## Training Progress

Expected learning progression on simple difficulty:

| Steps | Success Rate | Agent Behavior | Time (8x H100) |
|-------|-------------|----------------|----------------|
| 50    | ~5%         | Learns action format | ~30 min |
| 100   | ~15%        | Basic tool usage | ~1 hr |
| 200   | ~30%        | Sequential reasoning | ~2 hr |
| 450   | ~45%        | Systematic attack paths | ~4 hr |

## Repository Structure

```
├── configs/
│   ├── accelerate/
│   │   └── deepspeed_zero3.yaml     # DeepSpeed distributed training config
│   └── training/
│       └── grpo-attack-graph.yaml   # GRPO hyperparameters
├── scripts/
│   ├── run_r1_grpo.py              # Main training script
│   └── dataset_generation/
│       ├── generate_attack_dataset.py  # Dataset generation
│       └── README.md               # Dataset documentation
├── outputs/                         # Training outputs
│   ├── models/                     # Saved model checkpoints
│   ├── logs/                       # Training logs
│   └── samples/                    # Generated sample outputs
├── requirements.txt
├── run_training.sh                 # Convenience training script
└── README.md
```

## Key Hyperparameters

Adapted from the original DeepSeek R1 experiment:
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Dataset**: 50,000 samples from attack-graph-challenge
- **Learning rate**: 5e-7 
- **Beta (KL)**: 0.001 
- **Generations**: 8 per prompt
- **Steps**: 450

## Monitoring Training Progress

### Real-Time Metrics Dashboard
```bash
# Launch the custom training monitor
python scripts/monitor_training.py

# Or with custom settings
python scripts/monitor_training.py --log_dir runs --refresh 15
```

### TensorBoard Metrics
```bash
# Traditional tensorboard view
tensorboard --logdir runs/

# Key metrics to watch:
# - attack_success/overall_win_rate: Overall success rate
# - attack_success/recent_win_rate: Recent batch success rate  
# - attack_success/total_wins: Total successful attack plans
# - train/reward: Average reward scores
```

### Sample Output Analysis
```bash
# Check successful attack plans
tail -f completion_samples/success_completion_samples.txt

# Monitor all generated samples
tail -f completion_samples/completion_samples.txt

# Quick stats
ls -la completion_samples/
```

### Expected Training Progression

| Steps | Win Rate | Agent Behavior | Sample Success |
|-------|----------|----------------|----------------|
| 0-50  | 0-5%     | Learning format, random actions | Very rare |
| 50-100| 5-15%    | Basic tool recognition | Occasional |
| 100-200| 15-30%  | Sequential reasoning emerges | Regular |
| 200-350| 30-45%  | Systematic attack planning | Frequent |
| 350-450| 45-60%  | Consistent attack execution | Most attempts |

## Curriculum Learning Approach

**Stage 1 - Foundation (Simple Difficulty)**:
- Train on predictable scenarios with 2 attack paths
- Establish basic tool usage and sequential reasoning
- Validate core cybersecurity concepts

**Stage 2 - Generalization (Complex Difficulty)**:
- Fine-tune on randomized scenarios with 3 attack paths  
- Test adaptability to novel attack combinations
- Demonstrate robust cybersecurity reasoning

## Hardware Requirements

- **Minimum**: 1x GPU with 24GB VRAM (slow, not recommended)
- **Recommended**: 8x H100 80GB (optimal performance)
- **Budget**: 4x RTX 4090 24GB (reasonable performance)

## Important Notes

1. Uses full model training (no LoRA/PEFT) for maximum capability
2. One GPU dedicated to vLLM generation during training
3. Designed for authorized penetration testing education only
4. Based on simplified network scenarios for demonstration

## Credits

- **Original approach**: [TinyZero](https://github.com/Jiayi-Pan/TinyZero) by Jiayi Pan
- **Distributed implementation**: [Phil Schmid](https://www.philschmid.de/deepseek-r1-reproduction)
- **Attack graph methodology**: Inspired by MITRE ATT&CK framework

## License

MIT - Educational and research purposes only. Not intended for malicious use.