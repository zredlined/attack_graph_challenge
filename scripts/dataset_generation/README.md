# Attack Graph Dataset Generation

Generate synthetic cybersecurity attack scenarios for training attack graph models. This tool creates randomized penetration testing scenarios with varying difficulty levels and attack paths.

## Use Case

Training effective attack graph models requires diverse, realistic scenarios that cover different attack vectors, vulnerability types, and network configurations. This generator creates synthetic attack scenarios where an agent must navigate from initial access (WebServer) to a target system (DomainController) through various attack paths:

- **Exploit Path**: CVE-based vulnerabilities requiring exploitation
- **Misconfiguration Path**: Configuration weaknesses and credential theft
- **Hybrid Path**: Combination of exploits and misconfigurations

Each scenario includes realistic noise (red herrings, irrelevant ports, decoy vulnerabilities) to train models on decision-making and path planning.

## Quick Start

Generate 50,000 attack scenarios:

```bash
python design_attack_dataset.py
```

This creates `attack_graph_complex_50000.jsonl` with randomized scenarios in ~90 seconds.

## Configuration Options

### Difficulty Levels

**Simple Mode** (`difficulty="simple"`):
- 2 attack paths: exploit_path, misconfig_path
- Minimal noise (fixed port 8080 on DBServer)
- Straightforward vulnerability patterns

**Complex Mode** (`difficulty="complex"`):
- 3 attack paths: exploit_path, misconfig_path, hybrid_path
- Dynamic noise (random hosts and ports)
- Red herrings (decoy CVEs, useless files)

### Customization

```python
from design_attack_dataset import generate_attack_graph_dataset

# Custom distribution favoring misconfigurations
data = generate_attack_graph_dataset(
    num_samples=10000,
    difficulty="complex", 
    path_bias=[0.1, 0.8, 0.1]  # [exploit, misconfig, hybrid]
)

# Simple scenarios for testing
test_data = generate_attack_graph_dataset(
    num_samples=100, 
    difficulty="simple"
)
```

## Output Format

Each record contains a cybersecurity scenario:

```json
{
  "target": "DomainController",
  "initial_state": {
    "agent_start": {
      "access": {"WebServer": "user"},
      "known_hosts": ["WebServer"],
      "known_creds": []
    },
    "scenario": {
      "vulnerabilities": {
        "WebServer": {"misconfiguration": "Readable-Config-File", "file": "web.config"},
        "DBServer": {"misconfiguration": "Weak-Service-Password"}
      },
      "secrets": {
        "WebServer": {"contains_creds_for": "DBServer"},
        "DBServer": {"contains_creds_for": "DomainController"}
      },
      "noise": {"DBServer": {"open_ports": [8080]}}
    },
    "metadata": {"solution_path": "misconfig_path"}
  }
}
```

## Prerequisites and Setup

This tool uses **NeMo Data Designer** for efficient statistical data generation. NeMo Data Designer provides purpose-built synthetic data generation capabilities with consistent, reliable results at scale.

### 1. Deploy NeMo Data Designer

Follow the [deployment guide](https://docs.nvidia.com/nemo/microservices/latest/generate-synthetic-data/index.html) to set up Data Designer locally. Ensure the service runs on `http://localhost:8080`.

### 2. Install Dependencies

```bash
pip install nemo-microservices[data-designer] pandas
```

### 3. Verify Setup

Check that Data Designer is running:
```bash
curl http://localhost:8080/health
```

## Modifying Parameters

### Adding New Components

Edit `design_attack_dataset.py` to customize the data pools:

```python
# Add new CVE products
CVE_PRODUCTS = ["Nexus", "DataFlow", "AuthServ", "YourProduct"]

# Add configuration files
CONFIG_FILES = ["config.json", "your_config.yml", "app.properties"]

# Add distractor files
USELESS_FILES = ["backup.zip", "readme.txt", "debug.log"]
```

### Changing Defaults

Modify the main section for different default parameters:

```python
if __name__ == "__main__":
    data = generate_attack_graph_dataset(
        num_samples=25000,      # Change default size
        difficulty="simple"     # Change default difficulty
    )
```

### Adding New Attack Paths

To add new scenario types, extend the path logic in the post-processing section and update the attack path samplers.

## Performance

- **50,000 records**: ~90 seconds
- **Memory usage**: Minimal (streaming generation)
- **Output size**: ~2-4MB per 1,000 records
- **Distribution**: Even spread across attack paths (33/33/33 for complex mode)

## How It Works

This generator leverages key NeMo Data Designer features to create consistent, high-quality synthetic data:

**Statistical Sampling Columns**: Generate randomized components like CVE numbers, product names, and configuration files using uniform and categorical distributions. This ensures realistic variety without the overhead of LLM generation.

**Expression Columns**: Use Jinja templates to combine sampled values into structured formats like CVE identifiers (`CVE-2024-{{ product }}-{{ number }}`) and build complex JSON structures deterministically.

**Post-Processing Logic**: After statistical generation, Python logic constructs the final attack scenarios based on the original attack graph rules, ensuring each scenario follows realistic penetration testing patterns.

This approach combines the speed and consistency of statistical sampling with the flexibility of template-based data construction, generating 50,000 records in ~90 seconds while maintaining logical consistency across all attack scenarios.

## Resources

- **NeMo Data Designer Examples**: https://github.com/NVIDIA/GenerativeAIExamples/tree/main/nemo/NeMo-Data-Designer/intro-tutorials
- **Documentation**: https://docs.nvidia.com/nemo/microservices/latest/generate-synthetic-data/index.html