import argparse
import json
import random

import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm

# --- Game Configuration ---
GOAL = "DomainController"
HOSTS = ["WebServer", "DBServer", "DomainController"]

# --- Realistic and Randomized Data Pools ---
CONFIG_FILES = [
    "config.json",
    "settings.ini",
    "db.conf",
    "app.env",
    "credentials.xml",
    "web.config",
    "db_conn.properties",
    "local_settings.py",
    "secrets.yml",
]
CVE_PRODUCTS = ["Nexus", "DataFlow", "AuthServ", "LogForged", "SecureStream", "CoreAPI"]
USELESS_PORTS = [8080, 8443, 9001, 1337, 3000, 5000]
USELESS_FILES = [
    "app.log.old",
    "temp_notes.txt",
    "archive.zip",
    "main.js",
    "style.css",
    "logo.png",
]

# --- Dynamic Vulnerability Generation Functions ---


def generate_exploit_vuln():
    """Generates a random, plausible-looking CVE for an exploit."""
    product = random.choice(CVE_PRODUCTS)
    number = random.randint(10000, 99999)
    return {"cve": f"CVE-2024-{product}-{number}"}


def generate_privesc_vuln():
    """Generates a random CVE for privilege escalation."""
    product = random.choice(CVE_PRODUCTS)
    number = random.randint(10000, 99999)
    return {"cve": f"CVE-2025-{product}-PrivEsc-{number}"}


def generate_misconfig_vuln():
    """Generates a misconfiguration pointing to a random, plausible config file."""
    return {
        "misconfiguration": "Readable-Config-File",
        "file": random.choice(CONFIG_FILES),
    }


def generate_decoy_cve():
    """Generates a red herring CVE that leads nowhere."""
    product = random.choice(CVE_PRODUCTS)
    number = random.randint(10000, 99999)
    return {"cve": f"CVE-2024-{product}-Decoy-{number}"}


# --- Dataset Generation ---


def generate_attack_graph_dataset(
    num_samples: int, difficulty: str, path_bias: list = None
):
    """
    Generates a dataset of attack graph scenarios with randomized vulnerabilities.
    In 'complex' mode, it adds red herrings to force decision-making.
    An optional path_bias can skew the distribution of attack paths.
    """
    dataset = []

    # Define the possible paths and their default weights (for uniform distribution)
    attack_paths = ["exploit_path", "misconfig_path"]
    weights = [0.5, 0.5]
    if difficulty == "complex":
        attack_paths.append("hybrid_path")
        weights = [1 / 3, 1 / 3, 1 / 3]  # Default to equal distribution

    # If a bias is provided, use it to skew the path selection
    if difficulty == "complex" and path_bias and len(path_bias) == 3:
        weights = path_bias
        print(f"Using biased path distribution: {list(zip(attack_paths, weights))}")

    for _ in tqdm(range(num_samples), desc=f"Generating '{difficulty}' Scenarios"):
        # Use random.choices to allow for weighted selection
        path_choice = random.choices(attack_paths, weights=weights, k=1)[0]

        vulnerabilities = {}
        secrets = {
            "WebServer": {"contains_creds_for": "DBServer"},
            "DBServer": {"contains_creds_for": "DomainController"},
        }

        # Path construction logic now uses dynamic vulnerability generation
        if path_choice == "exploit_path":
            vulnerabilities["WebServer"] = generate_exploit_vuln()
            vulnerabilities["DBServer"] = generate_privesc_vuln()
            if difficulty == "complex":
                vulnerabilities["WebServer"]["useless_file"] = random.choice(
                    USELESS_FILES
                )
        elif path_choice == "misconfig_path":
            vulnerabilities["WebServer"] = generate_misconfig_vuln()
            vulnerabilities["DBServer"] = {"misconfiguration": "Weak-Service-Password"}
            if difficulty == "complex":
                vulnerabilities["WebServer"].update(generate_decoy_cve())
        elif path_choice == "hybrid_path":
            vulnerabilities["WebServer"] = generate_exploit_vuln()
            vulnerabilities["DBServer"] = {"misconfiguration": "Weak-Service-Password"}
            if difficulty == "complex":
                vulnerabilities["WebServer"]["useless_file"] = random.choice(
                    USELESS_FILES
                )

        if difficulty == "simple":
            noise = {"DBServer": {"open_ports": [8080]}}
        else:
            noise = {
                random.choice(["WebServer", "DBServer"]): {
                    "open_ports": [random.choice(USELESS_PORTS)]
                }
            }

        initial_state = {
            "agent_start": {
                "access": {"WebServer": "user"},
                "known_hosts": ["WebServer"],
                "known_creds": [],
            },
            "scenario": {
                "vulnerabilities": vulnerabilities,
                "secrets": secrets,
                "noise": noise,
            },
            "metadata": {"solution_path": path_choice},
        }

        record = {"target": GOAL, "initial_state": json.dumps(initial_state, indent=2)}
        dataset.append(record)

    return dataset


def upload_to_huggingface(dataset_data, dataset_name, difficulty, hf_username=None):
    """
    Upload the dataset to Hugging Face Hub
    """
    try:
        df = pd.DataFrame(dataset_data)
        hf_dataset = Dataset.from_pandas(df)
        full_dataset_name = (
            f"{hf_username}/{dataset_name}" if hf_username else dataset_name
        )
        print(f"Uploading dataset to: {full_dataset_name}")
        hf_dataset.push_to_hub(
            full_dataset_name,
            config_name=difficulty,
            commit_message=f"Add {difficulty} difficulty with red herrings and bias support",
        )
        print(
            f"✅ Successfully uploaded to https://huggingface.co/datasets/{full_dataset_name}"
        )
        return full_dataset_name
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the Attack Graph Challenge dataset."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of scenarios to generate.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="simple",
        choices=["simple", "complex"],
        help="Difficulty level: 'simple' or 'complex' (adds red herrings).",
    )
    # Argument to control the distribution of attack paths for complex difficulty
    parser.add_argument(
        "--path_bias",
        type=float,
        nargs=3,
        default=None,
        metavar=("EXPLOIT_W", "MISCONFIG_W", "HYBRID_W"),
        help="For 'complex' mode, set weights for path distribution (e.g., --path_bias 0.1 0.8 0.1)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Specify output file name.",
    )
    parser.add_argument(
        "--upload_to_hf", action="store_true", help="Upload dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf_username", type=str, default=None, help="Hugging Face username (optional)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="attack_graph_challenge",
        help="Dataset name for HF Hub",
    )
    args = parser.parse_args()

    if args.output_file is None:
        num_k = args.num_samples // 1000
        args.output_file = f"attack_graph_{args.difficulty}_{num_k}k.jsonl"

    print(
        f"Generating {args.num_samples} scenarios for '{args.difficulty}' difficulty..."
    )
    dataset_data = generate_attack_graph_dataset(
        args.num_samples, args.difficulty, args.path_bias
    )

    df = pd.DataFrame(dataset_data)
    df.to_json(args.output_file, orient="records", lines=True)
    print(f"Dataset saved locally to {args.output_file}")

    if args.upload_to_hf:
        upload_to_huggingface(
            dataset_data, args.dataset_name, args.difficulty, args.hf_username
        )

    print("\n--- Sample Record ('complex' path might include decoys) ---")
    if not df.empty:
        # Show last record which is more likely to be rare path in a biased set
        sample_state = json.loads(df.iloc[-1]["initial_state"])
        print(f"Target: {df.iloc[-1]['target']}")
        print("Initial State:\n" + json.dumps(sample_state, indent=2))

    print("\n--- Dataset Stats ---")
    if not df.empty:
        # Use normalize=True to show percentages, which is better for biased datasets
        path_counts = pd.json_normalize(df["initial_state"].apply(json.loads))[
            "metadata.solution_path"
        ].value_counts(normalize=True)
        print("Distribution of solution paths:")
        print(path_counts)
