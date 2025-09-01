import json
import random
import pandas as pd
from tqdm import tqdm
import argparse
from datasets import Dataset
from huggingface_hub import login

# --- Game Configuration ---
GOAL = "DomainController"
HOSTS = ["WebServer", "DBServer", "DomainController"]

VULNERABILITIES = {
    "exploit": {"cve": "SQLi-CVE"},
    "misconfig": {"misconfiguration": "Readable-Config-File"},
    "privesc": {"cve": "PrivEsc-CVE"},
    "weak_pw": {"misconfiguration": "Weak-Service-Password"}
}
SECRETS = {
    "web_to_db": {"contains_creds_for": "DBServer"},
    "db_to_dc": {"contains_creds_for": "DomainController"}
}

# Configuration for Randomized Noise (used in 'complex' difficulty)
USELESS_PORTS = [8080, 8443, 9001, 1337]
USELESS_FILES = ["config.txt.bak", "app.log.old", "temp_notes.txt", "archive.zip"]

# --- Dataset Generation ---

def generate_attack_graph_dataset(num_samples: int, difficulty: str):
    """
    Generates a dataset of attack graph scenarios based on the specified difficulty.
    """
    dataset = []

    for _ in tqdm(range(num_samples), desc=f"Generating '{difficulty}' Scenarios"):
        # --- MODIFIED: Select attack paths based on difficulty ---
        if difficulty == 'simple':
            attack_paths = ['exploit_path', 'misconfig_path']
        else:  # complex
            attack_paths = ['exploit_path', 'misconfig_path', 'hybrid_path']
        path_choice = random.choice(attack_paths)

        vulnerabilities = {}
        secrets = {}

        # Path construction logic (includes new hybrid path for complex mode)
        if path_choice == 'exploit_path':
            vulnerabilities["WebServer"] = VULNERABILITIES["exploit"]
            secrets["WebServer"] = SECRETS["web_to_db"]
            vulnerabilities["DBServer"] = VULNERABILITIES["privesc"]
            secrets["DBServer"] = SECRETS["db_to_dc"]
        elif path_choice == 'misconfig_path':
            vulnerabilities["WebServer"] = VULNERABILITIES["misconfig"]
            secrets["WebServer"] = SECRETS["web_to_db"]
            vulnerabilities["DBServer"] = VULNERABILITIES["weak_pw"]
            secrets["DBServer"] = SECRETS["db_to_dc"]
        elif path_choice == 'hybrid_path': # Only chosen in 'complex' mode
            vulnerabilities["WebServer"] = VULNERABILITIES["exploit"]
            secrets["WebServer"] = SECRETS["web_to_db"]
            vulnerabilities["DBServer"] = VULNERABILITIES["weak_pw"]
            secrets["DBServer"] = SECRETS["db_to_dc"]

        # --- MODIFIED: Noise generation is now conditional on difficulty ---
        if difficulty == 'simple':
            # Static, predictable noise for the baseline challenge
            noise = {
                "DBServer": {"open_ports": [8080]},
                "WebServer": {"files": ["config.txt.bak"]}
            }
        else: # complex
            # Randomized noise for the advanced challenge
            noise = {
                random.choice(["WebServer", "DBServer"]): { "open_ports": [random.choice(USELESS_PORTS)] },
                random.choice(HOSTS): { "files": [random.choice(USELESS_FILES)] }
            }

        initial_state = {
          "agent_start": {
            "access": {"WebServer": "user"},
            "known_hosts": ["WebServer"], "known_creds": []
          },
          "scenario": {
              "vulnerabilities": vulnerabilities, "secrets": secrets, "noise": noise
          },
          "metadata": { "solution_path": path_choice }
        }

        record = { "target": GOAL, "initial_state": json.dumps(initial_state, indent=2) }
        dataset.append(record)

    return dataset

def upload_to_huggingface(dataset_data, dataset_name, difficulty, hf_username=None):
    """
    Upload the dataset to Hugging Face Hub
    """
    try:
        # Convert to Hugging Face Dataset format
        df = pd.DataFrame(dataset_data)
        hf_dataset = Dataset.from_pandas(df)
        
        # Generate dataset name
        if hf_username:
            full_dataset_name = f"{hf_username}/{dataset_name}"
        else:
            full_dataset_name = dataset_name
            
        print(f"Uploading dataset to: {full_dataset_name}")
        
        # Upload to Hugging Face Hub
        hf_dataset.push_to_hub(
            full_dataset_name,
            config_name=difficulty,  # Use difficulty as config name
            commit_message=f"Add {difficulty} difficulty attack graph scenarios"
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{full_dataset_name}")
        return full_dataset_name
        
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Attack Graph Challenge dataset.")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of scenarios to generate.")
    # --- Difficulty setting for curriculum learning ---
    parser.add_argument("--difficulty", type=str, default="simple", choices=['simple', 'complex'],
                        help="Difficulty level: 'simple' (2 paths, static noise) or 'complex' (3 paths, random noise).")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: Specify output file name.")
    parser.add_argument("--upload_to_hf", action="store_true", help="Upload dataset to Hugging Face Hub")
    parser.add_argument("--hf_username", type=str, default=None, help="Hugging Face username (optional)")
    parser.add_argument("--dataset_name", type=str, default="attack_graph_challenge", help="Dataset name for HF Hub")
    args = parser.parse_args()

    # --- Auto-generate filename if not provided ---
    if args.output_file is None:
        num_k = args.num_samples // 1000
        args.output_file = f"attack_graph_{args.difficulty}_{num_k}k.jsonl"

    print(f"Generating {args.num_samples} scenarios for '{args.difficulty}' difficulty...")

    dataset_data = generate_attack_graph_dataset(args.num_samples, args.difficulty)

    # Save locally
    df = pd.DataFrame(dataset_data)
    df.to_json(args.output_file, orient='records', lines=True)
    print(f"Dataset saved locally to {args.output_file}")

    # Upload to Hugging Face if requested
    if args.upload_to_hf:
        upload_to_huggingface(dataset_data, args.dataset_name, args.difficulty, args.hf_username)

    print("\n--- Sample Record ---")
    sample_state = json.loads(df.iloc[0]['initial_state'])
    print(f"Target: {df.iloc[0]['target']}")
    print("Initial State:\n" + json.dumps(sample_state, indent=2))

    print("\n--- Dataset Stats ---")
    path_counts = pd.json_normalize(df['initial_state'].apply(json.loads))['metadata.solution_path'].value_counts()
    print("Distribution of solution paths:")
    print(path_counts)