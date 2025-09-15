import json
import os

import pandas as pd
from nemo_microservices import NeMoMicroservices
from nemo_microservices.beta.data_designer import (DataDesignerClient,
                                                   DataDesignerConfigBuilder)
from nemo_microservices.beta.data_designer.config import columns as C
from nemo_microservices.beta.data_designer.config import params as P


def generate_attack_graph_dataset(
    num_samples: int = 100, difficulty: str = "simple", path_bias=None
):
    """Generate attack graph dataset using NeMo Data Designer - simplified approach"""

    # Initialize client
    data_designer_client = DataDesignerClient(
        client=NeMoMicroservices(base_url="http://localhost:8080")
    )

    config_builder = DataDesignerConfigBuilder()

    # Define data pools
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

    CVE_PRODUCTS = [
        "Nexus",
        "DataFlow",
        "AuthServ",
        "LogForged",
        "SecureStream",
        "CoreAPI",
    ]
    USELESS_PORTS = [8080, 8443, 9001, 1337, 3000, 5000]
    USELESS_FILES = [
        "app.log.old",
        "temp_notes.txt",
        "archive.zip",
        "main.js",
        "style.css",
        "logo.png",
    ]

    # 1. Attack path selection
    if difficulty == "simple":
        attack_paths = ["exploit_path", "misconfig_path"]
        weights = [0.5, 0.5]
    else:  # complex
        attack_paths = ["exploit_path", "misconfig_path", "hybrid_path"]
        weights = path_bias if path_bias else [1 / 3, 1 / 3, 1 / 3]

    config_builder.add_column(
        C.SamplerColumn(
            name="solution_path",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(values=attack_paths, weights=weights),
        )
    )

    # 2. Component samplers
    config_builder.add_column(
        C.SamplerColumn(
            name="cve_product",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(values=CVE_PRODUCTS),
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="cve_number",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=10000, high=99999),
            convert_to="int",
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="config_file",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(values=CONFIG_FILES),
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="useless_file",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(values=USELESS_FILES),
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="useless_port",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(values=USELESS_PORTS),
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="noise_host",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(values=["WebServer", "DBServer"]),
        )
    )

    # 3. Build CVE strings
    config_builder.add_column(
        C.ExpressionColumn(
            name="exploit_cve", expr="CVE-2024-{{ cve_product }}-{{ cve_number }}"
        )
    )

    config_builder.add_column(
        C.ExpressionColumn(
            name="privesc_cve",
            expr="CVE-2025-{{ cve_product }}-PrivEsc-{{ cve_number }}",
        )
    )

    config_builder.add_column(
        C.ExpressionColumn(
            name="decoy_cve", expr="CVE-2024-{{ cve_product }}-Decoy-{{ cve_number }}"
        )
    )

    # 4. Target column
    config_builder.add_column(
        C.ExpressionColumn(name="target", expr="DomainController")
    )

    # Validate and generate
    config_builder.validate()

    print(f"Generating {num_samples} scenarios for '{difficulty}' difficulty...")
    if path_bias:
        print(f"Using biased path distribution: {list(zip(attack_paths, path_bias))}")

    # Generate the dataset (use preview for small samples to avoid the complex generation job)
    if num_samples <= 10:
        preview = data_designer_client.preview(config_builder, verbose_logging=False)
        dataset_df = preview.dataset
    else:
        results = data_designer_client.create(
            config_builder, num_records=num_samples, wait_until_done=True
        )
        dataset_df = results.load_dataset()

    # Post-process to create the final JSON structure
    formatted_data = []
    for _, record in dataset_df.iterrows():
        # Build the vulnerabilities based on the logic from original script
        vulnerabilities = {}

        if record["solution_path"] == "exploit_path":
            vulnerabilities["WebServer"] = {"cve": record["exploit_cve"]}
            vulnerabilities["DBServer"] = {"cve": record["privesc_cve"]}
            if difficulty == "complex":
                vulnerabilities["WebServer"]["useless_file"] = record["useless_file"]

        elif record["solution_path"] == "misconfig_path":
            vulnerabilities["WebServer"] = {
                "misconfiguration": "Readable-Config-File",
                "file": record["config_file"],
            }
            vulnerabilities["DBServer"] = {"misconfiguration": "Weak-Service-Password"}
            if difficulty == "complex":
                vulnerabilities["WebServer"]["cve"] = record["decoy_cve"]

        elif record["solution_path"] == "hybrid_path":
            vulnerabilities["WebServer"] = {"cve": record["exploit_cve"]}
            vulnerabilities["DBServer"] = {"misconfiguration": "Weak-Service-Password"}
            if difficulty == "complex":
                vulnerabilities["WebServer"]["useless_file"] = record["useless_file"]

        # Build noise
        if difficulty == "simple":
            noise = {"DBServer": {"open_ports": [8080]}}
        else:
            noise = {record["noise_host"]: {"open_ports": [record["useless_port"]]}}

        # Build complete initial state
        initial_state = {
            "agent_start": {
                "access": {"WebServer": "user"},
                "known_hosts": ["WebServer"],
                "known_creds": [],
            },
            "scenario": {
                "vulnerabilities": vulnerabilities,
                "secrets": {
                    "WebServer": {"contains_creds_for": "DBServer"},
                    "DBServer": {"contains_creds_for": "DomainController"},
                },
                "noise": noise,
            },
            "metadata": {"solution_path": record["solution_path"]},
        }

        formatted_record = {
            "target": record["target"],
            "initial_state": json.dumps(initial_state, indent=2),
        }
        formatted_data.append(formatted_record)

    return formatted_data


# Simple usage
if __name__ == "__main__":
    # Generate simple dataset
    num_samples = 50_000
    difficulty = "complex"
    simple_data = generate_attack_graph_dataset(
        num_samples=num_samples, difficulty=difficulty
    )

    # Save to file
    df = pd.DataFrame(simple_data)
    df.to_json(
        f"attack_graph_{difficulty}_{num_samples}.jsonl", orient="records", lines=True
    )

    print("--- Sample Record ---")
    sample_state = json.loads(simple_data[0]["initial_state"])
    print(f"Target: {simple_data[0]['target']}")
    print("Initial State:")
    print(json.dumps(sample_state, indent=2))

    print("\n--- Dataset Stats ---")
    paths = [
        json.loads(record["initial_state"])["metadata"]["solution_path"]
        for record in simple_data
    ]
    from collections import Counter

    path_counts = Counter(paths)
    total = len(paths)
    print("Distribution of solution paths:")
    for path, count in path_counts.items():
        print(f"{path}: {count/total:.3f}")
