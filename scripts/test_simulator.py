import json
import random

from run_r1_grpo import run_simulation

# Ensure reproducibility for testing
random.seed(42)
if __name__ == "__main__":
    # --- Simple test scenario ---
    initial_state = {
        "agent_start": {
            "known_hosts": ["WebServer"],
            "access": {"WebServer": "user"},  # Start with user access
        },
        "scenario": {
            "vulnerabilities": {
                "WebServer": {"cve": "CVE-2025-0001"},
                "DBServer": {"cve": "CVE-2025-0002", "misconfiguration": True},
                "DomainController": {"cve": "CVE-2025-0003"},
            },
            "network_links": {
                "WebServer": ["DBServer"],
                "DBServer": ["DomainController"],
            },
            "secrets": {
                "WebServer": {"contains_creds_for": "DBServer"},
                "DBServer": {"contains_creds_for": "DomainController"},
            },
        },
    }

    # --- Action sequence that should succeed ---
    actions = [
        {"tool": "scan", "args": ["WebServer"]},
        {"tool": "exploit", "args": ["WebServer", "CVE-2025-0001"]},
        {"tool": "dump_creds", "args": ["WebServer"]},
        {"tool": "scan", "args": ["DBServer"]},
        {"tool": "read_file", "args": ["DBServer", "/etc/passwd"]},
        {"tool": "use_creds", "args": ["DomainController", "creds_for_DomainController"]},
        {"tool": "scan", "args": ["DomainController"]},
        {"tool": "exploit", "args": ["DomainController", "CVE-2025-0003"]},
    ]

    result = run_simulation(actions, initial_state, "DomainController")

    # --- Pretty print the trace ---
    print("\n--- Simulation Trace ---")
    for step in result["trace"]:
        print(f"Step {step['step']}: {step['action']}")
        print(f"  Valid: {step['is_valid']}, Reward: {step['step_reward']}")
        print(f"  Outcome: {step['outcome']}\n")

    print(f"Final reward: {result['final_reward']}")
    print(f"Success reaching DomainController? {result['is_success']}")