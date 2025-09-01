import json
import random

# Ensure reproducibility for testing
random.seed(42)

# --- Minimal test network scenario ---
test_initial_state = {
    "agent_start": {
        "known_hosts": ["WebServer"],
        "access": {"WebServer": "user"},
        "known_creds": [],
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
            "DBServer": {"contains_creds_for": "DomainController"},
        },
    },
}

# --- Example action sequences to test ---
test_actions = [
    {"tool": "scan", "args": ["WebServer"]},
    {"tool": "exploit", "args": ["WebServer", "CVE-2025-0001"]},
    {"tool": "dump_creds", "args": ["WebServer"]},  # No creds expected here
    {"tool": "scan", "args": ["DBServer"]},
    {
        "tool": "read_file",
        "args": ["DBServer", "/etc/passwd"],
    },  # Should find creds for DomainController
    {"tool": "use_creds", "args": ["DomainController", "creds_for_DomainController"]},
]

# --- Run the simulation ---
from run_r1_grpo import \
    run_simulation  # or ensure run_simulation is in the same file

result = run_simulation(test_actions, test_initial_state, target="DomainController")

print("\n--- Simulation Trace ---")
for step in result["trace"]:
    print(
        f"Step {step['step']}: {step['action']}\n"
        f"  Valid: {step['is_valid']}, Reward: {step['step_reward']}\n"
        f"  Outcome: {step['outcome']}\n"
    )

print(f"Final reward: {result['final_reward']}")
print(f"Success reaching DomainController? {result['is_success']}")
