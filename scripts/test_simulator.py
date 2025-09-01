import json
import random
from run_r1_grpo import run_simulation

# Ensure reproducibility for testing
random.seed(42)

# --- Use the same definitions from the dataset generator for consistency ---
VULNERABILITIES = {
    "exploit": {"cve": "CVE-2024-SQLi"},
    "misconfig": {"misconfiguration": "Readable-Config-File", "file": "config.json"},
    "privesc": {"cve": "CVE-2024-PrivEsc"},
}
SECRETS = {
    "web_to_db": {"contains_creds_for": "DBServer"},
    "db_to_dc": {"contains_creds_for": "DomainController"}
}

def run_test(name, initial_state, actions):
    """Helper function to run a test and print the results."""
    print(f"\n--- 🧪 Testing {name} ---")
    
    result = run_simulation(actions, initial_state, "DomainController")

    # Pretty print the trace
    for step in result["trace"]:
        print(f"Step {step['step']}: {step['action']} -> Reward: {step['step_reward']}, Outcome: {step['outcome']}")

    print(f"\nFinal reward: {result['final_reward']:.2f}")
    if result['is_success']:
        print(f"Success reaching DomainController? {result['is_success']} ✅")
    else:
        print(f"Success reaching DomainController? {result['is_success']} ❌")
    return result['is_success']

def test_exploit_path():
    """Tests the full attack path using CVE exploits."""
    initial_state = {
        "agent_start": {"known_hosts": ["WebServer"], "access": {"WebServer": "user"}},
        "scenario": {
            "vulnerabilities": {
                "WebServer": VULNERABILITIES["exploit"],
                "DBServer": VULNERABILITIES["privesc"],
                "DomainController": {"cve": "CVE-FINAL-DC"},
            },
            "secrets": {
                "WebServer": SECRETS["web_to_db"],
                "DBServer": SECRETS["db_to_dc"],
            },
        },
    }
    actions = [
        {"tool": "scan", "args": ["WebServer"]},
        {"tool": "exploit", "args": ["WebServer", "CVE-2024-SQLi"]},
        {"tool": "dump_creds", "args": ["WebServer"]},
        {"tool": "use_creds", "args": ["DBServer", "creds_for_DBServer"]},
        {"tool": "scan", "args": ["DBServer"]},
        {"tool": "exploit", "args": ["DBServer", "CVE-2024-PrivEsc"]},
        {"tool": "dump_creds", "args": ["DBServer"]},
        {"tool": "use_creds", "args": ["DomainController", "creds_for_DomainController"]},
        # --- FIX: Scan the final target before exploiting ---
        {"tool": "scan", "args": ["DomainController"]},
        {"tool": "exploit", "args": ["DomainController", "CVE-FINAL-DC"]},
    ]
    return run_test("Exploit Path", initial_state, actions)

def test_misconfig_path():
    """Tests the full attack path using misconfigurations."""
    initial_state = {
        "agent_start": {"known_hosts": ["WebServer"], "access": {"WebServer": "user"}},
        "scenario": {
            "vulnerabilities": {
                "WebServer": VULNERABILITIES["misconfig"],
                "DBServer": {"cve": "CVE-DB-ADMIN"}, # Simplified for testing
                "DomainController": {"cve": "CVE-FINAL-DC"},
            },
            "secrets": {
                "WebServer": SECRETS["web_to_db"],
                "DBServer": SECRETS["db_to_dc"],
            },
        },
    }
    actions = [
        {"tool": "scan", "args": ["WebServer"]},
        {"tool": "read_file", "args": ["WebServer", "config.json"]},
        {"tool": "use_creds", "args": ["DBServer", "creds_for_DBServer"]},
        {"tool": "scan", "args": ["DBServer"]},
        {"tool": "exploit", "args": ["DBServer", "CVE-DB-ADMIN"]},
        {"tool": "dump_creds", "args": ["DBServer"]},
        {"tool": "use_creds", "args": ["DomainController", "creds_for_DomainController"]},
        # --- FIX: Scan the final target before exploiting ---
        {"tool": "scan", "args": ["DomainController"]},
        {"tool": "exploit", "args": ["DomainController", "CVE-FINAL-DC"]},
    ]
    return run_test("Misconfiguration Path", initial_state, actions)

if __name__ == "__main__":
    exploit_success = test_exploit_path()
    misconfig_success = test_misconfig_path()
    
    print("\n--- 📊 Test Summary ---")
    print(f"Exploit Path Test: {'PASS' if exploit_success else 'FAIL'}")
    print(f"Misconfig Path Test: {'PASS' if misconfig_success else 'FAIL'}")