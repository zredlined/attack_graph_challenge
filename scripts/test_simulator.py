import unittest
import json
from run_r1_grpo import run_simulation

# --- Test Configuration: Mirrors the dataset generator for consistency ---
VULNERABILITIES = {
    "exploit": {"cve": "CVE-2024-SQLi"},
    "misconfig": {"misconfiguration": "Readable-Config-File", "file": "config.json"},
    "privesc": {"cve": "CVE-2024-PrivEsc"},
}
SECRETS = {
    "web_to_db": {"contains_creds_for": "DBServer"},
    "db_to_dc": {"contains_creds_for": "DomainController"}
}

class TestAttackSimulator(unittest.TestCase):

    def setUp(self):
        """A fresh, basic scenario is created before each test method runs."""
        self.initial_state = {
            "agent_start": {"known_hosts": ["WebServer"], "access": {"WebServer": "user"}},
            "scenario": {
                "vulnerabilities": {
                    "WebServer": VULNERABILITIES["exploit"],
                    "DBServer": VULNERABILITIES["privesc"],
                },
                "secrets": { "WebServer": SECRETS["web_to_db"] },
            },
        }

    # --- 1. Sanity Checks for Invalid and Illogical Actions ---
    def test_fails_on_invalid_tool_name(self):
        """The simulator should penalize the use of a non-existent tool."""
        actions = [{"tool": "hack", "args": ["WebServer"]}]
        result = run_simulation(actions, self.initial_state, "DomainController")
        self.assertFalse(result["trace"][0]["is_valid"], "Invalid tool should be marked as not valid.")
        self.assertLess(result["trace"][0]["step_reward"], 0, "Invalid tool should receive a negative reward.")

    def test_fails_on_dump_creds_without_admin_access(self):
        """The agent must have 'admin' access to dump credentials."""
        # Agent starts with 'user' access, so this must fail.
        actions = [{"tool": "dump_creds", "args": ["WebServer"]}]
        result = run_simulation(actions, self.initial_state, "DomainController")
        self.assertFalse(result["trace"][0]["is_valid"], "dump_creds without admin access should be invalid.")

    def test_fails_on_exploit_before_scan(self):
        """The agent cannot use a CVE it hasn't discovered via a scan yet."""
        actions = [{"tool": "exploit", "args": ["WebServer", "CVE-2024-SQLi"]}]
        result = run_simulation(actions, self.initial_state, "DomainController")
        self.assertFalse(result["trace"][0]["is_valid"], "Exploiting an unknown CVE should be invalid.")

    # --- 2. End-to-End Tests for Successful Attack Paths ---
    def test_full_successful_exploit_path(self):
        """The simulator should correctly reward a full, logical exploit chain."""
        state = {
            "agent_start": {"known_hosts": ["WebServer"], "access": {"WebServer": "user"}},
            "scenario": {
                "vulnerabilities": {
                    "WebServer": VULNERABILITIES["exploit"],
                    "DBServer": VULNERABILITIES["privesc"],
                    "DomainController": {"cve": "CVE-FINAL-DC"},
                },
                "secrets": { "WebServer": SECRETS["web_to_db"], "DBServer": SECRETS["db_to_dc"] },
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
            {"tool": "scan", "args": ["DomainController"]},
            {"tool": "exploit", "args": ["DomainController", "CVE-FINAL-DC"]},
        ]
        result = run_simulation(actions, state, "DomainController")
        self.assertTrue(result["is_success"], "A correct exploit path should result in success.")
        self.assertGreater(result["final_reward"], 1.0, "A successful run should have a final reward greater than 1.0.")

    def test_full_successful_misconfig_path(self):
        """The simulator should correctly reward a full, logical misconfiguration chain."""
        state = {
            "agent_start": {"known_hosts": ["WebServer"], "access": {"WebServer": "user"}},
            "scenario": {
                "vulnerabilities": {
                    "WebServer": VULNERABILITIES["misconfig"],
                    "DBServer": {"cve": "CVE-DB-ADMIN"}, # Using a CVE for simplicity to grant admin
                    "DomainController": {"cve": "CVE-FINAL-DC"},
                },
                "secrets": { "WebServer": SECRETS["web_to_db"], "DBServer": SECRETS["db_to_dc"] },
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
            {"tool": "scan", "args": ["DomainController"]},
            {"tool": "exploit", "args": ["DomainController", "CVE-FINAL-DC"]},
        ]
        result = run_simulation(actions, state, "DomainController")
        self.assertTrue(result["is_success"], "A correct misconfig path should result in success.")
        self.assertGreater(result["final_reward"], 1.0, "A successful run should have a final reward greater than 1.0.")


if __name__ == '__main__':
    print("🔬 Running comprehensive simulator sanity checks...")
    unittest.main(verbosity=2)