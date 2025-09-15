import copy
import unittest
import random

# Import the simulator function from your main NeMo RL training script
from run_grpo_attack_graph import run_simulation

# --- Helper functions for generating random test data ---
def generate_random_cve():
    """Generates a random, plausible-looking CVE."""
    products = ["Nexus", "DataFlow", "AuthServ", "LogForged", "SecureStream", "CoreAPI"]
    return f"CVE-2024-{random.choice(products)}-{random.randint(10000, 99999)}"


def generate_random_file():
    """Generates a random, plausible config file name."""
    files = ["config.json", "settings.ini", "db.conf", "app.env", "credentials.xml"]
    return random.choice(files)


class TestAttackSimulator(unittest.TestCase):

    def setUp(self):
        """
        Creates a comprehensive but randomized scenario before each test.
        This prevents tests from relying on hardcoded "magic strings".
        """
        self.web_cve = generate_random_cve()
        self.db_cve = generate_random_cve()
        self.web_file = generate_random_file()
        self.decoy_cve = generate_random_cve().replace("CVE", "CVE-Decoy")

        self.base_initial_state = {
            "agent_start": {
                "known_hosts": ["WebServer"],
                "access": {"WebServer": "user"},
                "known_creds": [],
            },
            "scenario": {
                "vulnerabilities": {
                    "WebServer": {
                        "cve": self.web_cve,
                        "misconfiguration": "Readable-Config-File",
                        "file": self.web_file,
                    },
                    "DBServer": {
                        "cve": self.db_cve,
                        "misconfiguration": "Weak-Service-Password",
                    },
                },
                "secrets": {
                    "WebServer": {"contains_creds_for": "DBServer"},
                    "DBServer": {"contains_creds_for": "DomainController"},
                },
            },
        }
        self.target = "DomainController"

    def test_placeholder_substitution_works_correctly(self):
        """Tests that the simulator correctly substitutes a placeholder from a prior step's output."""
        actions = [
            {"tool": "scan", "args": ["WebServer"]},
            {"tool": "exploit", "args": ["WebServer", "@step1_cve"]},
        ]
        result = run_simulation(actions, self.base_initial_state, self.target)
        self.assertTrue(
            result["trace"][1]["is_valid"],
            "Exploit action should be valid after substitution",
        )
        self.assertEqual(
            result["trace"][1]["action"], f"exploit(WebServer, {self.web_cve})"
        )

    def test_non_optimal_path_with_correct_references(self):
        """Tests a longer, non-optimal path to ensure placeholders reference the correct step number."""
        actions = [
            {"tool": "scan", "args": ["WebServer"]},
            {"tool": "scan", "args": ["DBServer"]},  # A useless, intervening step
            {
                "tool": "exploit",
                "args": ["WebServer", "@step1_cve"],
            },  # Must refer to step 1, not 2
        ]
        state = copy.deepcopy(self.base_initial_state)
        state["agent_start"]["known_hosts"].append("DBServer")  # Make step 2 valid
        result = run_simulation(actions, state, self.target)
        self.assertTrue(
            result["trace"][2]["is_valid"],
            "Exploit should be valid, referencing step 1",
        )
        self.assertEqual(
            result["trace"][2]["action"], f"exploit(WebServer, {self.web_cve})"
        )

    def test_full_exploit_path_succeeds_with_placeholders(self):
        """Tests the full 'exploit_path' using the placeholder system."""
        actions = [
            {"tool": "scan", "args": ["WebServer"]},
            {"tool": "exploit", "args": ["WebServer", "@step1_cve"]},
            {"tool": "dump_creds", "args": ["WebServer"]},
            {"tool": "use_creds", "args": ["DBServer", "@step3_creds"]},
            {"tool": "scan", "args": ["DBServer"]},
            {"tool": "exploit", "args": ["DBServer", "@step5_cve"]},
            {"tool": "dump_creds", "args": ["DBServer"]},
            {"tool": "use_creds", "args": ["DomainController", "@step7_creds"]},
        ]
        result = run_simulation(actions, self.base_initial_state, self.target)
        self.assertTrue(
            result["is_success"],
            "The full exploit path with placeholders should succeed",
        )
        self.assertGreater(result["final_reward"], 1.0)

    def test_full_misconfig_path_succeeds_with_placeholders(self):
        """Tests the full 'misconfig_path' using the placeholder system."""
        actions = [
            {"tool": "scan", "args": ["WebServer"]},
            {"tool": "read_file", "args": ["WebServer", "@step1_file"]},
            {"tool": "use_creds", "args": ["DBServer", "@step2_creds"]},
            {"tool": "scan", "args": ["DBServer"]},
            {"tool": "exploit", "args": ["DBServer", "Weak-Service-Password"]},
            {"tool": "dump_creds", "args": ["DBServer"]},
            {"tool": "use_creds", "args": ["DomainController", "@step6_creds"]},
        ]
        result = run_simulation(actions, self.base_initial_state, self.target)
        self.assertTrue(
            result["is_success"],
            "The full misconfig path with placeholders should succeed",
        )
        self.assertGreater(result["final_reward"], 1.0)

    def test_agent_correctly_ignores_decoy(self):
        """
        Tests that the simulator correctly adjudicates a scenario with a decoy.
        The agent must learn to ignore the decoy CVE and follow the file path.
        """
        state_with_decoy = copy.deepcopy(self.base_initial_state)
        state_with_decoy["scenario"]["vulnerabilities"]["WebServer"]["cve"] = self.decoy_cve

        decoy_actions = [
            {"tool": "scan", "args": ["WebServer"]},
            {"tool": "exploit", "args": ["WebServer", "@step1_cve"]},
            {"tool": "dump_creds", "args": ["WebServer"]},
        ]

        correct_actions = [
            {"tool": "scan", "args": ["WebServer"]},
            {"tool": "read_file", "args": ["WebServer", "@step1_file"]},
            {"tool": "use_creds", "args": ["DBServer", "@step2_creds"]},
            {"tool": "scan", "args": ["DBServer"]},
            {"tool": "exploit", "args": ["DBServer", "Weak-Service-Password"]},
            {"tool": "dump_creds", "args": ["DBServer"]},
            {"tool": "use_creds", "args": ["DomainController", "@step6_creds"]},
        ]

        decoy_result = run_simulation(decoy_actions, state_with_decoy, self.target)
        correct_result = run_simulation(correct_actions, state_with_decoy, self.target)

        self.assertTrue(
            decoy_result["trace"][1]["is_valid"],
            "Exploiting the decoy CVE is a valid action...",
        )
        self.assertFalse(
            decoy_result["is_success"],
            "...but it should not lead to a successful outcome for the entire scenario.",
        )
        self.assertTrue(
            correct_result["is_success"],
            "Ignoring the decoy and following the file path should lead to success.",
        )
        self.assertGreater(correct_result["final_reward"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
