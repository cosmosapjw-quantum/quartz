"""
Unit tests for operations documentation validation.

Tests verify that the operations runbook contains all required sections,
procedures are documented with proper commands, and references to files
and configurations are accurate.
"""

import os
import re
import unittest
from pathlib import Path
from typing import List, Dict, Any


class TestOperationsDocumentation(unittest.TestCase):
    """Test operations runbook completeness and accuracy."""

    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent.parent / "docs"
        self.operations_file = self.docs_dir / "operations.md"
        self.project_root = Path(__file__).parent.parent.parent

        # Load operations documentation
        with open(self.operations_file, 'r') as f:
            self.operations_content = f.read()

    def test_operations_file_exists(self):
        """Test that operations.md file exists."""
        self.assertTrue(self.operations_file.exists(),
                       "Operations runbook file does not exist")

    def test_required_sections_present(self):
        """Test that all required sections are present in the documentation."""
        required_sections = [
            "Deployment Procedures",
            "Configuration Management",
            "Monitoring & Observability",
            "Troubleshooting Guide",
            "Maintenance Tasks",
            "Performance Optimization",
            "Security & Compliance",
            "Disaster Recovery"
        ]

        for section in required_sections:
            self.assertIn(section, self.operations_content,
                         f"Required section '{section}' not found in operations.md")

    def test_deployment_procedures_comprehensive(self):
        """Test that deployment procedures cover all deployment methods."""
        deployment_methods = [
            "Docker Deployment",
            "Bare Metal Deployment",
            "Cloud Deployment",
            "Configuration Validation"
        ]

        for method in deployment_methods:
            self.assertIn(method, self.operations_content,
                         f"Deployment method '{method}' not documented")

    def test_docker_commands_present(self):
        """Test that essential Docker commands are documented."""
        docker_commands = [
            "docker-compose up -d runtime",
            "docker-compose logs runtime",
            "./scripts/docker/build.sh",
            "docker-compose up -d dev",
            "docker-compose up -d training"
        ]

        for command in docker_commands:
            self.assertIn(command, self.operations_content,
                         f"Docker command '{command}' not documented")

    def test_configuration_references_valid(self):
        """Test that configuration file references are valid."""
        config_files = [
            "config/default.yaml",
            "config/development.yaml",
            "config/production.yaml"
        ]

        for config_file in config_files:
            # Check if mentioned in documentation
            self.assertIn(config_file, self.operations_content,
                         f"Configuration file '{config_file}' not referenced")

            # Check if file actually exists
            config_path = self.project_root / config_file
            self.assertTrue(config_path.exists(),
                           f"Referenced configuration file '{config_file}' does not exist")

    def test_environment_variables_documented(self):
        """Test that environment variable patterns are documented."""
        env_patterns = [
            "ALPHAZERO_MCTS_SIMULATIONS",
            "ALPHAZERO_NEURAL_NETWORK_LEARNING_RATE",
            "ALPHAZERO_SYSTEM_LOG_LEVEL",
            "ALPHAZERO_TRAINING_BATCH_SIZE"
        ]

        for env_var in env_patterns:
            self.assertIn(env_var, self.operations_content,
                         f"Environment variable '{env_var}' not documented")

    def test_monitoring_metrics_documented(self):
        """Test that key monitoring metrics are documented."""
        metrics = [
            "alphazero_simulations_per_second",
            "alphazero_gpu_utilization_percent",
            "alphazero_memory_usage_gb",
            "alphazero_games_generated_per_hour"
        ]

        for metric in metrics:
            self.assertIn(metric, self.operations_content,
                         f"Monitoring metric '{metric}' not documented")

    def test_troubleshooting_scenarios_covered(self):
        """Test that common troubleshooting scenarios are covered."""
        scenarios = [
            "CUDA Out of Memory",
            "Low MCTS Performance",
            "Training Instability",
            "Docker Container Issues",
            "Configuration Errors"
        ]

        for scenario in scenarios:
            self.assertIn(scenario, self.operations_content,
                         f"Troubleshooting scenario '{scenario}' not covered")

    def test_emergency_procedures_present(self):
        """Test that emergency procedures are documented."""
        emergency_procedures = [
            "Emergency stop",
            "Emergency restart",
            "Critical System Recovery",
            "Data Recovery"
        ]

        for procedure in emergency_procedures:
            self.assertIn(procedure, self.operations_content,
                         f"Emergency procedure '{procedure}' not documented")

    def test_maintenance_tasks_scheduled(self):
        """Test that maintenance tasks are properly scheduled."""
        maintenance_schedules = [
            "Daily Tasks",
            "Weekly Tasks",
            "Monthly Tasks"
        ]

        for schedule in maintenance_schedules:
            self.assertIn(schedule, self.operations_content,
                         f"Maintenance schedule '{schedule}' not documented")

    def test_performance_targets_documented(self):
        """Test that performance targets are clearly documented."""
        performance_targets = [
            "30,000+ simulations/second",
            "80-92% sustained during search",
            "<1GB for 10M node trees",
            "200+ self-play games/hour"
        ]

        for target in performance_targets:
            self.assertIn(target, self.operations_content,
                         f"Performance target '{target}' not documented")

    def test_security_hardening_covered(self):
        """Test that security hardening procedures are covered."""
        security_topics = [
            "Container Security",
            "Network Security",
            "Data Protection",
            "Compliance Monitoring"
        ]

        for topic in security_topics:
            self.assertIn(topic, self.operations_content,
                         f"Security topic '{topic}' not covered")

    def test_disaster_recovery_procedures(self):
        """Test that disaster recovery procedures are comprehensive."""
        dr_components = [
            "Recovery Time Objectives",
            "Recovery Point Objectives",
            "Complete System Recovery",
            "Automated Failover"
        ]

        for component in dr_components:
            self.assertIn(component, self.operations_content,
                         f"Disaster recovery component '{component}' not documented")

    def test_command_syntax_valid(self):
        """Test that documented commands have valid syntax."""
        # Extract code blocks containing shell commands
        code_blocks = re.findall(r'```bash\n(.*?)\n```', self.operations_content, re.DOTALL)

        # Basic syntax validation for common commands
        invalid_commands = []

        for block in code_blocks:
            lines = block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Check for common syntax errors
                if line.endswith('\\') and not line.endswith(' \\'):
                    invalid_commands.append(f"Invalid line continuation: {line}")

                # Check for unmatched quotes (basic check)
                single_quotes = line.count("'") - line.count("\\'")
                double_quotes = line.count('"') - line.count('\\"')

                if single_quotes % 2 != 0:
                    invalid_commands.append(f"Unmatched single quotes: {line}")
                if double_quotes % 2 != 0:
                    invalid_commands.append(f"Unmatched double quotes: {line}")

        self.assertEqual(len(invalid_commands), 0,
                        f"Found syntax errors in commands: {invalid_commands}")

    def test_file_references_valid(self):
        """Test that referenced files and directories exist or are reasonable."""
        # Files that should exist
        expected_files = [
            "requirements.txt",
            "pyproject.toml",
            "docker-compose.yml"
        ]

        for file_name in expected_files:
            if file_name in self.operations_content:
                file_path = self.project_root / file_name
                self.assertTrue(file_path.exists(),
                               f"Referenced file '{file_name}' does not exist")

    def test_table_of_contents_matches_content(self):
        """Test that table of contents matches actual section headers."""
        # Extract TOC items
        toc_match = re.search(r'## Table of Contents\n\n(.*?)\n\n---',
                             self.operations_content, re.DOTALL)

        if toc_match:
            toc_content = toc_match.group(1)
            toc_items = re.findall(r'\[([^\]]+)\]', toc_content)

            # Extract actual section headers
            section_headers = re.findall(r'^## ([^\n]+)', self.operations_content, re.MULTILINE)

            # Remove "Table of Contents" from section headers
            section_headers = [h for h in section_headers if h != "Table of Contents"]

            # Check that TOC items exist as section headers
            for toc_item in toc_items:
                self.assertIn(toc_item, section_headers,
                             f"TOC item '{toc_item}' not found in section headers")

    def test_version_and_metadata_present(self):
        """Test that document version and metadata are present."""
        metadata_items = [
            "Version:",
            "Last Updated:",
            "Document Version:",
            "Next Review:"
        ]

        for item in metadata_items:
            self.assertIn(item, self.operations_content,
                         f"Metadata item '{item}' not found")

    def test_contact_information_present(self):
        """Test that contact information is provided."""
        contact_patterns = [
            "Emergency Contacts",
            "@",  # Email addresses should contain @
            "Team"  # Should have team contact info
        ]

        for pattern in contact_patterns:
            self.assertIn(pattern, self.operations_content,
                         f"Contact pattern '{pattern}' not found")

    def test_quick_reference_section(self):
        """Test that quick reference section is comprehensive."""
        quick_ref_items = [
            "Emergency Contacts",
            "Critical Commands",
            "Configuration Shortcuts"
        ]

        for item in quick_ref_items:
            self.assertIn(item, self.operations_content,
                         f"Quick reference item '{item}' not found")


class TestOperationsScriptReferences(unittest.TestCase):
    """Test that scripts referenced in operations documentation are reasonable."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.operations_file = self.project_root / "docs" / "operations.md"

        with open(self.operations_file, 'r') as f:
            self.operations_content = f.read()

    def test_script_references_documented(self):
        """Test that script references are properly documented."""
        # Extract script references from the documentation
        script_patterns = re.findall(r'python scripts/([a-zA-Z_]+\.py)', self.operations_content)

        # These scripts should be documented with their purpose
        essential_scripts = [
            "health_check.py",
            "validate_config.py",
            "tune_threads.py",
            "tune_batch_size.py",
            "backup_daily.sh"
        ]

        documented_scripts = set(script_patterns)

        # Check that essential operational scripts are referenced
        for script in essential_scripts:
            if script.endswith('.sh'):
                # Shell scripts might be referenced differently
                continue
            script_mentioned = any(script in self.operations_content for script in [script])
            self.assertTrue(script_mentioned,
                           f"Essential operational script '{script}' not referenced")

    def test_docker_scripts_referenced(self):
        """Test that Docker helper scripts are referenced."""
        # Check for build script (required)
        self.assertIn("./scripts/docker/build.sh", self.operations_content,
                     "Docker build script not referenced")

        # Check for any docker run patterns (more flexible)
        docker_run_patterns = [
            "./scripts/docker/run.sh",
            "docker-compose up",
            "docker run"
        ]

        has_docker_run = any(pattern in self.operations_content for pattern in docker_run_patterns)
        self.assertTrue(has_docker_run,
                       "No Docker run commands or scripts referenced")


if __name__ == '__main__':
    # Create test suite
    test_cases = [
        TestOperationsDocumentation,
        TestOperationsScriptReferences
    ]

    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    exit(0 if result.wasSuccessful() else 1)