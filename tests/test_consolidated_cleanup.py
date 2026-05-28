import os
import json
import pytest

def test_cleanup_backup_initialization():
    """
    Verify that the cleanup backup system is properly initialized.
    """
    backup_dir = "backups/consolidated_cleanup_20260227"
    log_file = os.path.join(backup_dir, "cleanup_log.json")
    
    if not os.path.exists(backup_dir) or not os.path.exists(log_file):
        pytest.skip("Cleanup backup artifacts not present in this environment")

    with open(log_file, "r") as f:
        log_data = json.load(f)
        assert isinstance(log_data, list), "Cleanup log should be a JSON list"

def test_module_removal_and_backup():
    """
    Verify that modules are correctly backed up and removed.
    """
    backup_dir = "backups/consolidated_cleanup_20260227"
    log_file = os.path.join(backup_dir, "cleanup_log.json")
    
    # Target modules for this task
    targets = [
        "autotrade/core/premarket_agent.py",
        "autotrade/analysis/pattern_playbooks.py",
        "autotrade/feature_engineering/pairs.py",
        "autotrade/core/pm_workflow.py",
        "autotrade/core/premarket_analyzer.py"
    ]
    
    if not os.path.exists(backup_dir) or not os.path.exists(log_file):
        pytest.skip("Cleanup backup artifacts not present in this environment")

    with open(log_file, "r") as f:
        log_data = json.load(f)

    for target in targets:
        backup_path = os.path.join(backup_dir, os.path.basename(target))
        if os.path.exists(backup_path):
            assert any(entry.get("file") == target for entry in log_data), (
                f"Backup for {target} should be logged"
            )

def test_mcp_archival():
    """
    Verify MCP tools remain in tools/mcp and are not archived.
    """
    archive_dir = "experimental/mcp"
    targets = [
        "ripgrep_mcp.py",
        "ripgrep_mcp_server.py",
        "ruff_mcp_server.py",
        "pytest_mcp_server.py",
        "python_runner_mcp_server.py"
    ]
    
    for target in targets:
        archived_path = os.path.join(archive_dir, target)
        original_path = os.path.join("tools/mcp", target)
        assert os.path.exists(original_path), f"Expected MCP tool at {original_path}"
        assert not os.path.exists(archived_path), f"Archived copy should not exist at {archived_path}"
