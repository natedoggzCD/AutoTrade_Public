"""
Safe Config Updater - Apply strategy improvements with backup and rollback.
"""
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class SafeConfigUpdater:
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        self.config_path = Path(config_path)
        
    def propose_change(self, param_path: str, old_value: Any, new_value: Any, evidence: Dict) -> Dict:
        """
        Propose a config change with backtest evidence.
        """
        return {
            "param_path": param_path,
            "old_value": old_value,
            "new_value": new_value,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat()
        }

    def apply_changes(self, changes: List[Dict], require_approval: bool = True) -> bool:
        """
        Apply accepted changes to trading_config.yaml.
        """
        if not changes:
            print("No changes to apply.")
            return False
            
        # 1. Provide Summary
        print("\n=== PROPOSED CONFIG CHANGES ===")
        for i, change in enumerate(changes):
            print(f"{i+1}. {change['param_path']}: {change['old_value']} -> {change['new_value']}")
            print(f"   Evidence: Win Rate +{change['evidence'].get('win_rate_delta', 0):.2f}%, Return +{change['evidence'].get('avg_return_delta', 0):.2f}%")
            
        if require_approval:
            ans = input("\nApply these changes? [y/N]: ").strip().lower()
            if ans != 'y':
                print("Changes cancelled.")
                return False
                
        # 2. Backup
        backup_path = self._create_backup()
        print(f"Created backup at {backup_path}")
        
        # 3. Apply
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            for change in changes:
                path_parts = change['param_path'].split('.')
                
                # Navigate to the correct level
                current = config
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                    
                # Set value
                current[path_parts[-1]] = change['new_value']
                
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            print("Configuration updated successfully.")
            return True
            
        except Exception as e:
            print(f"Error updating config: {e}")
            self.rollback(str(backup_path))
            return False

    def _create_backup(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"{self.config_path.name}.{timestamp}.bak"
        shutil.copy2(self.config_path, backup_path)
        return backup_path

    def rollback(self, backup_path: str):
        """Rollback to previous config if new config performs worse."""
        try:
            shutil.copy2(backup_path, self.config_path)
            print(f"Rolled back config to {backup_path}")
        except Exception as e:
            print(f"CRITICAL: Failed to rollback config: {e}")
