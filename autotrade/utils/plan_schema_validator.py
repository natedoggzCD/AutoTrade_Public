"""
Plan Schema Validator and Migration Helper
============================================
Ensures consistent plan file schema across all workflows.

CANONICAL SCHEMA:
- All plan files MUST have a 'signals' key (primary)
- Legacy files may have 'buy_signals', 'entry_candidates', or 'entry_orders'
- This module provides migration and validation utilities

Usage:
    from autotrade.utils.plan_schema_validator import migrate_plan_schema, validate_plan_schema
    
    # Migrate legacy plan
    plan = migrate_plan_schema(plan)
    
    # Validate plan
    is_valid, errors = validate_plan_schema(plan)
"""

import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def migrate_plan_schema(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy plan files to canonical schema.
    
    Canonical schema:
    - 'signals': List[Dict] - Primary key for entry signals
    - 'positions': List[Dict] - Current positions (optional)
    - 'morning_exits': List[Dict] - Planned exits (optional)
    - 'generated_at': str - ISO timestamp
    
    Legacy keys that get migrated:
    - 'buy_signals' -> 'signals'
    - 'entry_candidates' -> 'signals'
    - 'entry_orders' -> 'signals'
    
    Args:
        plan: Plan dictionary (potentially legacy format)
    
    Returns:
        Migrated plan with canonical schema (may be invalid - use validate_plan_schema to check)
    """
    if not isinstance(plan, dict):
        logger.error(f"Invalid plan type: {type(plan)}. Expected dict.")
        # Return invalid structure that will fail validation
        return {'_invalid': True, '_error': f'not a dict: {type(plan)}'}
    
    migrated = plan.copy()
    
    # Migrate signal keys to canonical 'signals'
    if 'signals' not in migrated:
        # Try legacy keys in priority order
        for legacy_key in ['buy_signals', 'entry_candidates', 'entry_orders']:
            if legacy_key in migrated:
                migrated['signals'] = migrated[legacy_key]
                logger.info(f"Migrated '{legacy_key}' -> 'signals' ({len(migrated['signals'])} signals)")
                break
        else:
            # No signal key found, create empty
            migrated['signals'] = []
            logger.warning("No signal keys found in plan. Created empty 'signals' list.")
    
    # Check if 'signals' is valid (but don't auto-fix, let validation catch it)
    if 'signals' in migrated and not isinstance(migrated['signals'], list):
        logger.error(f"Invalid 'signals' type: {type(migrated.get('signals'))}. Expected list.")
        # Mark as having schema error but don't auto-fix to empty list
        migrated['_schema_error'] = f"signals is {type(migrated['signals'])}, expected list"
    
    # Add migration metadata
    if 'signals' not in plan:
        migrated['_migrated'] = True
        migrated['_migrated_at'] = datetime.now().isoformat()
    
    # Ensure generated_at exists
    if 'generated_at' not in migrated:
        migrated['generated_at'] = datetime.now().isoformat()
        logger.warning("Added missing 'generated_at' timestamp")
    
    return migrated


def validate_plan_schema(plan: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate plan schema against canonical format.
    
    Args:
        plan: Plan dictionary to validate
        strict: If True, require all optional fields
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if plan is a dict
    if not isinstance(plan, dict):
        errors.append(f"Plan must be a dict, got {type(plan)}")
        return False, errors
    
    # Required fields
    if 'signals' not in plan:
        errors.append("Missing required field: 'signals'")
    elif not isinstance(plan['signals'], list):
        errors.append(f"Field 'signals' must be a list, got {type(plan['signals'])}")
    
    if 'generated_at' not in plan:
        errors.append("Missing required field: 'generated_at'")
    
    # Optional but recommended fields
    if strict:
        recommended = ['positions', 'morning_exits', 'account']
        for field in recommended:
            if field not in plan:
                errors.append(f"Missing recommended field: '{field}'")
    
    # Validate signal structure
    if 'signals' in plan and isinstance(plan['signals'], list):
        for i, sig in enumerate(plan['signals'][:5]):  # Check first 5
            if not isinstance(sig, dict):
                errors.append(f"Signal {i} must be a dict, got {type(sig)}")
            elif 'symbol' not in sig and 'ticker' not in sig:
                errors.append(f"Signal {i} missing 'symbol' or 'ticker' field")
    
    is_valid = len(errors) == 0
    
    if errors:
        logger.warning(f"Plan validation failed: {'; '.join(errors)}")
    
    return is_valid, errors


def normalize_signal_fields(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize signal field names to canonical format.
    
    Canonical signal fields:
    - symbol: Stock ticker
    - entry_price: Entry price
    - stop_loss: Stop loss price
    - target: Target price
    - confidence: Confidence score (0-100)
    - recommendation: Action (BUY, SELL, HOLD)
    
    Args:
        signal: Signal dictionary with potentially varied field names
    
    Returns:
        Normalized signal dictionary
    """
    normalized = signal.copy()
    
    # Normalize symbol
    if 'symbol' not in normalized:
        normalized['symbol'] = normalized.get('ticker', '')
    
    # Normalize entry price
    if 'entry_price' not in normalized:
        normalized['entry_price'] = normalized.get('entry', normalized.get('price', 0))
    
    # Normalize stop loss
    if 'stop_loss' not in normalized:
        normalized['stop_loss'] = normalized.get('stop', normalized.get('stop_price', 0))
    
    # Normalize target
    if 'target' not in normalized:
        normalized['target'] = normalized.get('take_profit', normalized.get('target_price', 0))
    
    # Normalize confidence
    if 'confidence' not in normalized:
        normalized['confidence'] = normalized.get('final_score', normalized.get('score', 0))
    
    # Normalize recommendation
    if 'recommendation' not in normalized:
        normalized['recommendation'] = normalized.get('action', 'BUY')
    
    return normalized


def load_and_migrate_plan(plan_path: str) -> Dict[str, Any]:
    """
    Load a plan file and automatically migrate it to canonical schema.
    
    Args:
        plan_path: Path to plan JSON file
    
    Returns:
        Migrated plan dictionary
    
    Raises:
        FileNotFoundError: If plan file doesn't exist
        json.JSONDecodeError: If plan file is invalid JSON
    """
    import json
    from pathlib import Path
    
    plan_file = Path(plan_path)
    
    if not plan_file.exists():
        logger.error(f"Plan file not found: {plan_path}")
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    
    try:
        with open(plan_file, 'r') as f:
            plan = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in plan file {plan_path}: {e}")
        raise
    
    # Migrate to canonical schema
    migrated_plan = migrate_plan_schema(plan)
    
    # Validate
    is_valid, errors = validate_plan_schema(migrated_plan)
    
    if not is_valid:
        logger.warning(f"Plan validation errors: {', '.join(errors)}")
    else:
        logger.info(f"Successfully loaded and validated plan: {plan_path}")
    
    return migrated_plan
