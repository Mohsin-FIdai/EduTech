"""
roman_urdu_nlp/utils.py — Shared utility functions.
"""

from __future__ import annotations
import os


def env_flag(name: str, default: bool = True) -> bool:
    """
    Parse an environment variable as a boolean.

    Truthy values: 'true', '1', 'yes', 'on' (case-insensitive).
    Falsy values: 'false', '0', 'no', 'off', '' (case-insensitive).
    Returns default if the environment variable is not set.
    """
    val = os.getenv(name)
    if val is None:
        return default
    val_clean = val.strip().lower()
    if not val_clean:
        return False
    return val_clean in ("true", "1", "yes", "on")
