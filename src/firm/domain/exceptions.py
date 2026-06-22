"""Domain exceptions — no IO imports."""

from __future__ import annotations


class InsufficientHolding(Exception):
    """Raised when a close request exceeds the open position."""


class InsufficientCash(Exception):
    """Raised when cash is insufficient to fund a trade."""
