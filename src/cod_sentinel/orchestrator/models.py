"""Anthropic model routing for orchestrator specialists."""

import os

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "claude-sonnet-4-6")
NEGOTIATOR_MODEL = os.getenv("NEGOTIATOR_MODEL", "claude-sonnet-4-6")
ADDRESS_MODEL = os.getenv("ADDRESS_MODEL", "claude-haiku-4-5-20251001")
DEALMAKER_MODEL = os.getenv("DEALMAKER_MODEL", "claude-haiku-4-5-20251001")
