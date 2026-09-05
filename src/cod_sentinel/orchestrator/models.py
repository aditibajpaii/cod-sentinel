"""Anthropic model routing for orchestrator specialists."""

import os

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "claude-sonnet-5")
NEGOTIATOR_MODEL = os.getenv("NEGOTIATOR_MODEL", "claude-sonnet-5")
ADDRESS_MODEL = os.getenv("ADDRESS_MODEL", "claude-haiku-4-5")
DEALMAKER_MODEL = os.getenv("DEALMAKER_MODEL", "claude-haiku-4-5")
