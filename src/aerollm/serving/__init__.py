"""Minimal, dependency-injected HTTP serving boundary."""

from aerollm.serving.app import create_app
from aerollm.serving.bootstrap import create_production_app

__all__ = ["create_app", "create_production_app"]
