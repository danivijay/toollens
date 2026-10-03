"""Synthetic task generation."""

from toollens.generate.llm import ChatClient, generate_llm_tasks
from toollens.generate.template import generate_template_tasks

__all__ = ["ChatClient", "generate_llm_tasks", "generate_template_tasks"]
