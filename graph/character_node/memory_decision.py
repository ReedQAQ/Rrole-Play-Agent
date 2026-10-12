"""Compatibility import for the former Role Play memory-decision node.

Role memory proposals and check-gated persistence now live in
``graph.memory_update``.
"""

from graph.memory_update import RoleMemoryDecision as MemoryDecision
from graph.memory_update import role_memory_proposal_node as memory_decision_node

__all__ = ["MemoryDecision", "memory_decision_node"]
