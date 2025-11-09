# src/agent_service_toolkit/tools/registry/shared_registry.py
from typing import Dict, List, Any
from langchain_core.tools import Tool

# This is the single, shared instance of the tool registry.
# All registration functions must import and modify this object.
agent_tools: Dict[str, List[Tool]] = {}