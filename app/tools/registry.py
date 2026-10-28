import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """Descriptor for a callable tool in the local voice agent."""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs) -> Any:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


class ToolRegistry:
    """Registry managing available callable tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a function as a voice agent tool."""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip()
            sig = inspect.signature(func)
            params = {}
            for p_name, param in sig.parameters.items():
                params[p_name] = {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "str",
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                }
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=tool_desc,
                func=func,
                parameters=params,
            )
            return func
        return decorator

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_system_prompt_snippet(self) -> str:
        """Generate concise tool instructions for LLM system prompt."""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)


default_registry = ToolRegistry()
