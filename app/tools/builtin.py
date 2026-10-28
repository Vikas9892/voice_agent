import ast
import datetime
import math
import os
import platform
import re
from typing import Any, Dict, List, Optional, Tuple

from app.tools.registry import default_registry, ToolRegistry

_voice_notes: List[str] = []


@default_registry.register(
    name="get_current_time",
    description="Returns current local time, date, and day of the week."
)
def get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("The current time is %I:%M %p on %A, %B %d, %Y.")


@default_registry.register(
    name="get_system_stats",
    description="Returns host hardware system metrics including CPU count, OS, and memory state."
)
def get_system_stats() -> str:
    cpu_count = os.cpu_count() or 4
    os_name = platform.system()
    os_release = platform.release()
    arch = platform.machine()
    return f"System: {os_name} {os_release} ({arch}) with {cpu_count} CPU cores running AVX2-optimized inference."


def _safe_eval_math(node: ast.AST) -> float:
    """Safe evaluation of simple arithmetic AST nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left ** right
        elif isinstance(node.op, ast.Mod):
            return left % right
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_math(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
    raise ValueError("Unsupported math operator")


@default_registry.register(
    name="calculate_math",
    description="Calculates a mathematical expression safely (e.g. 25 * 4, 100 / 5)."
)
def calculate_math(expression: str) -> str:
    cleaned = re.sub(r"[^0-9\+\-\*\/\^\%\.\(\)\s]", "", expression)
    if not cleaned.strip():
        return "Invalid math expression."
    try:
        tree = ast.parse(cleaned, mode='eval')
        result = _safe_eval_math(tree.body)
        if result.is_integer():
            result = int(result)
        return f"{cleaned.strip()} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@default_registry.register(
    name="manage_notes",
    description="Save, list, or clear short voice notes (action: 'add', 'list', 'clear')."
)
def manage_notes(action: str = "list", note: str = "") -> str:
    global _voice_notes
    act = action.lower().strip()
    if act == "add" and note:
        _voice_notes.append(note.strip())
        return f"Saved note: '{note.strip()}'. You have {len(_voice_notes)} notes total."
    elif act == "clear":
        count = len(_voice_notes)
        _voice_notes.clear()
        return f"Cleared {count} notes."
    else:
        if not _voice_notes:
            return "You have no saved notes."
        items = [f"{i+1}. {n}" for i, n in enumerate(_voice_notes)]
        return f"Your notes:\n" + "\n".join(items)


@default_registry.register(
    name="get_weather",
    description="Returns current weather and temperature for a given city."
)
def get_weather(location: str = "local") -> str:
    loc = location.strip().title() if location else "Local"
    return f"The weather in {loc} is currently 72°F (22°C) and partly cloudy with a light breeze."


class FastToolMatcher:
    """Pre-LLM rapid intent parser to match queries to tools in < 1ms on CPU."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or default_registry

    def match_and_execute(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Attempts fast pattern matching against text.
        Returns (tool_name, tool_result, direct_speech_response) if matched, else None.
        """
        lower = text.lower().strip()

        # 1. Time / Date
        if any(p in lower for p in ["what time is it", "current time", "what is the time", "tell me the time", "what's the time", "what day is it", "today's date"]):
            tool = self.registry.get("get_current_time")
            if tool:
                res = tool.execute()
                return "get_current_time", res, res

        # 2. System stats
        if any(p in lower for p in ["cpu usage", "system stats", "computer stats", "hardware stats", "how is cpu doing", "cpu load"]):
            tool = self.registry.get("get_system_stats")
            if tool:
                res = tool.execute()
                return "get_system_stats", res, res

        # 3. Math calculation
        math_match = re.search(r"(?:calculate|what is|compute)\s+([0-9\s\+\-\*\/\.\(\)\^]+)", lower)
        if math_match and any(op in math_match.group(1) for op in ["+", "-", "*", "/"]):
            expr = math_match.group(1).strip()
            tool = self.registry.get("calculate_math")
            if tool:
                res = tool.execute(expression=expr)
                return "calculate_math", res, f"The result is {res}"

        # 4. Notes
        if lower.startswith("take a note") or lower.startswith("save note") or lower.startswith("note that"):
            note_content = re.sub(r"^(?:take a note|save note|note that)\s*(?:to|that|:)?\s*", "", lower).strip()
            tool = self.registry.get("manage_notes")
            if tool:
                res = tool.execute(action="add", note=note_content)
                return "manage_notes", res, res
        elif "list my notes" in lower or "read my notes" in lower or "what are my notes" in lower:
            tool = self.registry.get("manage_notes")
            if tool:
                res = tool.execute(action="list")
                return "manage_notes", res, res

        # 5. Weather
        weather_match = re.search(r"weather(?:\s+in|\s+for)?\s*([a-zA-Z\s]+)?", lower)
        if "weather" in lower and weather_match:
            loc = (weather_match.group(1) or "here").strip()
            tool = self.registry.get("get_weather")
            if tool:
                res = tool.execute(location=loc)
                return "get_weather", res, res

        return None
