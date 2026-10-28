from app.tools.builtin import (
    calculate_math,
    get_current_time,
    get_system_stats,
    manage_notes,
    get_weather,
    FastToolMatcher,
)
from app.tools.registry import ToolRegistry, default_registry


def test_tool_registry():
    reg = ToolRegistry()

    @reg.register(name="custom_echo", description="Echoes input")
    def echo_func(msg: str):
        return f"Echo: {msg}"

    tool = reg.get("custom_echo")
    assert tool is not None
    assert tool.name == "custom_echo"
    assert tool.execute(msg="hello") == "Echo: hello"
    assert "custom_echo" in reg.get_system_prompt_snippet()


def test_builtin_tools():
    # 1. Time
    time_str = get_current_time()
    assert "The current time is" in time_str

    # 2. System stats
    stats = get_system_stats()
    assert "System:" in stats
    assert "CPU cores" in stats

    # 3. Math
    assert calculate_math("25 * 4") == "25 * 4 = 100"
    assert calculate_math("100 / 5") == "100 / 5 = 20"
    assert "ZeroDivisionError" in calculate_math("10 / 0") or "division by zero" in calculate_math("10 / 0")

    # 4. Notes
    manage_notes(action="clear")
    add_res = manage_notes(action="add", note="buy groceries")
    assert "buy groceries" in add_res
    list_res = manage_notes(action="list")
    assert "buy groceries" in list_res

    # 5. Weather
    weather = get_weather("London")
    assert "London" in weather


def test_fast_tool_matcher():
    matcher = FastToolMatcher(default_registry)

    # Time match
    res = matcher.match_and_execute("Hey what time is it right now?")
    assert res is not None
    assert res[0] == "get_current_time"
    assert "The current time is" in res[1]

    # Math match
    res = matcher.match_and_execute("Can you please calculate 15 * 6")
    assert res is not None
    assert res[0] == "calculate_math"
    assert "90" in res[1]

    # Non-tool query
    res = matcher.match_and_execute("Tell me a philosophical story about consciousness")
    assert res is None
