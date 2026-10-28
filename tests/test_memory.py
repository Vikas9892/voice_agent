import os
import shutil
import pytest

from app.conversation.memory import ConversationMemory, TurnRecord
from app.llm.main_model import LocalLLM


def test_conversation_memory_turn_tracking():
    mem = ConversationMemory(max_turns=3)
    mem.add_turn("What is Paris?", "Paris is the capital of France.")
    mem.add_turn("What is its population?", "About 2.1 million people.")

    assert len(mem.turns) == 2
    assert mem.turns[0].user == "What is Paris?"
    assert mem.turns[1].user == "What is its population?"

    msgs = mem.get_chat_messages(
        system_prompt="You are a voice assistant.",
        current_user_prompt="What language do they speak?"
    )
    # system + 2 turns (2*2 = 4) + current user = 6 messages
    assert len(msgs) == 6
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[-1]["content"] == "What language do they speak?"


def test_conversation_memory_sliding_window_compression():
    mem = ConversationMemory(max_turns=2)
    mem.add_turn("Turn 1 user", "Turn 1 agent")
    mem.add_turn("Turn 2 user", "Turn 2 agent")
    assert len(mem.turns) == 2
    assert mem.rolling_summary is None

    # Adding 3rd turn triggers eviction of turn 1 and updates rolling summary
    mem.add_turn("Turn 3 user", "Turn 3 agent")
    assert len(mem.turns) == 2
    assert mem.turns[0].user == "Turn 2 user"
    assert mem.turns[1].user == "Turn 3 user"
    assert mem.rolling_summary is not None
    assert "Earlier context:" in mem.rolling_summary


def test_conversation_memory_session_save_and_load(tmp_path):
    session_dir = str(tmp_path / "sessions")
    mem = ConversationMemory(max_turns=3, session_dir=session_dir)
    mem.add_turn("Hello", "Hi there!")
    path = mem.save_session(session_id="test_sess_1")
    assert os.path.exists(path)

    new_mem = ConversationMemory(max_turns=3, session_dir=session_dir)
    loaded = new_mem.load_session("test_sess_1")
    assert loaded is True
    assert len(new_mem.turns) == 1
    assert new_mem.turns[0].user == "Hello"


def test_llm_with_memory():
    mem = ConversationMemory(max_turns=3)
    llm = LocalLLM(memory=mem)
    # Using fallback generator
    resp1 = llm.generate("Hello there!")
    assert "Hello there!" in resp1
    assert len(mem.turns) == 1

    resp2 = llm.generate("Remember this?")
    assert len(mem.turns) == 2
