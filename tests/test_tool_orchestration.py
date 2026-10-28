import asyncio
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.conversation.cancellation import CancellationCoordinator
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager
from app.tools.builtin import FastToolMatcher


class MockSTT:
    def __init__(self, transcript: str):
        self._transcript = transcript

    def transcribe(self, audio):
        return self._transcript


class MockTTS:
    def synthesize(self, text, sample_rate=22050):
        # return dummy audio
        return np.zeros(200, dtype=np.float32)


class MockLLM:
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str):
        self.call_count += 1
        return "LLM response: " + prompt


@pytest.mark.asyncio
async def test_orchestrator_tool_execution_bypasses_llm():
    # Setup orchestrator with STT returning a tool question
    stt = MockSTT("What time is it right now?")
    llm = MockLLM()
    tts = MockTTS()
    turn_manager = MagicMock(spec=TurnManager)

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=stt,
        llm=llm,
        tts=tts,
        tool_matcher=FastToolMatcher(),
    )

    audio_buf = np.zeros(16000, dtype=np.float32)
    await orchestrator._handle_turn(audio_buf, turn_id=1)

    # Verify tool matched, so LLM was never called!
    assert llm.call_count == 0
    assert "tool_execution_completed" in orchestrator.active_metrics.timestamps


@pytest.mark.asyncio
async def test_orchestrator_non_tool_query_calls_llm():
    stt = MockSTT("Tell me why the sky is blue.")
    llm = MockLLM()
    tts = MockTTS()
    turn_manager = MagicMock(spec=TurnManager)

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=stt,
        llm=llm,
        tts=tts,
        tool_matcher=FastToolMatcher(),
    )

    audio_buf = np.zeros(16000, dtype=np.float32)
    await orchestrator._handle_turn(audio_buf, turn_id=2)

    # Verify normal non-tool query called LLM
    assert llm.call_count == 1
    assert "tool_execution_completed" not in orchestrator.active_metrics.timestamps
