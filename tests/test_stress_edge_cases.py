import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.audio.preprocessing import preprocess_audio
from app.audio.vad import SileroVAD
from app.conversation.cancellation import CancellationCoordinator
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager
from app.intent.readiness import IntentReadinessDetector


@pytest.mark.asyncio
async def test_rapid_barge_in_stress():
    """Stress test: 10 back-to-back rapid barge-in interruptions during active speech."""
    mock_vad = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "rapid interruption query"
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Long ongoing answer from agent..."
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(22050, dtype=np.float32)

    turn_manager = TurnManager(vad=mock_vad)
    coord = CancellationCoordinator()

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=mock_stt,
        llm=mock_llm,
        tts=mock_tts,
        cancellation=coord,
    )

    with patch("app.conversation.orchestrator.play_audio"), \
         patch("app.conversation.cancellation.stop_playback") as mock_stop:

        for i in range(10):
            turn_id = coord.start_new_turn()
            assert not coord.is_cancelled()
            assert coord.current_turn_id == i + 1

            # Launch playback turn
            dummy_audio = np.zeros(8000, dtype=np.float32)
            task = asyncio.create_task(orchestrator._handle_turn(dummy_audio, turn_id=turn_id))

            # Rapid interruption after only 15ms
            await asyncio.sleep(0.015)
            coord.cancel_current_turn()
            assert coord.is_cancelled()

            await task
            assert mock_stop.call_count >= i + 1


def test_background_office_noise_rejection():
    """Verify stationary background noise and short transient clicks do not trigger false turns."""
    vad = SileroVAD(threshold=0.5)
    tm = TurnManager(
        vad=vad,
        sample_rate=16000,
        frame_samples=512,
        min_speech_duration_ms=200,   # ~6 frames
        min_silence_duration_ms=500,
    )

    # 1. Stationary low-level background noise (RMS ~0.005, typical fan / HVAC noise)
    fan_noise_frame = np.random.uniform(-0.005, 0.005, 512).astype(np.float32)
    for _ in range(20):
        state, audio = tm.process_frame(fan_noise_frame)
        assert state == TurnState.IDLE
        assert audio is None

    # 2. Transient click (e.g. keyboard tap or mouse click lasting 1 frame = 32ms)
    transient_frame = np.random.uniform(-0.15, 0.15, 512).astype(np.float32)
    state, audio = tm.process_frame(transient_frame)
    # Even if RMS spikes for 1 frame, 1 frame is less than min_speech_frames (6 frames)
    assert state == TurnState.IDLE
    assert audio is None

    # Return to silence immediately clears transient chunk
    state, audio = tm.process_frame(fan_noise_frame)
    assert state == TurnState.IDLE
    assert len(tm.audio_chunks) == 0


def test_spectral_noise_gating_preserves_speech_while_clearing_noise():
    """Verify preprocessing suppresses low-level noise floor while preserving speech signals."""
    # Low-level noise floor (RMS < 0.005)
    dirty_noise = np.random.normal(0, 0.003, 512).astype(np.float32) + 0.05  # with DC offset
    cleaned_noise = preprocess_audio(dirty_noise)
    # DC offset eliminated and low-energy noise gated
    assert abs(float(np.mean(cleaned_noise))) < 1e-4
    assert np.max(np.abs(cleaned_noise)) < np.max(np.abs(dirty_noise))

    # Strong speech signal (sine wave @ 440Hz + ambient noise)
    t = np.linspace(0, 0.032, 512, endpoint=False)
    speech = (0.2 * np.sin(2 * np.pi * 440 * t) + dirty_noise).astype(np.float32)
    cleaned_speech = preprocess_audio(speech)
    # Speech energy is retained
    assert np.max(np.abs(cleaned_speech)) > 0.15


def test_compound_question_natural_pause_handling():
    """Verify multi-clause utterance with a 320ms pause does not trigger premature endpointing."""
    mock_vad = MagicMock()
    frame = np.zeros(512, dtype=np.float32)

    tm = TurnManager(
        vad=mock_vad,
        sample_rate=16000,
        frame_samples=512,
        min_speech_duration_ms=160,   # 5 frames
        min_silence_duration_ms=500,  # 15 frames
    )

    # Clause 1: User speaks for 8 frames (~256ms)
    mock_vad.is_speech.return_value = True
    for _ in range(8):
        state, audio = tm.process_frame(frame)
    assert state == TurnState.USER_SPEAKING

    # Natural mid-sentence pause: 10 frames of silence (~320ms, less than 500ms min silence)
    mock_vad.is_speech.return_value = False
    for _ in range(10):
        state, audio = tm.process_frame(frame)
        assert state == TurnState.USER_PAUSED
        assert audio is None  # Should NOT finalize turn yet!

    # Clause 2: User resumes speaking ("and also...") for 8 frames
    mock_vad.is_speech.return_value = True
    for _ in range(8):
        state, audio = tm.process_frame(frame)
    assert state == TurnState.USER_SPEAKING

    # Final pause: 16 frames of silence (~512ms >= 500ms threshold)
    mock_vad.is_speech.return_value = False
    final_state = None
    completed_audio = None
    for _ in range(16):
        state, audio = tm.process_frame(frame)
        if audio is not None:
            completed_audio = audio
            final_state = state

    assert final_state == TurnState.USER_TURN_FINALIZING
    assert completed_audio is not None
    # All 8 + 10 + 8 + 15 = 41 frames preserved in completed buffer
    assert len(completed_audio) == 41 * 512


def test_compound_intent_readiness_evaluation():
    """Verify IntentReadinessDetector evaluates compound sentences accurately."""
    detector = IntentReadinessDetector(min_words=3, threshold=0.7)

    # Compound query with coordinator
    q1 = "can you explain how whisper works and also compare it to vosk"
    res1 = detector.evaluate(q1)
    assert res1.is_ready
    assert res1.is_question
    assert res1.score >= 0.7

    # Multi-clause instruction
    q2 = "tell me a quick joke but make it about software engineering"
    res2 = detector.evaluate(q2)
    assert res2.is_ready


def test_thread_pool_executor_concurrency_saturation():
    """Verify bounded thread pool handles burst of 12 concurrent inference tasks without leaks."""
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-worker")

    def dummy_inference(idx: int) -> int:
        import time
        time.sleep(0.01)
        return idx * 2

    futures = [executor.submit(dummy_inference, i) for i in range(12)]
    results = [f.result(timeout=2.0) for f in futures]
    assert results == [i * 2 for i in range(12)]

    executor.shutdown(wait=True)
