import numpy as np
from typing import TYPE_CHECKING, Optional
from app.audio.vad import SileroVAD
from app.conversation.state import TurnState

if TYPE_CHECKING:
    from app.audio.echo_cancellation import AcousticEchoSuppressor


class TurnManager:
    """Manages conversational turn state transitions, endpointing, and barge-in detection."""

    def __init__(
        self,
        vad: SileroVAD,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 500,
        barge_in_frames: int = 3,
        echo_suppressor: Optional["AcousticEchoSuppressor"] = None,
    ):
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.frame_duration_ms = (frame_samples / sample_rate) * 1000  # 32ms for 512 @ 16kHz

        self.min_speech_frames = max(int(min_speech_duration_ms / self.frame_duration_ms), 1)
        self.min_silence_frames = max(int(min_silence_duration_ms / self.frame_duration_ms), 1)
        self.barge_in_frames = barge_in_frames
        self.echo_suppressor = echo_suppressor

        self.state = TurnState.IDLE
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.audio_chunks = []

    def reset(self):
        """Reset conversation turn to IDLE."""
        self.state = TurnState.IDLE
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.audio_chunks.clear()
        self.vad.reset()
        if self.echo_suppressor:
            self.echo_suppressor.set_playback_active(False)

    def set_agent_speaking(self):
        """Signal that agent has started audio playback."""
        self.state = TurnState.AGENT_SPEAKING
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.audio_chunks.clear()
        if self.echo_suppressor:
            self.echo_suppressor.set_playback_active(True)

    def process_frame(self, frame: np.ndarray) -> tuple[TurnState, np.ndarray | None]:
        """
        Process a 32ms audio frame and evaluate state transitions.
        Returns (new_state, completed_audio_buffer_or_none).
        """
        if self.echo_suppressor and self.state == TurnState.AGENT_SPEAKING:
            frame, is_echo = self.echo_suppressor.process_frame(frame, is_agent_speaking=True)
            if is_echo:
                is_speech = False
            else:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
        else:
            is_speech = self.vad.is_speech(frame, self.sample_rate)

        if self.state == TurnState.IDLE:
            if is_speech:
                self.consecutive_speech_frames += 1
                self.audio_chunks.append(frame)
                if self.consecutive_speech_frames >= self.min_speech_frames:
                    self.state = TurnState.USER_SPEAKING
                    self.consecutive_silence_frames = 0
            else:
                self.consecutive_speech_frames = 0
                self.audio_chunks.clear()
            return self.state, None

        elif self.state == TurnState.USER_SPEAKING:
            self.audio_chunks.append(frame)
            if not is_speech:
                self.consecutive_silence_frames = 1
                self.consecutive_speech_frames = 0
                self.state = TurnState.USER_PAUSED
            else:
                self.consecutive_speech_frames += 1
            return self.state, None

        elif self.state == TurnState.USER_PAUSED:
            self.audio_chunks.append(frame)
            if is_speech:
                # User continued speaking after brief pause
                self.state = TurnState.USER_SPEAKING
                self.consecutive_silence_frames = 0
                self.consecutive_speech_frames = 1
                return self.state, None
            else:
                self.consecutive_silence_frames += 1
                if self.consecutive_silence_frames >= self.min_silence_frames:
                    # Endpointing threshold reached: Turn complete!
                    self.state = TurnState.USER_TURN_FINALIZING
                    completed_audio = np.concatenate(self.audio_chunks) if self.audio_chunks else np.zeros(0, dtype=np.float32)
                    self.audio_chunks.clear()
                    return self.state, completed_audio
                return self.state, None

        elif self.state == TurnState.AGENT_SPEAKING:
            # Check for barge-in
            if is_speech:
                self.consecutive_speech_frames += 1
                if self.consecutive_speech_frames >= self.barge_in_frames:
                    self.state = TurnState.INTERRUPTED
                    self.audio_chunks = [frame]
                    return self.state, None
            else:
                self.consecutive_speech_frames = 0
            return self.state, None

        return self.state, None
