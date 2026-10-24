from dataclasses import dataclass, field
import time


@dataclass
class TurnMetrics:
    """Monotonic latency checkpoints and calculated metrics for a single conversational turn."""

    session_id: str
    turn_id: int
    t_start: float = field(default_factory=time.perf_counter)
    timestamps: dict[str, float] = field(default_factory=dict)

    def mark(self, checkpoint_name: str) -> float:
        """Record high-precision monotonic timestamp for a named pipeline checkpoint."""
        now = time.perf_counter()
        self.timestamps[checkpoint_name] = now
        return now

    def delta_ms(self, start_checkpoint: str, end_checkpoint: str) -> float | None:
        """Calculate elapsed milliseconds between two recorded checkpoints."""
        t0 = self.timestamps.get(start_checkpoint)
        t1 = self.timestamps.get(end_checkpoint)
        if t0 is not None and t1 is not None:
            return round((t1 - t0) * 1000, 2)
        return None

    def summary(self) -> dict:
        """Produce structured latency KPI summary."""
        kpis = {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "stt_ms": self.delta_ms("user_turn_end_detected", "stt_final_transcript"),
            "llm_ttft_ms": self.delta_ms("main_llm_request_started", "main_llm_first_token"),
            "llm_total_ms": self.delta_ms("main_llm_request_started", "main_llm_generation_completed"),
            "tts_first_audio_ms": self.delta_ms("tts_request_started", "tts_first_audio_ready"),
            # Perceived Latency: User turn end to first audible agent sound
            "perceived_latency_ms": self.delta_ms("user_turn_end_detected", "agent_first_audio_played"),
            # Main Response Latency: User turn end to main answer audio
            "main_answer_latency_ms": self.delta_ms("user_turn_end_detected", "main_answer_first_audio_played"),
            # Barge-In Cutoff Latency
            "barge_in_cutoff_ms": self.delta_ms("interruption_detected", "playback_cancelled"),
        }
        # If agent audio was played, calculate total turnaround from speech end
        t_end = self.timestamps.get("agent_first_audio_played") or self.timestamps.get("main_llm_generation_completed")
        if t_end is not None and "user_turn_end_detected" in self.timestamps:
            kpis["total_turnaround_ms"] = round((t_end - self.timestamps["user_turn_end_detected"]) * 1000, 2)
        else:
            kpis["total_turnaround_ms"] = None
        return kpis
