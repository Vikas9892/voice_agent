import json
import numpy as np
from app.observability.events import TurnMetrics


class MetricsLogger:
    """Collects, logs, and aggregates latency metrics across conversational turns."""

    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.history: list[dict] = []

    def create_turn_metrics(self, turn_id: int) -> TurnMetrics:
        """Instantiate a new monotonic timer record for a turn."""
        return TurnMetrics(session_id=self.session_id, turn_id=turn_id)

    def record_turn(self, metrics: TurnMetrics) -> dict:
        """Record and log summary KPIs for a completed turn."""
        summary = metrics.summary()
        self.history.append(summary)
        print(f"[Metrics JSON] {json.dumps(summary)}")
        return summary

    def calculate_percentiles(self, values: list[float]) -> dict[str, float]:
        """Calculate p50, p90, and p95 across numeric metric samples."""
        valid = [v for v in values if v is not None]
        if not valid:
            return {"count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
        arr = np.array(valid)
        return {
            "count": len(valid),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
        }

    def aggregate_kpis(self) -> dict[str, dict]:
        """Compute aggregate p50/p90/p95 statistics across all recorded turns."""
        metrics_keys = [
            "stt_ms",
            "llm_ttft_ms",
            "llm_total_ms",
            "tts_first_audio_ms",
            "perceived_latency_ms",
            "main_answer_latency_ms",
            "total_turnaround_ms",
        ]
        report = {}
        for key in metrics_keys:
            samples = [item.get(key) for item in self.history if item.get(key) is not None]
            report[key] = self.calculate_percentiles(samples)
        return report
