import time
from app.observability.events import TurnMetrics
from app.observability.metrics import MetricsLogger


def test_turn_metrics_checkpoints():
    m = TurnMetrics(session_id="test_sess", turn_id=1)
    m.mark("user_turn_end_detected")
    time.sleep(0.01)
    m.mark("stt_final_transcript")
    time.sleep(0.01)
    m.mark("agent_first_audio_played")

    stt_latency = m.delta_ms("user_turn_end_detected", "stt_final_transcript")
    assert stt_latency is not None
    assert stt_latency >= 9.0  # At least 10ms

    summary = m.summary()
    assert summary["session_id"] == "test_sess"
    assert summary["turn_id"] == 1
    assert summary["stt_ms"] == stt_latency
    assert summary["perceived_latency_ms"] is not None


def test_metrics_logger_aggregation():
    logger = MetricsLogger(session_id="bench_sess")
    # Simulate 5 turns with known stt values
    for i, val in enumerate([100.0, 120.0, 110.0, 150.0, 200.0]):
        m = logger.create_turn_metrics(turn_id=i + 1)
        m.timestamps["user_turn_end_detected"] = 1.0
        m.timestamps["stt_final_transcript"] = 1.0 + (val / 1000.0)
        logger.record_turn(m)

    stats = logger.aggregate_kpis()
    stt_stats = stats["stt_ms"]
    assert stt_stats["count"] == 5
    assert stt_stats["p50"] == 120.0
    assert stt_stats["p90"] > 150.0
    assert stt_stats["p95"] > stt_stats["p90"]
