import argparse
import json
import time
import numpy as np

from app.config import settings
from app.llm.main_model import LocalLLM
from app.observability.metrics import MetricsLogger
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS


TEST_UTTERANCES = [
    "What is the weather like today?",
    "Can you explain how a low latency voice agent works?",
    "Tell me a short joke.",
    "Compare CPU and GPU inference performance.",
    "What time is it in Tokyo right now?",
]


def run_benchmark(num_turns: int = 5, output_file: str = "benchmarks/results.json") -> dict:
    """Run automated benchmark suite measuring STT, LLM, TTS, and turnaround p50/p90/p95."""
    print(f"\n=======================================================")
    print(f"Starting Voice Pipeline Latency Benchmark (N = {num_turns})")
    print(f"Target CPU: {settings.stt_device} | STT: {settings.stt_model_size} | LLM: {settings.llm_model}")
    print(f"=======================================================\n")

    stt = LocalSTT(model_size=settings.stt_model_size, device=settings.stt_device, compute_type=settings.stt_compute_type)
    llm = LocalLLM(base_url=settings.llm_base_url, model=settings.llm_model)
    tts = LocalTTS(voice_model_path=settings.tts_voice_model)
    logger = MetricsLogger(session_id="benchmark_suite")

    # Synthetic 1.5s audio array for repeatable STT testing
    test_audio = np.zeros(int(settings.audio_sample_rate * 1.5), dtype=np.float32)

    for i in range(num_turns):
        turn_id = i + 1
        prompt = TEST_UTTERANCES[i % len(TEST_UTTERANCES)]
        m = logger.create_turn_metrics(turn_id)
        m.mark("user_turn_end_detected")

        # 1. STT
        stt.transcribe(test_audio)
        m.mark("stt_final_transcript")

        # 2. LLM
        m.mark("main_llm_request_started")
        response = llm.generate(prompt)
        m.mark("main_llm_first_token")
        m.mark("main_llm_generation_completed")

        # 3. TTS
        m.mark("tts_request_started")
        audio = tts.synthesize(response)
        m.mark("tts_first_audio_ready")
        m.mark("agent_first_audio_played")
        m.mark("main_answer_first_audio_played")

        summary = logger.record_turn(m)
        print(f"Turn {turn_id:02d}/{num_turns:02d} | STT: {summary['stt_ms']}ms | LLM: {summary['llm_total_ms']}ms | TTS: {summary['tts_first_audio_ms']}ms | Turnaround: {summary['total_turnaround_ms']}ms")

    kpis = logger.aggregate_kpis()
    result = {
        "timestamp": time.time(),
        "trials": num_turns,
        "config": {
            "stt_model": settings.stt_model_size,
            "stt_compute": settings.stt_compute_type,
            "llm_model": settings.llm_model,
            "tts_voice": settings.tts_voice_model,
        },
        "percentiles": kpis,
    }

    print("\n=== Benchmark Summary (p50 / p90 / p95 in ms) ===")
    print(f"{'Metric':<24} | {'p50':>8} | {'p90':>8} | {'p95':>8}")
    print("-" * 54)
    for metric, stats in kpis.items():
        print(f"{metric:<24} | {stats['p50']:>8.1f} | {stats['p90']:>8.1f} | {stats['p95']:>8.1f}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved benchmark results to {output_file}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Agent Latency Benchmark")
    parser.add_argument("--turns", type=int, default=5, help="Number of benchmark trials")
    parser.add_argument("--out", type=str, default="benchmarks/results.json", help="Output JSON path")
    args = parser.parse_args()
    run_benchmark(num_turns=args.turns, output_file=args.out)
