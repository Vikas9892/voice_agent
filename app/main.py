import argparse
import sys
import time
import numpy as np

from app.audio.input import record_audio
from app.audio.playback import play_audio
from app.config import settings
from app.llm.main_model import LocalLLM
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS


def run_turn(
    stt: LocalSTT,
    llm: LocalLLM,
    tts: LocalTTS,
    duration_sec: float = 3.0,
    simulated_audio: np.ndarray | None = None,
) -> dict:
    """Execute one complete voice turn: Record -> STT -> LLM -> TTS -> Playback."""
    metrics = {}
    print("\n--- Listening... (Speak now) ---")
    t0 = time.perf_counter()

    if simulated_audio is not None:
        audio = simulated_audio
        record_ms = 0.0
    else:
        audio = record_audio(duration_sec=duration_sec, sample_rate=settings.audio_sample_rate)
        record_ms = (time.perf_counter() - t0) * 1000

    metrics["record_ms"] = round(record_ms, 1)

    # 1. STT
    t_stt_start = time.perf_counter()
    transcript = stt.transcribe(audio)
    metrics["stt_ms"] = round((time.perf_counter() - t_stt_start) * 1000, 1)
    print(f"[User]: {transcript if transcript else '(no speech detected)'}")

    if not transcript:
        return metrics

    # 2. LLM
    t_llm_start = time.perf_counter()
    response = llm.generate(transcript)
    metrics["llm_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 1)
    print(f"[Agent]: {response}")

    # 3. TTS
    t_tts_start = time.perf_counter()
    tts_audio = tts.synthesize(response, sample_rate=settings.tts_sample_rate)
    metrics["tts_ms"] = round((time.perf_counter() - t_tts_start) * 1000, 1)

    # 4. Playback
    t_play_start = time.perf_counter()
    play_audio(tts_audio, sample_rate=settings.tts_sample_rate, blocking=True)
    metrics["playback_ms"] = round((time.perf_counter() - t_play_start) * 1000, 1)

    metrics["total_turnaround_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(
        f"[Latency] STT: {metrics['stt_ms']}ms | LLM: {metrics['llm_ms']}ms | "
        f"TTS: {metrics['tts_ms']}ms | Total: {metrics['total_turnaround_ms']}ms"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Local Voice Agent - Phase 1 Minimal Loop")
    parser.add_argument("--duration", type=float, default=3.0, help="Microphone record duration per turn")
    parser.add_argument("--turns", type=int, default=1, help="Number of turns to execute (default: 1)")
    parser.add_argument("--simulate", action="store_true", help="Simulate audio input for automated testing")
    args = parser.parse_args()

    print("Initializing Phase 1 Voice Loop engines...")
    stt = LocalSTT(model_size=settings.stt_model_size, device=settings.stt_device, compute_type=settings.stt_compute_type)
    llm = LocalLLM(base_url=settings.llm_base_url, model=settings.llm_model)
    tts = LocalTTS(voice_model_path=settings.tts_voice_model)
    print("Engines ready.")

    simulated_audio = None
    if args.simulate:
        # 1-second 440Hz tone simulation
        simulated_audio = np.zeros(settings.audio_sample_rate, dtype=np.float32)

    for i in range(args.turns):
        print(f"\n=== Turn {i + 1}/{args.turns} ===")
        try:
            run_turn(stt, llm, tts, duration_sec=args.duration, simulated_audio=simulated_audio)
        except KeyboardInterrupt:
            print("\nExiting voice agent...")
            break


if __name__ == "__main__":
    main()
