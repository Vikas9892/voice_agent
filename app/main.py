import argparse
import asyncio
import sys
import time
import numpy as np

from app.audio.echo_cancellation import AcousticEchoSuppressor
from app.audio.input import record_audio
from app.audio.playback import play_audio
from app.audio.vad import SileroVAD
from app.config import settings
from app.conversation.memory import ConversationMemory
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager
from app.intent.filler_policy import FillerPolicy
from app.intent.readiness import IntentReadinessDetector
from app.lifecycle import PrewarmingManager
from app.llm.filler_model import FillerModel
from app.llm.main_model import LocalLLM
from app.stt.local_stt import LocalSTT
from app.tools.builtin import FastToolMatcher
from app.tts.local_tts import LocalTTS


def run_turn(
    stt: LocalSTT,
    llm: LocalLLM,
    tts: LocalTTS,
    duration_sec: float = 3.0,
    simulated_audio: np.ndarray | None = None,
    tool_matcher: FastToolMatcher | None = None,
) -> dict:
    """Execute one complete voice turn: Record -> STT -> (Tool || LLM) -> TTS -> Playback."""
    metrics = {}
    matcher = tool_matcher or FastToolMatcher()
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

    # 2. Tool Match or LLM
    t_llm_start = time.perf_counter()
    tool_match = tool_matcher.match_and_execute(transcript) if tool_matcher else None
    if tool_match:
        tool_name, tool_raw, response = tool_match
        print(f"[Tool: {tool_name}]: {tool_raw}")
    else:
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
        f"[Latency] STT: {metrics['stt_ms']}ms | LLM/Tool: {metrics['llm_ms']}ms | "
        f"TTS: {metrics['tts_ms']}ms | Total: {metrics['total_turnaround_ms']}ms"
    )
    return metrics


async def run_with_dashboard(orchestrator: StreamingVoiceOrchestrator, port: int = 8000, max_turns: int | None = None):
    """Run streaming voice agent with embedded FastAPI WebSocket dashboard."""
    import uvicorn
    from app.api.server import create_app
    app = create_app(orchestrator=orchestrator)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    print(f"\n🚀 Real-time Voice Agent Dashboard active at: http://localhost:{port}")
    await asyncio.gather(
        server.serve(),
        orchestrator.run_streaming(max_turns=max_turns),
    )


def main():
    parser = argparse.ArgumentParser(description="Local Voice Agent - Production Full-Stack Voice AI")
    parser.add_argument("--streaming", action="store_true", help="Run live streaming audio mode with barge-in")
    parser.add_argument("--dashboard", action="store_true", help="Launch live web dashboard UI on port 8000")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port (default: 8000)")
    parser.add_argument("--duration", type=float, default=3.0, help="Record duration for batch mode")
    parser.add_argument("--turns", type=int, default=1, help="Number of turns to execute (default: 1)")
    parser.add_argument("--simulate", action="store_true", help="Simulate audio input for automated testing")
    parser.add_argument("--no-prewarm", action="store_true", help="Skip model prewarming on startup")
    args = parser.parse_args()

    print("Initializing Voice Agent engines...")
    vad = SileroVAD(
        threshold=settings.vad_threshold,
        intra_op_threads=settings.onnx_intra_op_threads,
        inter_op_threads=settings.onnx_inter_op_threads,
    )
    stt = LocalSTT(
        model_size=settings.stt_model_size,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        beam_size=settings.stt_beam_size,
        cpu_threads=settings.stt_cpu_threads,
    )
    memory = ConversationMemory()
    llm = LocalLLM(base_url=settings.llm_base_url, model=settings.llm_model, memory=memory)
    tts = LocalTTS(voice_model_path=settings.tts_voice_model)
    filler_model = FillerModel(base_url=settings.llm_base_url)
    filler_policy = FillerPolicy()
    readiness_detector = IntentReadinessDetector()
    echo_suppressor = AcousticEchoSuppressor()
    tool_matcher = FastToolMatcher()

    lifecycle = PrewarmingManager(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        max_workers=settings.executor_max_workers,
    )
    if not args.no_prewarm and not args.simulate:
        lifecycle.warmup_all()

    print("Engines ready.")

    try:
        if args.streaming or args.dashboard:
            from app.api.server import broadcaster
            turn_manager = TurnManager(
                vad=vad,
                sample_rate=settings.audio_sample_rate,
                frame_samples=settings.audio_chunk_samples,
                min_speech_duration_ms=settings.vad_min_speech_duration_ms,
                min_silence_duration_ms=settings.vad_min_silence_duration_ms,
                echo_suppressor=echo_suppressor,
            )
            orchestrator = StreamingVoiceOrchestrator(
                turn_manager=turn_manager,
                stt=stt,
                llm=llm,
                tts=tts,
                executor=lifecycle.executor,
                filler_model=filler_model,
                filler_policy=filler_policy,
                readiness_detector=readiness_detector,
                sample_rate=settings.audio_sample_rate,
                frame_samples=settings.audio_chunk_samples,
                broadcaster=broadcaster if args.dashboard else None,
                tool_matcher=tool_matcher,
            )
            try:
                if args.dashboard:
                    asyncio.run(run_with_dashboard(orchestrator, port=args.port, max_turns=args.turns if args.turns > 1 else None))
                else:
                    asyncio.run(orchestrator.run_streaming(max_turns=args.turns if args.turns > 1 else None))
            except KeyboardInterrupt:
                print("\nShutting down streaming voice agent...")
        else:
            simulated_audio = None
            if args.simulate:
                simulated_audio = np.zeros(settings.audio_sample_rate, dtype=np.float32)

            for i in range(args.turns):
                print(f"\n=== Turn {i + 1}/{args.turns} ===")
                try:
                    run_turn(stt, llm, tts, duration_sec=args.duration, simulated_audio=simulated_audio, tool_matcher=tool_matcher)
                except KeyboardInterrupt:
                    print("\nExiting voice agent...")
                    break
    finally:
        lifecycle.shutdown()


if __name__ == "__main__":
    main()
