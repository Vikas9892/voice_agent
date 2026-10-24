import asyncio
import time
import numpy as np

from app.audio.playback import play_audio, stop_playback
from app.audio.preprocessing import preprocess_audio
from app.config import settings
from app.conversation.cancellation import CancellationCoordinator
from app.conversation.gate import PlaybackGate
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager
from app.intent.filler_policy import FillerPolicy
from app.llm.filler_model import FillerModel
from app.llm.main_model import LocalLLM
from app.observability.events import TurnMetrics
from app.observability.metrics import MetricsLogger
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS


class StreamingVoiceOrchestrator:
    """Asynchronous orchestrator coordinating streaming VAD, turn taking, and interruptible playback."""

    def __init__(
        self,
        turn_manager: TurnManager,
        stt: LocalSTT,
        llm: LocalLLM,
        tts: LocalTTS,
        cancellation: CancellationCoordinator | None = None,
        executor=None,
        metrics_logger: MetricsLogger | None = None,
        filler_model: FillerModel | None = None,
        filler_policy: FillerPolicy | None = None,
        sample_rate: int = 16000,
        frame_samples: int = 512,
    ):
        self.turn_manager = turn_manager
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.cancellation = cancellation or CancellationCoordinator()
        self.executor = executor
        self.metrics_logger = metrics_logger or MetricsLogger()
        self.filler_model = filler_model
        self.filler_policy = filler_policy or FillerPolicy()
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.active_turn_task: asyncio.Task | None = None
        self.active_metrics: TurnMetrics | None = None

    async def _wait_playback(self, duration_sec: float, metrics: TurnMetrics) -> bool:
        """Wait for audio playback while watching for user barge-in. Returns False if interrupted."""
        elapsed = 0.0
        check_interval = 0.03
        while elapsed < duration_sec:
            if self.cancellation.is_cancelled():
                stop_playback()
                metrics.mark("playback_cancelled")
                return False
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        return True

    async def _handle_turn(self, audio_buffer: np.ndarray, turn_id: int):
        """Asynchronous execution of dual-track STT -> (Filler || Main LLM) -> TTS -> Playback."""
        metrics = self.metrics_logger.create_turn_metrics(turn_id)
        self.active_metrics = metrics
        metrics.mark("user_turn_end_detected")

        # 1. STT (run in bounded thread executor to keep event loop free)
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(self.executor, self.stt.transcribe, audio_buffer)
        metrics.mark("stt_final_transcript")

        if self.cancellation.is_cancelled() or not transcript.strip():
            self.turn_manager.reset()
            return

        print(f"\n[User (Turn {turn_id})]: {transcript}")
        gate = PlaybackGate(self.filler_policy)

        # 2. Speculative Filler Generation (Parallel Track)
        filler_task = None
        if self.filler_policy.is_eligible(transcript) and self.filler_model is not None:
            metrics.mark("filler_generation_started")

            async def _generate_filler():
                cand = await loop.run_in_executor(self.executor, self.filler_model.generate_filler, transcript)
                phrase = self.filler_policy.select_filler(transcript, cand)
                audio = await loop.run_in_executor(self.executor, self.tts.synthesize, phrase, settings.tts_sample_rate)
                metrics.mark("filler_generation_completed")
                return phrase, audio

            filler_task = asyncio.create_task(_generate_filler())

        # 3. Main LLM Generation
        metrics.mark("main_llm_request_started")
        response = await loop.run_in_executor(self.executor, self.llm.generate, transcript)
        metrics.mark("main_llm_first_token")
        metrics.mark("main_llm_generation_completed")

        if self.cancellation.is_cancelled() or not response.strip():
            self.turn_manager.reset()
            return

        print(f"[Agent (Turn {turn_id})]: {response}")

        # Check if filler is ready to bridge latency gap before main TTS starts
        if filler_task and filler_task.done() and not gate.main_ready:
            try:
                f_phrase, f_audio = filler_task.result()
                if gate.should_play_filler(f_phrase) and len(f_audio) > 0:
                    self.turn_manager.set_agent_speaking()
                    metrics.mark("agent_first_audio_played")
                    print(f"[Agent Filler]: {f_phrase}")
                    play_audio(f_audio, sample_rate=settings.tts_sample_rate, blocking=False)
                    if not await self._wait_playback(len(f_audio) / settings.tts_sample_rate, metrics):
                        self.metrics_logger.record_turn(metrics)
                        return
            except Exception:
                pass

        # 4. Main TTS Synthesis
        metrics.mark("tts_request_started")
        tts_audio = await loop.run_in_executor(self.executor, self.tts.synthesize, response, settings.tts_sample_rate)
        metrics.mark("tts_first_audio_ready")
        gate.mark_main_ready()

        if self.cancellation.is_cancelled() or len(tts_audio) == 0:
            self.turn_manager.reset()
            return

        # 5. Playback Main Answer
        self.turn_manager.set_agent_speaking()
        if "agent_first_audio_played" not in metrics.timestamps:
            metrics.mark("agent_first_audio_played")
        metrics.mark("main_answer_first_audio_played")
        play_audio(tts_audio, sample_rate=settings.tts_sample_rate, blocking=False)

        completed_ok = await self._wait_playback(len(tts_audio) / settings.tts_sample_rate, metrics)
        if completed_ok and not self.cancellation.is_cancelled():
            self.turn_manager.reset()
            summary = self.metrics_logger.record_turn(metrics)
            print(f"[Turn {turn_id} complete. Perceived: {summary.get('perceived_latency_ms')}ms | Main: {summary.get('main_answer_latency_ms')}ms]")
        else:
            self.metrics_logger.record_turn(metrics)

    def _audio_callback(self, indata, frames, time_info, status):
        """SoundDevice InputStream callback pushing preprocessed 32ms frames to asyncio queue."""
        if status:
            pass
        frame = preprocess_audio(indata[:, 0].copy())
        self.audio_queue.put_nowait(frame)

    async def run_streaming(self, max_turns: int | None = None):
        """Run continuous live streaming audio loop with real-time barge-in."""
        import sounddevice as sd
        completed_turns = 0
        loop = asyncio.get_running_loop()

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=lambda indata, frames, time_info, status: loop.call_soon_threadsafe(
                self._audio_callback, indata, frames, time_info, status
            ),
        )

        print("\n=== Live Streaming Voice Agent Active (Barge-In Enabled) ===")
        print("Start speaking into your microphone. Say anything, pause, or interrupt the agent.")

        with stream:
            while max_turns is None or completed_turns < max_turns:
                frame = await self.audio_queue.get()
                new_state, completed_audio = self.turn_manager.process_frame(frame)

                if new_state == TurnState.INTERRUPTED:
                    # User interrupted agent during playback!
                    if self.active_metrics:
                        self.active_metrics.mark("interruption_detected")
                    self.cancellation.cancel_current_turn()
                    if self.active_turn_task and not self.active_turn_task.done():
                        self.active_turn_task.cancel()
                    self.turn_manager.state = TurnState.USER_SPEAKING

                elif completed_audio is not None and len(completed_audio) > 0:
                    turn_id = self.cancellation.start_new_turn()
                    completed_turns += 1
                    self.active_turn_task = asyncio.create_task(self._handle_turn(completed_audio, turn_id))
