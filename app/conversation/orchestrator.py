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
from app.intent.readiness import IntentReadinessDetector
from app.llm.filler_model import FillerModel
from app.llm.main_model import LocalLLM
from app.observability.events import TurnMetrics
from app.observability.metrics import MetricsLogger
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS


class StreamingVoiceOrchestrator:
    """Asynchronous orchestrator coordinating streaming VAD, speculative execution, and interruptible playback."""

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
        readiness_detector: IntentReadinessDetector | None = None,
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
        self.readiness_detector = readiness_detector or IntentReadinessDetector()
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.active_turn_task: asyncio.Task | None = None
        self.active_metrics: TurnMetrics | None = None
        self.gate: PlaybackGate = PlaybackGate(self.filler_policy)
        self.speculative_task: asyncio.Task | None = None
        self.speculative_checked_frames: int = 0

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

    async def _check_and_trigger_speculation(self, audio_snapshot: np.ndarray, current_turn_id: int):
        """Asynchronously analyze partial STT and trigger speculative filler if intent is ready."""
        loop = asyncio.get_running_loop()
        try:
            partial = await loop.run_in_executor(self.executor, self.stt.transcribe, audio_snapshot)
        except Exception:
            return

        if self.cancellation.is_cancelled() or not partial.strip():
            return

        res = self.readiness_detector.evaluate(partial)
        if res.is_ready and self.filler_policy.is_eligible(partial) and self.speculative_task is None:
            if self.active_metrics:
                self.active_metrics.mark("speculative_filler_triggered")

            async def _spec_gen():
                cand = await loop.run_in_executor(self.executor, self.filler_model.generate_filler, partial)
                phrase = self.filler_policy.select_filler(partial, cand)
                audio = await loop.run_in_executor(self.executor, self.tts.synthesize, phrase, settings.tts_sample_rate)
                return phrase, audio

            self.speculative_task = asyncio.create_task(_spec_gen())

    async def _handle_turn(
        self,
        audio_buffer: np.ndarray,
        turn_id: int,
        speculative_task: asyncio.Task | None = None,
    ):
        """Asynchronous execution of dual-track STT -> (Filler || Main LLM) -> TTS -> Playback."""
        metrics = self.metrics_logger.create_turn_metrics(turn_id)
        self.active_metrics = metrics
        metrics.mark("user_turn_end_detected")
        self.gate.set_user_speaking(False)
        self.gate.main_ready = False
        self.gate.filler_played = False
        self.readiness_detector.reset()

        # 1. STT (run in bounded thread executor to keep event loop free)
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(self.executor, self.stt.transcribe, audio_buffer)
        metrics.mark("stt_final_transcript")

        if self.cancellation.is_cancelled() or not transcript.strip():
            if speculative_task and not speculative_task.done():
                speculative_task.cancel()
            self.turn_manager.reset()
            return

        print(f"\n[User (Turn {turn_id})]: {transcript}")

        # 2. Speculative / Dual-Track Filler Generation
        filler_task = speculative_task
        if filler_task is None and self.filler_policy.is_eligible(transcript) and self.filler_model is not None:
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

        async def _call_main_llm():
            return await loop.run_in_executor(self.executor, self.llm.generate, transcript)

        main_llm_task = asyncio.create_task(_call_main_llm())

        # Check if speculative or background filler is ready to bridge latency gap
        if filler_task is not None and not self.gate.main_ready:
            try:
                await asyncio.wait({filler_task}, timeout=0.08)
                if filler_task.done() and not self.gate.main_ready:
                    f_phrase, f_audio = filler_task.result()
                    if self.gate.should_play_filler(f_phrase) and len(f_audio) > 0:
                        self.turn_manager.set_agent_speaking()
                        metrics.mark("agent_first_audio_played")
                        print(f"[Agent Filler]: {f_phrase}")
                        play_audio(f_audio, sample_rate=settings.tts_sample_rate, blocking=False)
                        interrupted = not await self._wait_playback(len(f_audio) / settings.tts_sample_rate, metrics)
                        if interrupted:
                            main_llm_task.cancel()
                            self.metrics_logger.record_turn(metrics)
                            return
            except Exception:
                pass

        # Await Main LLM response
        response = await main_llm_task
        metrics.mark("main_llm_first_token")
        metrics.mark("main_llm_generation_completed")

        if self.cancellation.is_cancelled() or not response.strip():
            self.turn_manager.reset()
            return

        print(f"[Agent (Turn {turn_id})]: {response}")

        # 4. Main TTS Synthesis
        metrics.mark("tts_request_started")
        tts_audio = await loop.run_in_executor(self.executor, self.tts.synthesize, response, settings.tts_sample_rate)
        metrics.mark("tts_first_audio_ready")
        self.gate.mark_main_ready()

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
        """Run continuous live streaming audio loop with real-time barge-in and speculative execution."""
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

        print("\n=== Live Streaming Voice Agent Active (Barge-In & Speculative Execution Enabled) ===")
        print("Start speaking into your microphone. Say anything, pause, or interrupt the agent.")

        with stream:
            while max_turns is None or completed_turns < max_turns:
                frame = await self.audio_queue.get()
                new_state, completed_audio = self.turn_manager.process_frame(frame)

                if new_state == TurnState.INTERRUPTED:
                    # User interrupted agent during playback!
                    self.gate.set_user_speaking(True)
                    if self.active_metrics:
                        self.active_metrics.mark("interruption_detected")
                    self.cancellation.cancel_current_turn()
                    if self.active_turn_task and not self.active_turn_task.done():
                        self.active_turn_task.cancel()
                    if self.speculative_task and not self.speculative_task.done():
                        self.speculative_task.cancel()
                    self.speculative_task = None
                    self.readiness_detector.reset()
                    self.turn_manager.state = TurnState.USER_SPEAKING

                elif new_state in (TurnState.USER_SPEAKING, TurnState.USER_PAUSED):
                    self.gate.set_user_speaking(True)
                    # Speculative trigger on partial speech (300-600ms before turn finalization)
                    n_frames = len(self.turn_manager.audio_chunks)
                    if (
                        self.filler_model is not None
                        and self.speculative_task is None
                        and n_frames >= 20  # ~640ms min accumulated speech
                        and (n_frames - self.speculative_checked_frames) >= 12  # every ~380ms
                    ):
                        self.speculative_checked_frames = n_frames
                        snapshot = np.concatenate(self.turn_manager.audio_chunks)
                        asyncio.create_task(self._check_and_trigger_speculation(snapshot, self.cancellation.turn_id))

                elif completed_audio is not None and len(completed_audio) > 0:
                    self.gate.set_user_speaking(False)
                    self.speculative_checked_frames = 0
                    turn_id = self.cancellation.start_new_turn()
                    completed_turns += 1
                    spec_task = self.speculative_task
                    self.speculative_task = None
                    self.active_turn_task = asyncio.create_task(
                        self._handle_turn(completed_audio, turn_id, speculative_task=spec_task)
                    )

