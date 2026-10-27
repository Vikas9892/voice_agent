import os
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

from app.audio.vad import SileroVAD
from app.config import settings
from app.llm.main_model import LocalLLM
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS


def configure_cpu_affinity(num_threads: int | None = None):
    """
    Set CPU thread budget and OpenMP/BLAS limits to avoid core oversubscription on AMD Ryzen 5 CPU.
    """
    threads = num_threads or getattr(settings, "stt_cpu_threads", 4)
    threads_str = str(threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, threads_str)


class PrewarmingManager:
    """Manages application startup lifecycle, model prewarming, and bounded execution threads."""

    def __init__(
        self,
        vad: SileroVAD,
        stt: LocalSTT,
        llm: LocalLLM,
        tts: LocalTTS,
        max_workers: int = 2,
    ):
        configure_cpu_affinity()
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        # Bounded thread pool keeps heavy C++ inference off the asyncio event loop
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="voice-worker")
        self.warmup_timings: dict[str, float] = {}

    def warmup_vad(self) -> float:
        """Run dummy 32ms frame through VAD to warm up ONNX runtime."""
        t0 = time.perf_counter()
        dummy_frame = np.zeros(512, dtype=np.float32)
        self.vad.is_speech(dummy_frame)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.warmup_timings["vad_ms"] = duration_ms
        return duration_ms

    def warmup_stt(self) -> float:
        """Run short 0.5s audio array through Whisper to initialize CTranslate2 buffers."""
        t0 = time.perf_counter()
        dummy_audio = np.zeros(8000, dtype=np.float32)
        self.stt.transcribe(dummy_audio)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.warmup_timings["stt_ms"] = duration_ms
        return duration_ms

    def warmup_llm(self) -> float:
        """Send a single-word prompt to load Ollama weights from disk into RAM."""
        t0 = time.perf_counter()
        self.llm.generate("hi")
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.warmup_timings["llm_ms"] = duration_ms
        return duration_ms

    def warmup_tts(self) -> float:
        """Synthesize a short word to warm up Piper ONNX synthesis session."""
        t0 = time.perf_counter()
        self.tts.synthesize("ready")
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.warmup_timings["tts_ms"] = duration_ms
        return duration_ms

    def warmup_all(self) -> dict[str, float]:
        """Preload all model pipelines and record cold-start duration."""
        t_total = time.perf_counter()
        print("Prewarming models to eliminate first-turn cold-start latency...")
        self.warmup_vad()
        self.warmup_stt()
        self.warmup_llm()
        self.warmup_tts()
        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        self.warmup_timings["total_cold_start_ms"] = total_ms
        print(
            f"Prewarming complete in {total_ms}ms "
            f"(VAD: {self.warmup_timings.get('vad_ms')}ms, "
            f"STT: {self.warmup_timings.get('stt_ms')}ms, "
            f"LLM: {self.warmup_timings.get('llm_ms')}ms, "
            f"TTS: {self.warmup_timings.get('tts_ms')}ms)"
        )
        return self.warmup_timings

    def shutdown(self):
        """Cleanly terminate bounded executor threads."""
        self.executor.shutdown(wait=False)
