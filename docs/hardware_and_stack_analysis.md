# Hardware Profile & Dependency Decision Analysis

## 1. System Hardware Audit

An automated hardware inspection performed on the host machine revealed the following technical specifications:

| Resource | Detected Specification | Architectural Implications |
|---|---|---|
| **Operating System** | Windows 11 (64-bit) | Native Win32 audio subsystems, PowerShell execution environment. |
| **CPU** | AMD Ryzen 5 7520U (4 Cores, 8 Threads) | Zen 2 architecture with AVX2 instructions. Thread budget must be strictly bounded across concurrent models. |
| **System Memory** | 16 GB DDR5 (~15.3 GB available) | Sufficient memory to simultaneously host STT (300MB), Main LLM (2.2GB), SLM (400MB), and TTS (150MB) within < 4GB total RAM footprint. |
| **Graphics / Accelerator** | AMD Radeon 610M (Integrated RDNA2, 2 CUs) | **No Nvidia CUDA support.** CUDA-dependent accelerated runtimes will not work. Must target CPU-optimized INT8/GGUF runtimes with AVX2 acceleration and ONNX Runtime CPU. |
| **Audio Hardware** | Realtek High Definition Audio | Direct stereo input/output supported via Windows DirectSound/WASAPI through PortAudio. |
| **Python Environment** | Python 3.12 (managed via `uv`) | Modern standard with broad binary wheel availability for NumPy 1.x, ONNX Runtime, and SoundDevice. |

---

## 2. Strategic Constraints for CPU-Only Execution

Because dedicated CUDA acceleration is unavailable on this machine, our architecture must adhere to three non-negotiable optimization rules:

1. **Lightweight Models over Massive Architectures:**
   - Main LLM capped at **3 Billion parameters** (4-bit quantized, Q4_K_M).
   - Filler SLM capped at **0.5 Billion parameters** (4-bit quantized).
   - Speech-to-Text capped at **Whisper `tiny.en` or `base.en`** (INT8 quantized via CTranslate2).
2. **ONNX Runtime & CTranslate2 Execution:**
   - Avoid loading full PyTorch graph runtimes for VAD and TTS.
   - Silero VAD runs under `onnxruntime` consuming < 1.5% CPU per 32ms frame.
   - Piper TTS runs under native ONNX C++ engine yielding a Real-Time Factor (RTF) of < 0.15 on CPU.
3. **Async Event Loop Thread Safety:**
   - Heavy blocking C++ model inference (faster-whisper, Piper) must execute on dedicated `ThreadPoolExecutor` workers or separate process pipes to prevent freezing the `asyncio` event loop.

---

## 3. Technology Evaluation & Decision Matrix

### 3.1. Audio I/O Subsystem
- **Options Evaluated:** `sounddevice` (PortAudio wrapper) vs `pyaudio` vs `PyAV`.
- **Selected:** **`sounddevice`**
- **Rationale:** Zero-copy NumPy buffer integration, cross-platform stability on Windows 11 WASAPI, non-blocking callback streams, and well-maintained wheels via `uv`.

### 3.2. Voice Activity Detection (VAD)
- **Options Evaluated:** `Silero VAD v5` vs `WebRTC VAD` vs `pyannote.audio`.
- **Selected:** **`Silero VAD v5 (ONNX)`**
- **Trade-off Analysis:**
  - *WebRTC VAD:* Extremely fast (< 0.1ms), but purely energy-based; triggers excessively on keyboard clatter, breathing, and background ambient office noise.
  - *pyannote.audio:* SOTA accuracy, but requires heavy PyTorch runtime and high CPU overhead, making real-time streaming sluggish.
  - *Silero VAD:* Deep learning accuracy (distinguishes real human phonemes from background clatter) while executing in ~1.2ms per 32ms frame on CPU via ONNX Runtime.

### 3.3. Noise Suppression
- **Options Evaluated:** `DTLN (Dual-Signal Transformation LSTM Network)` vs `noisereduce` (Spectral Gating) vs `Bypass (Raw Audio)`.
- **Selected:** **`Bypass with optional Spectral Gate hook`**
- **Trade-off Analysis:**
  - Running continuous recurrent neural noise suppression (like DTLN) consumes 1 to 1.5 dedicated CPU cores on a 4-core machine.
  - Silero VAD and Whisper `base.en` have inherent robustness to stationary background noise.
  - *Decision:* Keep noise suppression modular and disabled by default for Phase 0–1; evaluate lightweight spectral gating in Phase 3 if testing indicates dirty acoustic environments.

### 3.4. Speech-to-Text (STT)
- **Options Evaluated:** `faster-whisper` (CTranslate2) vs `whisper.cpp` vs `Vosk / Kaldi` vs `Sherpa-ONNX`.
- **Selected:** **`faster-whisper (CTranslate2 with INT8 quantization)`**
- **Trade-off Analysis:**
  - *whisper.cpp:* Excellent C++ implementation, but Python bindings require manual compilation or CMake tooling on Windows.
  - *faster-whisper:* Backed by CTranslate2, achieves up to 4x speedup over vanilla OpenAI Whisper on CPU, native INT8 quantization, built-in VAD chunk filtering, and easy pip installation on Windows.
  - *Model Selection:* `base.en` (~140MB) or `tiny.en` (~75MB). `base.en` provides the ideal sweet spot between word error rate (WER) and transcription latency (~150–250ms per audio chunk).

### 3.5. Conversational Intelligence (Main LLM)
- **Options Evaluated:** `Ollama (Qwen 2.5 3B / Llama 3.2 3B)` vs `llama.cpp direct server` vs `Transformers CPU`.
- **Selected:** **`Ollama hosting Qwen 2.5 3B Instruct (Q4_K_M)`**
- **Trade-off Analysis:**
  - *Qwen 2.5 3B:* Superior reasoning, instruction following, and multilingual support compared to earlier generation 3B models. Generates ~22–30 tokens/second on AMD Ryzen 5 CPU with AVX2.
  - *Time to First Token (TTFT):* ~250–380ms on CPU for short conversational context.
  - *Ollama Runtime:* Provides an isolated background service with native OpenAI-compatible HTTP/SSE streaming, automated model management, and low memory overhead.

### 3.6. Predictive Filler Engine (SLM)
- **Options Evaluated:** `Qwen 2.5 0.5B Instruct` vs `Llama 3.2 1B` vs `Static Rule-Based Table`.
- **Selected:** **`Qwen 2.5 0.5B Instruct (Q4_K_M) via Ollama` (with rule-based fallback)**
- **Trade-off Analysis:**
  - 0.5B parameters footprint is only ~380 MB in RAM.
  - Generates 5–10 filler tokens in under 120ms on CPU.
  - Enables dynamic, contextual acknowledgments based on partial input (e.g. recognizing whether the user is asking for code vs a comparison vs a date).

### 3.7. Text-to-Speech (TTS)
- **Options Evaluated:** `Piper TTS (ONNX)` vs `Kokoro-82M ONNX` vs `Coqui TTS` vs `Bark`.
- **Selected:** **`Piper TTS (ONNX / en_US-lessac-medium)`**
- **Trade-off Analysis:**
  - *Coqui / Bark:* Unacceptable CPU latency (RTF > 1.2 on CPU, cannot stream in real-time).
  - *Kokoro-82M:* Promising voice quality, but ONNX streaming pipelines on Windows are nascent.
  - *Piper TTS:* Industry standard for embedded/local speech synthesis. Runs locally via ONNX Runtime, generates high quality 22.05kHz audio with an RTF of ~0.12 (i.e., generates 1 second of speech in 120ms on CPU). Supports sentence and clause-level streaming.

---

## 4. Master Dependency Matrix

| Stage | Selected Engine | Model / Artifact | License | Quantization | RAM Budget | Expected CPU Latency |
|---|---|---|---|---|---|---|
| **Audio I/O** | `sounddevice` | PortAudio WASAPI | MIT | Raw PCM | ~20 MB | < 5 ms |
| **VAD** | `Silero VAD v5` | `silero_vad.onnx` | MIT | FP32 / ONNX | ~30 MB | 1–2 ms / 32ms frame |
| **STT** | `faster-whisper` | `base.en` | MIT / Apache 2.0 | INT8 | ~250 MB | 150–250 ms / chunk |
| **Main LLM** | `Ollama` | `qwen2.5:3b-instruct` | Apache 2.0 | Q4_K_M | ~2.1 GB | TTFT: 280–380 ms |
| **Filler SLM** | `Ollama` | `qwen2.5:0.5b-instruct`| Apache 2.0 | Q4_K_M | ~390 MB | TTFT: 90–140 ms |
| **TTS** | `Piper TTS` | `en_US-lessac-medium` | MIT | ONNX | ~120 MB | First chunk: 100–160 ms |
| **Total** | — | — | Fully Open Source | — | **< 3.0 GB** | **Perceived Latency: ~650–850 ms** |
