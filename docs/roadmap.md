# Implementation Roadmap & Phase Deliverables

## Overview

This project is built strictly in modular vertical slices. Each phase produces a fully runnable, measurable increment before introducing additional layers of complexity.

---

## 📅 Phased Execution Breakdown

### ✅ Phase 0: Hardware Inspection, Architecture & Decision Blueprint (Current)
- [x] Host hardware profiling (CPU, RAM, GPU, audio devices, environment).
- [x] CPU-optimized model stack selection (Whisper INT8, Qwen 2.5 Q4_K_M, Piper ONNX).
- [x] Git repository initialization and commit structure.
- [x] System architecture specification (`docs/architecture.md`).
- [x] Dependency decision matrix (`docs/hardware_and_stack_analysis.md`).
- [x] Configuration blueprint (`.env.example`) and packaging (`pyproject.toml`).

---

### ✅ Phase 1: Minimal Working Voice Loop (Vertical Slice) [Completed]
**Objective:** Establish a working, turn-based round-trip voice interaction without streaming complexity.
- **Components:**
  1. `app/audio/input.py`: Blocking microphone capture with fixed record duration.
  2. `app/stt/local_stt.py`: Audio buffer transcription via `faster-whisper` (`base.en`, INT8).
  3. `app/llm/main_model.py`: Synchronous / streaming LLM completion via local Ollama API with fallback.
  4. `app/tts/local_tts.py`: Synthesis of text response into audio array via `piper-tts`.
  5. `app/audio/playback.py`: Output playback to speaker via `sounddevice`.
  6. `app/main.py`: Command-line loop connecting all 5 steps with latency metrics.
- **Acceptance Criteria:**
  - [x] User can speak a prompt (e.g. *"What is the capital of France?"*).
  - [x] Terminal prints the transcript, the LLM response, and plays audible speech through speakers.
  - [x] End-to-end turn latency is printed to the terminal.

---

### ✅ Phase 2: Streaming Audio & Turn Management [Completed]
**Objective:** Transition from batch recording to continuous audio stream with Voice Activity Detection (VAD) and barge-in.
- **Components:**
  1. `app/audio/vad.py`: Silero VAD ONNX streaming frame evaluator (32ms frames) with energy fallback.
  2. `app/conversation/turn_manager.py`: State machine (`IDLE`, `USER_SPEAKING`, `USER_PAUSED`, `USER_TURN_FINALIZING`, `AGENT_SPEAKING`, `INTERRUPTED`, `CANCELLED`).
  3. `app/conversation/cancellation.py`: Real-time barge-in detector with `sounddevice` output buffer flush.
  4. `app/conversation/orchestrator.py`: Full asynchronous live stream loop with real-time barge-in.
- **Acceptance Criteria:**
  - [x] Agent automatically begins listening without manual keypresses.
  - [x] Natural pauses (< 500ms) do not cut the user off.
  - [x] Speaking while the agent is talking instantly stops playback within < 100ms.
  - Speaking while the agent is talking instantly stops playback within < 100ms.

---

### ✅ Phase 3: Audio Preprocessing & Model Prewarming [Completed]
**Objective:** Eliminate cold-start latency spikes and isolate heavy inference from the event loop.
- **Components:**
  1. `app/lifecycle.py`: Application startup orchestrator (`PrewarmingManager`) to preload ONNX sessions, warm up Ollama models, and initialize audio streams.
  2. `app/audio/preprocessing.py`: Lightweight DC offset removal, pre-emphasis filter, and spectral noise gate.
  3. Bounded `ThreadPoolExecutor` isolating blocking C++ model inference from the asyncio event loop.
- **Acceptance Criteria:**
  - [x] First conversation turn demonstrates zero cold-start delay compared to subsequent turns.
  - [x] Async event loop stays unblocked with bounded thread execution.

---

### ✅ Phase 4: Observability & Latency Benchmarking [Completed]
**Objective:** Instrument every pipeline boundary with high-precision monotonic timestamps.
- **Components:**
  1. `app/observability/events.py`: Turn lifecycle checkpoints and `TurnMetrics` calculation engine.
  2. `app/observability/metrics.py`: Monotonic timer recorder tracking KPIs with structured JSON and percentile aggregations.
  3. `benchmarks/run_latency_suite.py`: Automated benchmarking script reporting p50, p90, and p95 across trials.
- **Acceptance Criteria:**
  - [x] Every completed turn outputs a rich JSON summary with TTFT, TTFA, STT duration, and total turnaround time.
  - [x] Benchmark reports p50, p90, and p95 across repeated trials.

---

### ✅ Phase 5: Predictive Filler Generation (SLM Track) [Completed]
**Objective:** Implement background SLM acknowledgment generation.
- **Components:**
  1. `app/llm/filler_model.py`: Fast Ollama client for Qwen 2.5 0.5B with rule fallback.
  2. `app/intent/filler_policy.py`: Safe filler validation rules (prevents hallucinations, repetitive fillers, or false promises).
  3. `app/conversation/gate.py`: `PlaybackGate` coordinating filler playback and discard logic.
  4. `app/conversation/orchestrator.py`: Concurrent task dispatcher coordinating filler and main LLM.
- **Acceptance Criteria:**
  - [x] Generates context-aware acknowledgments (e.g., *"Let me check that..."*) in < 120ms.
  - [x] Filler is automatically discarded if the main LLM response is ready before filler playback starts.

---

### ✅ Phase 6: Intent Readiness & Speculative Execution [Completed]
**Objective:** Trigger filler generation before the user stops speaking by analyzing partial STT hypotheses.
- **Components:**
  1. `app/intent/readiness.py`: Real-time partial transcript analyzer checking syntax completeness, question classification, and transcript stability.
  2. `app/conversation/gate.py`: Playback gate ensuring audio is never played over the user with speech collision safety.
  3. `app/conversation/orchestrator.py`: Speculative filler pipeline running concurrently with streaming speech.
- **Acceptance Criteria:**
  - [x] Long queries trigger filler generation 300–600ms before user turn finalization.
  - [x] Zero speech collision (agent never speaks while user VAD is active).

---

### ✅ Phase 7: Optimization & Stress Testing [Completed]
**Objective:** Profile and squeeze performance on the AMD Ryzen 5 CPU.
- **Activities:**
  - [x] Chunk size tuning (16ms vs 32ms vs 64ms) via dynamic frame intervals in `SileroVAD` and `TurnManager`.
  - [x] Whisper beam size and quantization tuning (`beam_size=1` greedy baseline, `int8` CTranslate2).
  - [x] CPU thread affinity and concurrency limits (bounded workers and OpenMP/BLAS thread budgeting).
  - [x] Edge-case stress testing: rapid barge-in, background office noise, compound questions, thread saturation.
- **Acceptance Criteria:**
  - [x] VAD frame processing takes < 1.0ms on CPU with < 2.5% CPU overhead across 16ms/32ms/64ms frames.
  - [x] Rapid back-to-back barge-in interruptions abort cleanly without leaked audio or state desynchronization.
  - [x] Natural mid-sentence pauses (< 500ms) do not cut off compound questions.

---

### ✅ Phase 8: Local Web Dashboard & WebSocket Gateway [Completed]
**Objective:** Provide a sleek local control interface and real-time WebSocket event streaming.
- **Components:**
  1. `app/api/server.py`: FastAPI server with `EventBroadcaster`, WebSocket telemetry streaming (`/ws/events`), and REST configuration endpoints.
  2. `app/static/`: Sleek glassmorphic web dashboard (HTML5, Vanilla CSS, Vanilla JS) featuring real-time sine waveform canvas visualizer, dynamic turn-state badges, live partial transcript stream, and KPI waterfall breakdown.
  3. Integrated telemetry dispatcher in `StreamingVoiceOrchestrator` emitting audio levels, barge-in signals, transcripts, and latency checkpoints.
- **Acceptance Criteria:**
  - [x] Dashboard updates turn states (`IDLE`, `USER_SPEAKING`, `AGENT_SPEAKING`, `INTERRUPTED`) in real time.
  - [x] Dynamic audio waveform reflects live microphone RMS energy.
  - [x] Latency waterfall calculates STT, LLM TTFT, and TTS first audio chunk durations.

---

### ✅ Phase 9: Real-Time Tool Calling & Dynamic Agent Grounding [Completed]
**Objective:** Ground agent responses in real-time system facts, time, calculations, and local notes with sub-10ms tool execution.
- **Components:**
  1. `app/tools/registry.py`: Extensible tool registry and decorator engine.
  2. `app/tools/builtin.py`: Built-in tools for local time/date, system diagnostics (CPU/RAM load), safe arithmetic evaluation, voice notes management, and weather forecast.
  3. `FastToolMatcher`: Sub-millisecond pre-LLM intent classifier directing factual queries to tools without invoking the heavy 3B LLM, dropping turnaround latency to < 150ms.
- **Acceptance Criteria:**
  - [x] Time and date queries return accurate ground truth local time.
  - [x] Mathematical calculations compute safely without arbitrary eval vulnerabilities.
  - [x] Matching tool execution bypasses LLM latency and synthesizes speech immediately.

---

### ✅ Phase 10: Multi-Turn Conversation Memory, Acoustic Echo Suppression & Production Packaging [Completed]
**Objective:** Enable natural multi-turn context retention, software acoustic echo loopback suppression, and unified one-command execution.
- **Components:**
  1. `app/conversation/memory.py`: Sliding-window dialogue memory with rolling context compression preserving sub-300ms LLM TTFT and JSON session persistence.
  2. `app/audio/echo_cancellation.py`: `AcousticEchoSuppressor` filtering microphone speaker loopback during agent speech without hardware AEC DSP while allowing loud human barge-in breakthrough.
  3. Unified CLI launcher: Single command `python app/main.py --dashboard` and cross-platform launcher scripts (`run.ps1`, `run.sh`).
- **Acceptance Criteria:**
  - [x] Multi-turn context is tracked across conversation turns without token ballooning.
  - [x] Speaker output does not cause false self-barge-in interruptions.
  - [x] Cross-platform launcher scripts execute out-of-the-box across Windows and Unix environments.

---

### ✅ Phase 11: Full-Duplex Web Audio Streaming, Browser Voice I/O & Dynamic Voice Personas [Completed]
**Objective:** Support direct in-browser microphone capture, real-time bidirectional audio streaming over WebSocket, instant browser barge-in, and dynamic voice personas.
- **Components:**
  1. `app/api/server.py`: Bidirectional audio streaming gateway at `/ws/voice` accepting raw Float32/Int16 PCM frames from browser microphones and streaming synthesized audio bytes back for immediate client playback.
  2. `app/llm/personas.py`: `PersonaManager` with hot-swappable conversational personas (`assistant`, `technologist`, `tutor`, `companion`) modulating system prompt and sampling temperature.
  3. `app/static/`: Web dashboard call interface featuring one-click browser mic connection (`getUserMedia` + `AudioContext`), real-time browser speaker playback, instant audio cutoff on barge-in interruption, and live persona switcher.
- **Acceptance Criteria:**
  - [x] Browser client streams live microphone audio over WebSocket and plays agent responses through speakers.
  - [x] Speaking while the browser is playing agent audio instantly interrupts and halts client playback (< 60ms).
  - [x] Personas switch dynamically across REST (`/api/personas/active`) and WebSocket commands with immediate effect.


