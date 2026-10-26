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

### ⏳ Phase 7: Optimization & Stress Testing
**Objective:** Profile and squeeze performance on the AMD Ryzen 5 CPU.
- **Activities:**
  - Chunk size tuning (16ms vs 32ms vs 64ms).
  - Whisper beam size and quantization tuning.
  - Thread affinity and concurrency limits.
  - Edge-case testing: rapid barge-in, background office noise, compound questions.

---

### ⏳ Phase 8: Local Web Dashboard & Packaging
**Objective:** Provide a sleek local control interface and reproducible packaging.
- **Components:**
  1. `app/api/server.py`: FastAPI server with WebSocket event streaming.
  2. Modern web dashboard: Live waveform, live partial transcripts, turn state visualizer, waterfall latency breakdown, and parameter toggles.
  3. Complete documentation and one-command launch script.
