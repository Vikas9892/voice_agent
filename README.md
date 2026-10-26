# Local-First, Low-Latency Voice Agent

An open-source, production-grade conversational voice agent running entirely on your local machine with zero paid cloud dependencies.

## 🎙️ Overview

This project implements an end-to-end, real-time conversational voice pipeline designed for low latency, natural turn-taking, barge-in interruptibility, and predictive filler generation.

```
+----------------------------------------------------------------------------------------------------+
|                                    AUDIO & TURN-TAKING PIPELINE                                    |
|                                                                                                    |
|  [ Microphone ] ---> [ Audio Ingest (16kHz) ] ---> [ Silero VAD ] ---> [ Turn Manager ]            |
|                                                          |                      |                  |
|                                            (Speech Detected)          (User Turn End)              |
|                                                          v                      v                  |
|                                                 [ Streaming STT ]       [ Main LLM Stream ]        |
|                                                 (faster-whisper)         (Ollama Qwen2.5)          |
|                                                          |                      |                  |
|                                                  (Partial Words)        (Tokens Yielded)           |
|                                                          v                      v                  |
|                                               [ Intent Readiness ]      [ Streaming TTS ]          |
|                                                          |                (Piper ONNX)             |
|                                                 (Intent Predicted)              |                  |
|                                                          v                      v                  |
|                                                 [ SLM Filler Gen ] ---> [ Audio Playback Gate ]    |
|                                                                                 |                  |
|                                                                           [ Speaker ]              |
+----------------------------------------------------------------------------------------------------+
```

## ⚡ Core Engineering Pillars

1. **Local-First & Private:** Fully self-hosted stack. No proprietary cloud APIs, no recurring per-minute charges, and no external audio telemetry.
2. **Sub-Second Perceived Latency:** Streaming STT and streaming TTS coupled with asynchronous pipeline orchestration.
3. **Intent-Triggered Fillers:** Concurrently produces context-aware acknowledgements ("*Let me check that for you...*") while the user finishes complex queries, hiding generation latency.
4. **Barge-In Ready:** Instantly halts TTS audio generation and playback the moment the user starts speaking.
5. **Rigorous Instrumentation:** Captures monotonic stage timestamps from microphone input to speaker playback.

---

## 💻 Hardware Profile & Target Environment

The system is configured and tested for consumer-grade CPU architectures:

| Component | Target Spec (Local Machine) | Design Decision |
|---|---|---|
| **CPU** | AMD Ryzen 5 7520U (4C/8T) | Multi-threaded CPU inference with INT8 quantization |
| **RAM** | 16 GB DDR5 | Memory footprint capped under 6 GB total for all models |
| **GPU / Acceleration** | AMD Radeon 610M (Integrated) | Pure CPU / AVX2 / ONNX-runtime acceleration |
| **Python** | 3.12 (managed via `uv`) | Pre-built binary wheels for PyTorch, SoundDevice, ONNX |
| **Audio I/O** | 16 kHz Mono Input / 22.05 kHz Output | Optimized for speech recognition and low-buffer streaming |

---

## 🛠️ Project Phases

- [x] **Phase 0:** Hardware Inspection, Architecture & Decision Blueprint
- [x] **Phase 1:** Minimal Working Voice Loop (Microphone → STT → LLM → TTS → Speaker)
- [x] **Phase 2:** Streaming STT & Robust Turn Management (VAD, Endpointing, Barge-In)
- [x] **Phase 3:** Audio Preprocessing & Model Prewarming Lifecycle
- [x] **Phase 4:** Observability, Monotonic Timestamps & Latency Profiling
- [x] **Phase 5:** Contextual SLM Filler Engine
- [x] **Phase 6:** Intent Readiness Detection & Playback Gating
- [ ] **Phase 7:** End-to-End Benchmarking & Optimization
- [ ] **Phase 8:** Packaging, Local UI & Developer Documentation

---

## 📊 Running the Latency Benchmark Suite
```bash
# Run 5 automated trials measuring p50, p90, and p95 across all stages
uv run python benchmarks/run_latency_suite.py --turns 5
```

## 🏃 Running the Voice Agent

### 1. Batch Turn Mode (Phase 1)
```bash
# Run one turn with live microphone input
uv run python app/main.py

# Run with simulated audio (testing without mic)
uv run python app/main.py --simulate
```

### 2. Live Streaming Mode with Real-Time Barge-In (Phase 2)
```bash
# Run continuous streaming VAD and turn-taking loop with interruptible playback
uv run python app/main.py --streaming
```

## 🚀 Quickstart (Phase 0 Setup)

### Prerequisites

- **Python 3.12** (recommended) or 3.11.
- **[uv](https://github.com/astral-sh/uv)** installed on your system.
- **[Ollama](https://ollama.ai/)** running locally (`ollama run qwen2.5:3b-instruct-q4_K_M`).

### 1. Clone & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Vikas9892/voice_agent.git
cd voice_agent

# Create virtual environment with uv using Python 3.12
uv venv --python 3.12

# Activate virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate
```

### 2. Configure Environment

```bash
cp .env.example .env
```

---

## 📖 Detailed Documentation

See the [`docs/`](./docs) directory for architectural specifications and deep dives:
- [`docs/architecture.md`](./docs/architecture.md) — System topology, event model, and state machine.
- [`docs/hardware_and_stack_analysis.md`](./docs/hardware_and_stack_analysis.md) — Benchmark analysis and dependency decision matrix.
- [`docs/roadmap.md`](./docs/roadmap.md) — Phased rollout plan and evaluation criteria.

---

## 📄 License

MIT License. Designed for open research, learning, and self-hosted deployments.
