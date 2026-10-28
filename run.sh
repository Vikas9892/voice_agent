#!/usr/bin/env bash
# Local Voice Agent - Unix / Bash Launcher
set -e

echo "=========================================================="
echo "  Local-First Low-Latency Conversational Voice Agent"
echo "  CPU-Optimized (AVX2) - Real-Time Barge-In & Telemetry"
echo "=========================================================="

MODE="${1:---dashboard}"

case "$MODE" in
  --dashboard)
    echo "[Mode] Launching Streaming Voice Agent + Web Dashboard (http://localhost:8000)..."
    uv run python app/main.py --dashboard
    ;;
  --streaming)
    echo "[Mode] Launching Continuous Streaming Audio Mode..."
    uv run python app/main.py --streaming
    ;;
  --benchmark)
    echo "[Mode] Running Automated Latency Benchmark Suite..."
    uv run python benchmarks/run_latency_suite.py --turns 5
    ;;
  --simulate)
    echo "[Mode] Running Simulated Audio Turn..."
    uv run python app/main.py --simulate
    ;;
  *)
    echo "Usage: ./run.sh [--dashboard | --streaming | --benchmark | --simulate]"
    exit 1
    ;;
esac
