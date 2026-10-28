# Local Voice Agent - Windows PowerShell Launcher
param(
    [switch]$Dashboard,
    [switch]$Streaming,
    [switch]$Simulate,
    [switch]$Benchmark,
    [int]$Turns = 1,
    [int]$Port = 8000
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Local-First Low-Latency Conversational Voice Agent" -ForegroundColor White
Write-Host "  CPU-Optimized (AMD Ryzen 5 AVX2) - Real-Time Barge-In" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if ($Benchmark) {
    Write-Host "[Mode] Running Automated Latency Benchmark Suite ($Turns trials)..." -ForegroundColor Yellow
    uv run python benchmarks/run_latency_suite.py --turns $Turns
} elseif ($Dashboard) {
    Write-Host "[Mode] Launching Streaming Voice Agent + Web Dashboard on port $Port..." -ForegroundColor Green
    uv run python app/main.py --dashboard --port $Port
} elseif ($Streaming) {
    Write-Host "[Mode] Launching Continuous Streaming Audio Mode..." -ForegroundColor Green
    uv run python app/main.py --streaming
} else {
    $SimFlag = if ($Simulate) { "--simulate" } else { "" }
    Write-Host "[Mode] Running Batch Voice Turn Mode..." -ForegroundColor Green
    uv run python app/main.py --turns $Turns $SimFlag
}
