import json
import os
from unittest.mock import MagicMock, patch
import numpy as np

from benchmarks.run_latency_suite import run_benchmark


def test_run_benchmark_mocked(tmp_path):
    output_path = tmp_path / "test_results.json"

    with patch("benchmarks.run_latency_suite.LocalSTT") as mock_stt_cls, \
         patch("benchmarks.run_latency_suite.LocalLLM") as mock_llm_cls, \
         patch("benchmarks.run_latency_suite.LocalTTS") as mock_tts_cls:

        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "hello"
        mock_stt_cls.return_value = mock_stt

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "hi there"
        mock_llm_cls.return_value = mock_llm

        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = np.zeros(100, dtype=np.float32)
        mock_tts_cls.return_value = mock_tts

        result = run_benchmark(num_turns=3, output_file=str(output_path))

        assert result["trials"] == 3
        assert "percentiles" in result
        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
            assert saved["trials"] == 3
            assert "stt_ms" in saved["percentiles"]
