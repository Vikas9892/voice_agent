import os
import numpy as np


class SileroVAD:
    """Voice Activity Detector using Silero ONNX runtime with energy fallback."""

    def __init__(self, model_path: str = "models/silero_vad.onnx", threshold: float = 0.5):
        self.threshold = threshold
        self.session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None

    def reset(self):
        """Reset internal hidden state between conversation turns."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def get_speech_prob(self, frame: np.ndarray, sample_rate: int = 16000) -> float:
        """Compute speech probability for a 32ms frame (512 samples @ 16kHz)."""
        if len(frame) == 0:
            return 0.0

        if self.session is not None:
            try:
                # Silero expects float32 tensor [1, N]
                x = np.expand_dims(frame.astype(np.float32), axis=0)
                sr = np.array([sample_rate], dtype=np.int64)
                inputs = {
                    "input": x,
                    "state": self._state,
                    "sr": sr,
                }
                out, new_state = self.session.run(None, inputs)
                self._state = new_state
                return float(out[0][0])
            except Exception:
                pass

        # shortcut: RMS energy fallback when ONNX model is missing or fails
        rms = float(np.sqrt(np.mean(frame**2)))
        # Map typical conversational RMS (0.005 to 0.08) to 0.0 - 1.0 probability
        return min(max((rms - 0.01) / 0.04, 0.0), 1.0)

    def is_speech(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        """Binary speech decision based on probability threshold."""
        return self.get_speech_prob(frame, sample_rate) >= self.threshold
