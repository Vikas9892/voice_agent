import numpy as np
from app.audio.preprocessing import (
    apply_noise_gate,
    apply_preemphasis,
    preprocess_audio,
    remove_dc_offset,
)


def test_remove_dc_offset():
    # Audio with constant +0.5 DC bias
    raw = np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float32)
    cleaned = remove_dc_offset(raw)
    assert np.isclose(np.mean(cleaned), 0.0, atol=1e-6)


def test_apply_preemphasis():
    raw = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    filtered = apply_preemphasis(raw, coeff=0.97)
    assert len(filtered) == len(raw)
    assert filtered[0] == raw[0]
    assert np.isclose(filtered[1], 0.2 - 0.97 * 0.1)


def test_apply_noise_gate():
    # Quiet background noise (RMS << 0.008)
    quiet_noise = np.array([0.001, -0.001, 0.001, -0.001], dtype=np.float32)
    gated = apply_noise_gate(quiet_noise, threshold_rms=0.008, attenuation=0.1)
    assert np.max(np.abs(gated)) < np.max(np.abs(quiet_noise))

    # Real human speech (RMS >> 0.008)
    speech = np.array([0.2, -0.25, 0.3, -0.15], dtype=np.float32)
    preserved = apply_noise_gate(speech, threshold_rms=0.008, attenuation=0.1)
    assert np.array_equal(preserved, speech)


def test_preprocess_audio_end_to_end():
    audio = np.array([0.501, 0.499, 0.502, 0.498], dtype=np.float32)
    processed = preprocess_audio(audio)
    assert len(processed) == len(audio)
    assert np.isclose(np.mean(processed), 0.0, atol=1e-5)
