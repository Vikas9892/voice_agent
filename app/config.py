from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_samples: int = 512

    # VAD & Turn Taking
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 200
    vad_min_silence_duration_ms: int = 500

    # STT
    stt_model_size: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1

    # LLM
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    llm_system_prompt: str = "You are a concise voice assistant. Reply in 1-2 short sentences."
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150

    # TTS
    tts_voice_model: str = "en_US-lessac-medium.onnx"
    tts_sample_rate: int = 22050

    # Barge-In
    barge_in_enabled: bool = True


settings = Settings()
