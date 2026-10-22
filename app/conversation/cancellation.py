import asyncio
from app.audio.playback import stop_playback


class CancellationCoordinator:
    """Coordinates cancellation tokens and audio buffer flushing for real-time barge-in."""

    def __init__(self):
        self.current_turn_id: int = 0
        self.cancel_event: asyncio.Event = asyncio.Event()

    def start_new_turn(self) -> int:
        """Start a new turn, generating an incremented turn ID and clearing cancellation."""
        self.current_turn_id += 1
        self.cancel_event.clear()
        return self.current_turn_id

    def is_cancelled(self) -> bool:
        """Check if current active turn has been interrupted."""
        return self.cancel_event.is_set()

    def cancel_current_turn(self) -> None:
        """Immediately cancel current in-flight generation and flush soundcard playback buffer."""
        self.cancel_event.set()
        stop_playback()
