from app.intent.filler_policy import FillerPolicy


class PlaybackGate:
    """Coordinates playback eligibility between speculative fillers and main answers."""

    def __init__(self, policy: FillerPolicy):
        self.policy = policy
        self.filler_played: bool = False
        self.main_ready: bool = False

    def should_play_filler(self, filler_text: str | None) -> bool:
        """
        Evaluate if filler should be dispatched to speaker.
        Returns False if main answer is already prepared or filler is empty.
        """
        if self.main_ready or not filler_text:
            self.policy.mark_discarded()
            return False

        self.filler_played = True
        self.policy.mark_played(filler_text)
        return True

    def mark_main_ready(self):
        """Signal that main response audio is ready for playback."""
        self.main_ready = True
