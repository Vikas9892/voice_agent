from app.intent.filler_policy import FillerPolicy
from app.llm.filler_model import FillerModel


def test_filler_policy_eligibility():
    policy = FillerPolicy()

    # Greetings should NOT be eligible
    assert not policy.is_eligible("hi")
    assert not policy.is_eligible("hello")
    assert not policy.is_eligible("hey how are you")
    assert not policy.is_eligible("okay")

    # Short query (< 3 words)
    assert not policy.is_eligible("what time")

    # Complex questions SHOULD be eligible
    assert policy.is_eligible("can you explain quantum computing to me")
    assert policy.is_eligible("compare faster whisper and whisper cpp on cpu")


def test_filler_policy_anti_repetition():
    policy = FillerPolicy(history_len=2)

    # First selection
    f1 = policy.select_filler("query one")
    policy.mark_played(f1)
    assert f1 in policy.recent_fillers

    # Second selection shouldn't match f1
    f2 = policy.select_filler("query two")
    policy.mark_played(f2)
    assert f2 in policy.recent_fillers

    # Third selection
    f3 = policy.select_filler("query three")
    assert f3 != f2

    assert policy.stats()["played"] == 2
    policy.mark_discarded()
    assert policy.stats()["discarded"] == 1


def test_filler_model_fallback():
    model = FillerModel(base_url="http://invalid-host:9999")
    # When offline or unreachable, generate_filler returns None gracefully
    res = model.generate_filler("Can you explain latency?")
    assert res is None
