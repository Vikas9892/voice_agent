from app.intent.readiness import IntentReadinessDetector


def test_intent_readiness_too_short():
    detector = IntentReadinessDetector(min_words=4)
    res = detector.evaluate("what is")
    assert not res.is_ready
    assert res.reason == "too_few_words"


def test_intent_readiness_dangling_tail():
    detector = IntentReadinessDetector(min_words=4)
    res = detector.evaluate("can you tell me about the")
    assert not res.is_ready
    assert res.reason == "dangling_clause"


def test_intent_readiness_question_ready():
    detector = IntentReadinessDetector(min_words=4, threshold=0.65)
    res = detector.evaluate("what is the capital of France")
    assert res.is_ready
    assert res.is_question
    assert res.score >= 0.65
    assert res.reason == "intent_ready"


def test_intent_readiness_stability_progression():
    detector = IntentReadinessDetector(min_words=4, threshold=0.65)
    # First chunk: dangling clause -> not ready
    r1 = detector.evaluate("how do you optimize with")
    assert not r1.is_ready

    # Second extended chunk: complete clause -> ready and stability boosted
    r2 = detector.evaluate("how do you optimize with fast whisper")
    assert r2.is_ready
    assert detector.stability_count >= 2


def test_intent_readiness_reset():
    detector = IntentReadinessDetector()
    detector.evaluate("what is the time now")
    assert detector.stability_count >= 1
    assert detector.last_transcript != ""

    detector.reset()
    assert detector.stability_count == 0
    assert detector.last_transcript == ""
