import pandas as pd
import pytest

from preprocessing import clean_text, load_and_clean
import predict as legacy


def test_clean_text_basic():
    assert clean_text("Subject:  Hello   WORLD \n") == "hello world"


@pytest.mark.parametrize("bad", [None, 5, [], {}])
def test_clean_text_non_string(bad):
    assert clean_text(bad) == ""


def test_load_and_clean_removes_empty_and_dups(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"text": ["Subject: a", "Subject: a", "   ", "Subject: b"], "spam": [0, 0, 1, 1]}).to_csv(p, index=False)
    df = load_and_clean(str(p))
    assert list(df["text"]) == ["a", "b"]


def test_legacy_predict_still_works():
    """The original predict.py API must stay compatible with the shipped pipeline."""
    model = legacy.load_model()
    out = legacy.predict_email(model, "Subject: free prize winner click here now")
    assert set(out) == {"label", "label_name", "spam_probability", "ham_probability"}
    assert out["spam_probability"] + out["ham_probability"] == pytest.approx(1.0)


def test_engine_probability_and_label(engine, ham):
    r = engine.analyze_email(ham)
    p = r["ml"]["spam_probability"]
    assert 0 <= p <= 1 and p < 0.5 and r["ml"]["label"].startswith("Ham")


def test_engine_flags_obvious_spam(engine):
    r = engine.analyze_email("Subject: you won a free prize!!!\n\nClaim your free prize, cheap pills, discount offer, click here now")
    assert r["ml"]["spam_probability"] > 0.5


def test_batch_matches_single(engine, ham, phish):
    batch = engine.analyze_batch([{"text": ham}, {"text": phish}])
    for res, raw in zip(batch, (ham, phish)):
        single = engine.analyze_email(raw)
        assert res["risk"]["score"] == single["risk"]["score"]
        assert res["ml"]["spam_probability"] == pytest.approx(single["ml"]["spam_probability"])


def test_empty_vocabulary_message_does_not_crash(engine):
    r = engine.analyze_email("zzzzqqqq xxxyyy")
    assert 0 <= r["risk"]["score"] <= 100
    assert r["ml"]["explanation"]["note"]


def test_score_always_in_range(engine, phish):
    for raw in (phish, "x", "Subject: a\n\n" + "http://1.2.3.4/a.exe " * 50):
        s = engine.analyze_email(raw)["risk"]["score"]
        assert 0 <= s <= 100


def test_phishing_scores_higher_than_ham(engine, phish, ham):
    assert engine.analyze_email(phish)["risk"]["score"] > engine.analyze_email(ham)["risk"]["score"] + 40
