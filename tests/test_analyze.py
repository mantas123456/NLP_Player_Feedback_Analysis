import pandas as pd

from feedback_nlp import accuracy_report, analyze_frame, analyze_text
from feedback_nlp.analyze import to_label


def test_to_label_thresholds():
    assert to_label(0.5) == "positive"
    assert to_label(-0.5) == "negative"
    assert to_label(0.05) == "neutral"


def test_topics_and_entities():
    a = analyze_text("I won €500 but it took 4 days to withdraw it.")
    assert "Payments" in a.topics
    assert any(label in {"MONEY", "DATE"} for _, label in a.entities)


def test_comment_can_have_several_topics():
    a = analyze_text("Live chat was helpful but payouts were delayed by 3 days.")
    assert {"Payments", "Support"} <= set(a.topics)


def test_tokens_are_lemmatised_without_stopwords():
    a = analyze_text("The games were crashing")
    assert "the" not in a.tokens
    assert "game" in a.tokens


def test_frame_and_report():
    df = pd.DataFrame(
        {
            "feedback": ["I love the new design", "Support was terrible and unhelpful"],
            "label": ["positive", "negative"],
        }
    )
    out = analyze_frame(df)
    assert {"topics", "textblob_label", "vader_label"} <= set(out.columns)
    report = accuracy_report(out)
    assert set(report["method"]) == {"textblob", "vader"}
    assert report["accuracy"].between(0, 1).all()
