"""Analyse short customer feedback: clean tokens, entities, topics and sentiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Topic keywords, matched against lemmas. A comment can belong to several topics.
TOPICS: dict[str, set[str]] = {
    "Payments": {"payout", "withdraw", "withdrawal", "payment", "money", "deposit", "€"},
    "Support": {"support", "chat", "contact", "explain", "ban", "banned"},
    "Stability": {"crash", "freeze", "froze", "bug", "connection", "log", "loading"},
    "Bonuses": {"bonus", "promotion", "reward"},
    "Product & UX": {"interface", "design", "layout", "dashboard", "history", "room", "odds"},
}

POSITIVE_THRESHOLD = 0.1  # scores above this are positive, below -this negative


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def vader():
    return SentimentIntensityAnalyzer()


def to_label(score: float, threshold: float = POSITIVE_THRESHOLD) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


@dataclass
class Analysis:
    tokens: list[str]
    entities: list[tuple[str, str]]
    topics: list[str]
    textblob: float
    vader: float
    labels: dict[str, str] = field(default_factory=dict)


def analyze_text(text: str) -> Analysis:
    doc = nlp()(text)
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    lemmas = {t.lemma_.lower() for t in doc} | {t.text.lower() for t in doc}
    topics = [name for name, words in TOPICS.items() if lemmas & words]
    tb = round(TextBlob(text).sentiment.polarity, 3)
    vd = round(vader().polarity_scores(text)["compound"], 3)
    return Analysis(
        tokens=tokens,
        entities=[(e.text, e.label_) for e in doc.ents],
        topics=topics,
        textblob=tb,
        vader=vd,
        labels={"textblob": to_label(tb), "vader": to_label(vd)},
    )


def analyze_frame(df: pd.DataFrame, text_col: str = "feedback") -> pd.DataFrame:
    """Return one row per comment with tokens, entities, topics and both sentiment scores."""
    rows = []
    for text in df[text_col].astype(str):
        a = analyze_text(text)
        rows.append(
            {
                "cleaned_tokens": " ".join(a.tokens),
                "named_entities": "; ".join(f"{t} ({lab})" for t, lab in a.entities),
                "topics": "; ".join(a.topics) or "Other",
                "textblob_score": a.textblob,
                "textblob_label": a.labels["textblob"],
                "vader_score": a.vader,
                "vader_label": a.labels["vader"],
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def accuracy_report(results: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Compare each sentiment method with hand labels."""
    out = []
    for method in ("textblob", "vader"):
        pred = results[f"{method}_label"]
        truth = results[label_col]
        neg = truth == "negative"
        out.append(
            {
                "method": method,
                "accuracy": round(float((pred == truth).mean()), 2),
                "negatives_found": f"{int((pred[neg] == 'negative').sum())} of {int(neg.sum())}",
                "called_neutral": int((pred == "neutral").sum()),
            }
        )
    return pd.DataFrame(out)
