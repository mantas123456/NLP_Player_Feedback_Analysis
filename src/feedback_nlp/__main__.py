"""Command line: python -m feedback_nlp data/feedback.csv --out outputs"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analyze import accuracy_report, analyze_frame  # noqa: E402


def topic_sentiment_chart(results: pd.DataFrame, path: Path) -> None:
    """Share of negative comments per topic, using the hand label when present."""
    col = "label" if "label" in results else "vader_label"
    rows = results.assign(topic=results["topics"].str.split("; ")).explode("topic")
    table = (
        rows.groupby("topic")[col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    )
    table = table.loc[table.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    left = pd.Series(0, index=table.index)
    colors = {"negative": "#c0392b", "neutral": "#95a5a6", "positive": "#27ae60"}
    for label in table.columns:
        ax.barh(table.index, table[label], left=left, color=colors[label], label=label)
        left = left + table[label]
    ax.set_xlabel("Number of comments")
    ax.set_title("Feedback by topic and sentiment", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse a CSV of customer feedback")
    ap.add_argument("csv", help="CSV with a 'feedback' column (and optional 'label')")
    ap.add_argument("--out", default="outputs", help="output folder")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = analyze_frame(pd.read_csv(args.csv))
    results.to_csv(out / "processed_feedback.csv", index=False)
    topic_sentiment_chart(results, out / "topics_by_sentiment.png")
    print(results[["feedback", "topics", "textblob_label", "vader_label"]].to_string(index=False))

    if "label" in results:
        report = accuracy_report(results)
        report.to_csv(out / "sentiment_accuracy.csv", index=False)
        print("\nAgainst hand labels:\n" + report.to_string(index=False))


if __name__ == "__main__":
    main()
