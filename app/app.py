"""Streamlit app: analyse one comment or a whole CSV.

Run from the repository root:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedback_nlp import accuracy_report, analyze_frame, analyze_text  # noqa: E402

st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")
st.title("Customer feedback analyzer")
st.caption("Topics, named entities and two lexicon sentiment scores (TextBlob and VADER).")

single, batch = st.tabs(["One comment", "CSV file"])

with single:
    text = st.text_area("Feedback", "Live chat was helpful but payouts were delayed by 3 days.")
    if st.button("Analyze", type="primary") and text.strip():
        a = analyze_text(text)
        c1, c2, c3 = st.columns(3)
        c1.metric("TextBlob", a.labels["textblob"], f"{a.textblob:+.2f}", delta_color="off")
        c2.metric("VADER", a.labels["vader"], f"{a.vader:+.2f}", delta_color="off")
        c3.metric("Topics", ", ".join(a.topics) or "Other")
        st.write("**Entities:**", ", ".join(f"{t} ({lab})" for t, lab in a.entities) or "none")
        st.write("**Cleaned tokens:**", " ".join(a.tokens))

with batch:
    st.write(
        "Upload a CSV with a `feedback` column. Add a `label` column "
        "(positive, negative or neutral) to score the sentiment methods."
    )
    file = st.file_uploader("CSV file", type="csv")
    df = pd.read_csv(file) if file else pd.read_csv(Path(__file__).parents[1] / "data/feedback.csv")
    if "feedback" not in df:
        st.error("The file needs a column named 'feedback'.")
    else:
        results = analyze_frame(df)
        st.dataframe(results, hide_index=True, width="stretch")
        topics = results.assign(topic=results["topics"].str.split("; ")).explode("topic")
        st.bar_chart(topics.groupby("topic")["vader_label"].value_counts().unstack(fill_value=0))
        if "label" in results:
            st.subheader("Sentiment methods against the labels")
            st.dataframe(accuracy_report(results), hide_index=True)
