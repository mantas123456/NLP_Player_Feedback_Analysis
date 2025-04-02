# main.py
"""
NLP Pipeline with Sentiment and Visualizations
"""

import pandas as pd
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
import os

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Paths
input_path = "data/feedback.csv"
output_path = "outputs/processed_feedback.csv"
plot_path = "outputs/"

# Read data
df = pd.read_csv(input_path)

# Store processed results
processed_data = []
all_tokens = []
all_ents = []

for _, row in df.iterrows():
    text = row['feedback']
    doc = nlp(text)

    # Clean tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    all_tokens.extend(tokens)

    # Entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    all_ents.extend([ent.label_ for ent in doc.ents])

    # Sentiment
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    sentiment = (
        "positive" if polarity > 0.1 else
        "negative" if polarity < -0.1 else
        "neutral"
    )

    processed_data.append({
        "id": row['id'],
        "original_feedback": text,
        "cleaned_tokens": " ".join(tokens),
        "named_entities": "; ".join([f"{e[0]} ({e[1]})" for e in entities]),
        "sentiment_score": polarity,
        "sentiment_label": sentiment
    })

# Save processed CSV
output_df = pd.DataFrame(processed_data)
os.makedirs(plot_path, exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"✅ Processed feedback saved to {output_path}")

# ======================
# 📊 Sentiment Bar Chart
# ======================
sentiment_counts = output_df['sentiment_label'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{plot_path}/sentiment_distribution.png")
plt.clf()

# ==============================
# 🔤 Most Common Tokens (Top 10)
# ==============================
token_counts = Counter(all_tokens)
common_tokens = token_counts.most_common(10)

tokens, counts = zip(*common_tokens)
plt.bar(tokens, counts)
plt.title("Top 10 Most Common Tokens")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{plot_path}/common_tokens.png")
plt.clf()

# ====================================
# 🏷️ Most Common Named Entity Labels
# ====================================
if all_ents:
    entity_counts = Counter(all_ents)
    labels, counts = zip(*entity_counts.items())

    plt.bar(labels, counts, color="purple")
    plt.title("Named Entity Label Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/named_entity_labels.png")
    plt.clf()

print("📊 Visualizations saved in 'outputs/' folder.")
