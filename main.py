# main.py
"""
Expanded NLP script:
- Reads feedback from CSV
- Applies NLP preprocessing (tokenization, lemmatization, stopwords, NER)
- Analyzes sentiment using TextBlob
- Outputs processed data to CSV
"""

import pandas as pd
import spacy
from textblob import TextBlob
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Paths
input_path = "data/feedback.csv"
output_path = "outputs/processed_feedback.csv"

# Read the CSV
df = pd.read_csv(input_path)

# Prepare output columns
processed_data = []

for idx, row in df.iterrows():
    text = row['feedback']
    doc = nlp(text)

    # Tokenization, Lemmatization, Remove stopwords/punctuation
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # Named Entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Sentiment analysis
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    sentiment = (
        "positive" if polarity > 0.1
        else "negative" if polarity < -0.1
        else "neutral"
    )

    processed_data.append({
        "id": row['id'],
        "original_feedback": text,
        "cleaned_tokens": " ".join(tokens),
        "named_entities": "; ".join([f"{e[0]} ({e[1]})" for e in entities]),
        "sentiment_score": polarity,
        "sentiment_label": sentiment
    })

# Convert to DataFrame and save
output_df = pd.DataFrame(processed_data)
os.makedirs("outputs", exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"✅ Processed feedback with sentiment saved to {output_path}")
