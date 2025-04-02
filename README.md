# 🧠 NLP Player Feedback Analysis (iGaming)

This project processes player feedback from an iGaming platform using Natural Language Processing (NLP). It extracts cleaned tokens, named entities, performs sentiment analysis, and generates visual insights to help understand player experience.

---

## 📌 Features

- ✅ Text preprocessing using `spaCy`
  - Tokenization, Lemmatization, Stopword removal
  - Named Entity Recognition (NER)
- ✅ Sentiment analysis using `TextBlob`
- ✅ Bar charts for:
  - Sentiment label distribution
  - Most common words
  - Named entity label frequency
- ✅ Cleaned output saved to `processed_feedback.csv`

---

## 📥 Sample Input (from `data/feedback.csv`)

```csv
id,feedback
1,"The app crashed every time I tried placing a bet on football."
2,"Customer support was slow and unhelpful."
...
20,"Bonuses are inconsistent. Sometimes I get them, sometimes I don’t."

---

## 📝 Notebook Version

A complete, interactive Jupyter Notebook is available:

📂 [NLP_Feedback_Analysis.ipynb](NLP_Feedback_Analysis.ipynb)

The notebook contains:
- Feedback processing
- Sentiment and entity analysis
- Token cleaning and visualization
- Inline charts for hiring or demo purposes

---

