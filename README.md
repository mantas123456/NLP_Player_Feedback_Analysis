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

## 📥 Sample Input (from `data/feedback.csv`)

```csv
id,feedback
1,"The app crashed every time I tried placing a bet on football."
2,"Customer support was slow and unhelpful."
...
20,"Bonuses are inconsistent. Sometimes I get them, sometimes I don’t."

---


---

## 📦 Download the Project

You can download the full project as a ZIP from the GitHub Releases page:

➡️ [Download latest release](https://github.com/mantas123456/NLP_Player_Feedback_Analysis/releases)

Includes:
- Jupyter Notebook (`.ipynb`)
- Python script (`main.py`)
- Sample feedback dataset (`feedback.csv`)
- Cleaned output with sentiment & NER
- Charts and visualizations (`.png`)
- README file with project overview

## 📘 Version History

- **v1.0** – Initial release with NLP pipeline, sentiment analysis, and visualizations
