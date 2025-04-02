# 🧠 NLP Player Feedback Analysis (iGaming)

This project processes player feedback from an iGaming platform using Natural Language Processing (NLP). It extracts cleaned tokens, named entities (like game names, dates), and stores them in a structured format.

## 📌 Features

- Text preprocessing using `spaCy`
  - Tokenization
  - Lemmatization
  - Stopword removal
  - Named Entity Recognition (NER)
- Reads input from `data/feedback.csv`
- Outputs results to `outputs/processed_feedback.csv`

## 🧪 Sample Input

```csv
id,feedback
1,"The app crashed every time I tried placing a bet on football."
2,"Customer support was slow and unhelpful."
