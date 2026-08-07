# 🇬🇧 London YouTube Sentiment Dashboard

A live AI-powered sentiment analysis dashboard that fetches real YouTube comments about London restaurants, attractions and tourism — classifies them as Positive, Negative or Neutral using NLP — and displays results in an interactive dashboard.

🔗 **Live App:** [london-sentiment-dashboard.streamlit.app](https://london-sentiment-dashboard-jytfrtvzpctpvau4rwpo3p.streamlit.app)

---

## 📌 What This Project Does

This dashboard fetches live YouTube comments about London tourism and classifies each one as **Positive**, **Negative**, or **Neutral** using Natural Language Processing. Results appear in real-time interactive charts, metrics, and a filterable comment browser — updated every session with fresh data.

---

## 🔄 Pipeline
User selects a topic (e.g. "London restaurant review")
↓
YouTube Data API v3 searches for matching videos
↓
Up to 25 comments fetched per video
↓
TextBlob reads each comment and returns a polarity score (-1.0 to +1.0)
↓
Score converted to label: POSITIVE / NEGATIVE / NEUTRAL
↓
All results stored in a Pandas DataFrame
↓
Plotly renders donut chart + bar chart from the DataFrame
↓
Streamlit displays everything live in the browser

---

## 🎯 Why YouTube Comments?

YouTube comments are one of the most authentic sources of public opinion about real experiences. Unlike structured reviews or star ratings, comments capture nuanced sentiment in natural language — frustration, excitement, recommendations, criticism — from real visitors. For London tourism analysis, comments on restaurant vlogs, attraction videos and travel guides reflect genuine visitor sentiment at a scale that surveys cannot match.

---

## 🧠 Why TextBlob — Not HuggingFace or Transformers?

Three sentiment approaches were evaluated before settling on TextBlob:

**HuggingFace Transformers (DistilBERT)** — Most accurate but requires PyTorch (2GB+ dependency), which exceeds Streamlit Cloud's free tier memory limits. Also trialled via the HuggingFace Inference API, but Streamlit Cloud's infrastructure blocks outbound connections to external ML APIs — making it completely unusable in deployment.

**VADER** — Designed for social media text but lacks the clean polarity scoring that maps naturally to a three-class label system.

**TextBlob** — Selected because it runs entirely on the server with no external API calls, no heavy dependencies, and no deployment instability. It produces a polarity score between -1.0 and +1.0 which maps cleanly to sentiment labels. While less accurate than transformer-based models on complex text, it handles short informal YouTube comments reliably and deploys instantly on any server.

Sentiment thresholds used:
- Polarity > 0.05 → **POSITIVE**
- Polarity < -0.05 → **NEGATIVE**
- Between -0.05 and 0.05 → **NEUTRAL**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| YouTube Data API v3 | Fetches live comments from YouTube |
| TextBlob | NLP sentiment analysis — runs locally |
| Pandas | Data organisation and filtering |
| Plotly | Interactive charts |
| Streamlit | Web app framework and cloud deployment |

---

## 📊 Features

- Live YouTube comment fetching per session
- 5 London tourism topics to choose from
- Adjustable number of videos (2–8)
- Sentiment filter — All, Positive, Negative, Neutral
- Donut chart showing overall sentiment split
- Bar chart showing sentiment breakdown per video
- Expandable comment cards with score per comment

---

## 👨‍💻 Author

**Mithun Surriya KS**
BSc Artificial Intelligence and Data Science — University of East London (89%)
[LinkedIn](https://www.linkedin.com/in/mithun-surriya-ks-62b9a8203/) · [GitHub](https://github.com/mithun-ksms)
