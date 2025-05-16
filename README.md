
# 📈 Text Trend Forecasting from Social Media Sentiment

This project extracts and analyzes sentiment trends from Twitter data using **BERT-based sentiment analysis** and forecasts future sentiment using **transformer-based time series models**. It enables real-time insights into public opinion shifts around topics like AI, legal reforms, and social issues.

---

## 🚀 Features

- Collects live tweet data via Twitter API
- Applies BERT for accurate multilingual sentiment analysis
- Aggregates daily sentiment scores
- Forecasts future sentiment trends using transformers
- Visualizes historical and predicted sentiment

---

## 🛠 Tech Stack

- Python 3.8+
- Tweepy (Twitter API)
- Hugging Face Transformers (BERT)
- Pandas, Numpy, Matplotlib
- Darts (Time series forecasting)

---

## 📂 Project Structure

```
text-trend-forecasting/
├── data/
│   └── tweets.csv                # Collected and processed tweets
├── notebooks/
│   └── forecasting_pipeline.ipynb # Main sentiment + forecasting pipeline
├── models/
│   └── bert_sentiment_model/     # Optional saved model (if fine-tuned)
├── main.py                       # CLI or app entry point
├── requirements.txt              # Required dependencies
└── README.md
```

---

## 📊 Use Cases

- Forecast public sentiment on legal or political topics
- Detect and visualize mood shifts on trending hashtags
- Analyze public opinion trends for product launches or campaigns
- Social research on behavioral or emotional signals from text

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/text-trend-forecasting.git
cd text-trend-forecasting
pip install -r requirements.txt
```

---

## 🔐 Twitter API Setup

1. Sign up at [https://developer.twitter.com](https://developer.twitter.com)
2. Generate a Bearer Token
3. Replace in your script:
```python
bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
```

---

## 📈 How It Works

### 1. Data Collection
- Collect tweets by keyword/hashtag with Tweepy.

### 2. Sentiment Analysis
- Use BERT (`nlptown/bert-base-multilingual-uncased-sentiment`) to classify sentiment (1–5 scale).

### 3. Time Aggregation
- Average sentiment per day.

### 4. Forecasting
- Fit time series model with `darts` and forecast future sentiment.

---

## 🖼 Sample Output

- Daily sentiment line graph
- 7-day forecast overlay
- Optional anomaly detection for spikes or drops

---

## 📌 Example Keywords

```python
keywords = ["#AI", "#LegalTech", "privacy law", "social justice"]
```

---

## 📄 License

MIT License

---

## 🤝 Contributions

PRs and suggestions are welcome! Feel free to fork and improve.
