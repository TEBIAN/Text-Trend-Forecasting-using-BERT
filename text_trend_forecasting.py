
# text_trend_forecasting.py

import tweepy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TransformerModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape
import torch

# -------------------- CONFIG --------------------
BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"
QUERY = "#AI lang:en -is:retweet"
MAX_TWEETS = 100
DATE_FORMAT = "%Y-%m-%d"

# -------------------- TWITTER API --------------------
def fetch_tweets(query, max_tweets):
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    response = client.search_recent_tweets(query=query, max_results=max_tweets, tweet_fields=["created_at", "text"])
    tweets = [{"date": tweet.created_at.date(), "text": tweet.text} for tweet in response.data]
    return pd.DataFrame(tweets)

# -------------------- SENTIMENT ANALYSIS --------------------
def analyze_sentiment(df):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    df["score"] = df["text"].apply(lambda x: int(sentiment_pipeline(x[:512])[0]['label'][0]))
    return df

# -------------------- AGGREGATE TIME SERIES --------------------
def prepare_time_series(df):
    df_grouped = df.groupby("date")["score"].mean().reset_index()
    df_grouped.columns = ["date", "sentiment"]
    df_grouped["date"] = pd.to_datetime(df_grouped["date"])
    return df_grouped

# -------------------- FORECAST --------------------
def forecast_sentiment(df_grouped, n_forecast=7):
    series = TimeSeries.from_dataframe(df_grouped, "date", "sentiment")
    model = TransformerModel(input_chunk_length=7, output_chunk_length=7, n_epochs=100, random_state=42)
    model.fit(series)
    forecast = model.predict(n_forecast)
    return series, forecast

# -------------------- PLOT --------------------
def plot_forecast(series, forecast):
    plt.figure(figsize=(10, 5))
    series.plot(label="Past")
    forecast.plot(label="Forecast")
    plt.title("Sentiment Trend Forecast")
    plt.legend()
    plt.show()

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("Fetching tweets...")
    df = fetch_tweets(QUERY, MAX_TWEETS)

    print("Analyzing sentiment...")
    df = analyze_sentiment(df)

    print("Preparing time series...")
    df_grouped = prepare_time_series(df)

    print("Forecasting...")
    series, forecast = forecast_sentiment(df_grouped)

    print("Plotting...")
    plot_forecast(series, forecast)
