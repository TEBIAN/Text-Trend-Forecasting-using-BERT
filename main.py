
from text_trend_forecasting import fetch_tweets, analyze_sentiment, prepare_time_series, forecast_sentiment, plot_forecast

if __name__ == "__main__":
    QUERY = "#AI lang:en -is:retweet"
    MAX_TWEETS = 100

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
