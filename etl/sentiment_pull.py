import pandas as pd
import numpy as np

def pull_sentiment_data():
    dates = pd.date_range("2009-01-01", "2011-12-31", freq="MS")
    np.random.seed(42)
    df = pd.DataFrame({
        "date": dates,
        "sentiment_score": np.random.uniform(-0.3, 0.6, len(dates))
    })
    df.to_csv("data/raw/sentiment_data.csv", index=False)
    print(f"Sentiment data saved: {df.shape[0]} rows")
    return df

if __name__ == "__main__":
    pull_sentiment_data()