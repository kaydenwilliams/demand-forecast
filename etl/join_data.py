import pandas as pd

def join_all():
    fred     = pd.read_csv("data/raw/fred_data.csv",      parse_dates=["date"])
    trends   = pd.read_csv("data/raw/trends_data.csv",    parse_dates=["date"])
    sentiment= pd.read_csv("data/raw/sentiment_data.csv", parse_dates=["date"])
    sales    = pd.read_csv("data/raw/sales_data.csv",     parse_dates=["date"])

    df = sales.copy()
    df = df.merge(fred,      on="date", how="left")
    df = df.merge(trends,    on="date", how="left")
    df = df.merge(sentiment, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill()

    # promotional flag — marks months with revenue spike >10% above rolling mean
    df["promo_flag"] = (
        df["total_revenue"] > df["total_revenue"].rolling(3, min_periods=1).mean() * 1.1
    ).astype(int)

    # lag features — last month and 3 months ago revenue
    df["revenue_lag1"] = df["total_revenue"].shift(1)
    df["revenue_lag3"] = df["total_revenue"].shift(3)

    df = df.dropna()
    df.to_csv("data/processed/modeling_data.csv", index=False)
    print(f"Modeling dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df

if __name__ == "__main__":
    join_all()