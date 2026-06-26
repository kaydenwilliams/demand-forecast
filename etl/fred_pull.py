import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

def pull_fred_data(start="2009-01-01", end="2011-12-31"):
    series = {
        "consumer_sentiment": "UMCSENT",
        "disposable_income":  "DSPIC96",
        "cpi":                "CPIAUCSL",
        "unemployment":       "UNRATE"
    }

    frames = []
    for name, code in series.items():
        df = fred.get_series(code, start, end).rename(name)
        frames.append(df)

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.resample("MS").mean()
    combined.index.name = "date"
    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    combined.to_csv("data/raw/fred_data.csv", index=False)
    print(f"FRED data saved: {combined.shape[0]} rows")
    return combined

if __name__ == "__main__":
    pull_fred_data()