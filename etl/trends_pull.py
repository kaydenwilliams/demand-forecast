import pandas as pd
from pytrends.request import TrendReq
import time

def pull_trends_data(keywords=["grocery delivery","meal kit","food inflation","canned goods"],
                     start="2009-01-01", end="2011-12-31"):
    pytrends = TrendReq(hl="en-US", tz=360)
    frames = []

    for kw in keywords:
        pytrends.build_payload([kw], timeframe=f"{start} {end}", geo="US")
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df[[kw]].rename(columns={kw: kw.replace(" ", "_")})
            frames.append(df)
        time.sleep(2)  # avoid rate limiting

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.resample("MS").mean()
    combined.index.name = "date"
    combined.reset_index(inplace=True)
    combined.to_csv("data/raw/trends_data.csv", index=False)
    print(f"Trends data saved: {combined.shape[0]} rows")
    return combined

if __name__ == "__main__":
    pull_trends_data()