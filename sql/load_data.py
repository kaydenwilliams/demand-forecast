import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

def load_to_mysql():
    password = quote_plus("P@ssw0rd!")  # handles special characters
    engine = create_engine(f"mysql+pymysql://root:{password}@127.0.0.1/demand_forecast")
    df = pd.read_csv("data/processed/modeling_data.csv", parse_dates=["date"])
    df.to_sql("modeling_data", engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into MySQL")

if __name__ == "__main__":
    load_to_mysql()