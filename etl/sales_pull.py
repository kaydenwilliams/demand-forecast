import pandas as pd

def pull_sales_data():
    df1 = pd.read_excel("data/raw/retail_sales.xlsx", sheet_name="Year 2009-2010", engine="openpyxl")
    df2 = pd.read_excel("data/raw/retail_sales.xlsx", sheet_name="Year 2010-2011", engine="openpyxl")
    df = pd.concat([df1, df2], ignore_index=True)

    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Customer ID", "InvoiceDate"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["revenue"] = df["Quantity"] * df["Price"]
    df["date"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("date").agg(
        total_revenue=("revenue", "sum"),
        total_units=("Quantity", "sum"),
        num_transactions=("Invoice", "nunique")
    ).reset_index()

    monthly = monthly.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    monthly.to_csv("data/raw/sales_data.csv", index=False)
    print(f"Sales data saved: {monthly.shape[0]} rows")
    return monthly

if __name__ == "__main__":
    pull_sales_data()