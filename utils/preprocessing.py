import pandas as pd

def load_data():
    return pd.read_csv("data/traffic.csv")  # Update this path as needed

def prepare_data(df):
    df = df.rename(columns={
        "State/UT/City": "State/UT",
        "Grand Total - Cases": "Total"
    })

    # Drop rows with missing values
    df = df.dropna(subset=["State/UT", "Category", "Total"])

    # Clean types
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Total"])

    return df
