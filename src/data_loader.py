import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("timestamp")
    df.drop(["Date", "Time"], axis=1, inplace=True)
    return df
