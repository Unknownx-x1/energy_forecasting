import numpy as np

def create_features(df):
    # Aggregate per timestamp
    ts_df = df.groupby("timestamp")["Energy Consumption (kWh)"].sum().to_frame()

    # ---------- Time features ----------
    ts_df["hour"] = ts_df.index.hour
    ts_df["dayofweek"] = ts_df.index.dayofweek

    ts_df["hour_sin"] = np.sin(2 * np.pi * ts_df["hour"] / 24)
    ts_df["hour_cos"] = np.cos(2 * np.pi * ts_df["hour"] / 24)

    ts_df["dow_sin"] = np.sin(2 * np.pi * ts_df["dayofweek"] / 7)
    ts_df["dow_cos"] = np.cos(2 * np.pi * ts_df["dayofweek"] / 7)

    # ---------- Peak features ----------
    ts_df["daily_max"] = ts_df["Energy Consumption (kWh)"].rolling(24).max()
    ts_df["weekly_max"] = ts_df["Energy Consumption (kWh)"].rolling(24 * 7).max()

    # ---------- Lag features ----------
    ts_df["lag_1"] = ts_df["Energy Consumption (kWh)"].shift(1)
    ts_df["lag_24"] = ts_df["Energy Consumption (kWh)"].shift(24)
    ts_df["lag_48"] = ts_df["Energy Consumption (kWh)"].shift(48)

    # ---------- 🔥 TARGET FIX ----------
    # Predict EXPECTED next-hour load, not raw spike
    ts_df["target"] = (
        ts_df["Energy Consumption (kWh)"]
        .rolling(window=3)
        .mean()
        .shift(-1)
    )

    ts_df.dropna(inplace=True)
    return ts_df
