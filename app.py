import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Energy Forecast", layout="centered")

# Load trained model
model = joblib.load("models/cyclemax_energy_model.pkl")

st.title("⚡ Energy Consumption Forecast")
st.write("Predict **next-hour energy consumption** using a trained CycleMax model")

st.divider()

# -------------------------
# USER INPUTS
# -------------------------
lag_1 = st.number_input("Energy last hour (kWh)", value=1.0)
lag_24 = st.number_input("Energy 24 hours ago (kWh)", value=1.2)
lag_48 = st.number_input("Energy 48 hours ago (kWh)", value=1.1)

daily_max = st.number_input("Max energy (last 24h)", value=2.5)
weekly_max = st.number_input("Max energy (last 7 days)", value=3.5)

now = datetime.now()
hour = now.hour
dayofweek = now.weekday()

# Cyclic encoding
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin = np.sin(2 * np.pi * dayofweek / 7)
dow_cos = np.cos(2 * np.pi * dayofweek / 7)

# Create input dataframe
input_df = pd.DataFrame([{
    "hour": hour,
    "dayofweek": dayofweek,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "dow_sin": dow_sin,
    "dow_cos": dow_cos,
    "daily_max": daily_max,
    "weekly_max": weekly_max,
    "lag_1": lag_1,
    "lag_24": lag_24,
    "lag_48": lag_48
}])

st.divider()

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔮 Predict Next Hour"):
    pred_log = model.predict(input_df)[0]
    prediction = np.expm1(pred_log)

    st.success(f"⚡ Predicted Energy Consumption: **{prediction:.2f} kWh**")

    # Simple chart
    times = [
        now - timedelta(hours=2),
        now - timedelta(hours=1),
        now,
        now + timedelta(hours=1)
    ]
    values = [lag_48, lag_24, lag_1, prediction]

    fig, ax = plt.subplots()
    ax.plot(times, values, marker="o")
    ax.set_title("Energy Trend")
    ax.set_ylabel("kWh")
    st.pyplot(fig)
