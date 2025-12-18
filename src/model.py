from xgboost import XGBRegressor
import numpy as np
import joblib
import os

def train_model(ts_df):
    split = int(len(ts_df) * 0.8)
    train = ts_df.iloc[:split]
    test = ts_df.iloc[split:]

    X_train = train.drop("Energy Consumption (kWh)", axis=1)
    X_test = test.drop("Energy Consumption (kWh)", axis=1)

    y_train = np.log1p(train["Energy Consumption (kWh)"])
    y_test = np.log1p(test["Energy Consumption (kWh)"])

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    weights = train["Energy Consumption (kWh)"]
    weights = weights / weights.max()

    model.fit(X_train, y_train, sample_weight=weights)

    # 🔥 SAVE MODEL
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/cyclemax_energy_model.pkl")

    print("✅ Model saved as models/cyclemax_energy_model.pkl")

    return model, X_test, y_test


def predict(model, X_test):
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    return preds
