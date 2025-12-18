from xgboost import XGBRegressor
import numpy as np

def train_model(ts_df):
    split = int(len(ts_df) * 0.8)
    train = ts_df.iloc[:split]
    test = ts_df.iloc[split:]

    X_train = train.drop(["Energy Consumption (kWh)", "target"], axis=1)
    X_test = test.drop(["Energy Consumption (kWh)", "target"], axis=1)

    y_train = np.log1p(train["target"])
    y_test = np.log1p(test["target"])

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def predict(model, X_test):
    preds_log = model.predict(X_test)
    return np.expm1(preds_log)
