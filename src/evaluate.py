import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_test_log, preds):
    # Convert actual values back to kWh
    y_test = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test_log, np.log1p(preds))

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    plt.figure(figsize=(12,5))
    plt.plot(y_test.values[:500], label="Actual")
    plt.plot(preds[:500], label="Predicted")
    plt.legend()
    plt.title("CycleMax Energy Forecast")
    plt.show()
