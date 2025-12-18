import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(y_test_log, preds):
    # 🔥 Convert ACTUAL values back to kWh
    y_test = np.expm1(y_test_log)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

    plt.figure(figsize=(12,5))
    plt.plot(y_test.values[:500], label="Actual")
    plt.plot(preds[:500], label="Predicted")
    plt.legend()
    plt.title("CycleMax Energy Forecast (Corrected Scale)")
    plt.show()
