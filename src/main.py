from data_loader import load_data
from feature_engineering import create_features
from model import train_model, predict
from evaluate import evaluate_model

def main():
    df = load_data("data/smart_home_energy_consumption_large.csv")
    ts_df = create_features(df)

    model, X_test, y_test = train_model(ts_df)
    preds = predict(model, X_test)

    evaluate_model(y_test, preds)

if __name__ == "__main__":
    main()
