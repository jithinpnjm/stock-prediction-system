import joblib
import polars as pl
from sklearn.metrics import accuracy_score, classification_report

from src.models.lightgbm import predict_lightgbm


def run():
    print("Running pipeline step: 08_validate.py")

    try:
        X = pl.read_parquet("data/ml/X.parquet").to_numpy()
        y_true = pl.read_parquet("data/ml/y.parquet").to_series().to_numpy()
        models = joblib.load("models/lightgbm_cv_models.pkl")
    except FileNotFoundError:
        print("Run 07_train.py first.")
        return

    # Ensemble prediction (majority vote or average probability)
    # For simplicity, we just take the first model to demonstrate the validation loop.
    model = models[0]

    y_pred = predict_lightgbm(model, X)

    print("\nValidation Results (Using Fold 1 Model on Full Data for Demo):")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    run()
