import polars as pl
import numpy as np
import os
import joblib
import mlflow
import mlflow.lightgbm
from src.validation.purged_cv import PurgedTimeSeriesSplit
from src.models.lightgbm import train_lightgbm_classifier

def run():
    print("Running pipeline step: 07_train.py")
    
    try:
        X = pl.read_parquet("data/ml/X.parquet").to_numpy()
        y = pl.read_parquet("data/ml/y.parquet").to_series().to_numpy()
    except FileNotFoundError:
        print("Run 06_build_dataset.py first.")
        return

    # Assuming we will have thousands of 5m bars, embargo is ~75.
    cv = PurgedTimeSeriesSplit(n_splits=3, embargo_size=75)
    
    models = []
    
    mlflow.set_experiment("BankNifty_Baseline_LGBM")
    
    with mlflow.start_run(run_name="Purged_CV_GPU_Train"):
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            print(f"Training Fold {fold+1} on GPU...")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Since mock data won't have enough rows for an embargo of 75, we fallback if indices are empty
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"Skipping fold {fold+1} due to small data size for embargo testing.")
                continue

            model = train_lightgbm_classifier(X_train, y_train, X_val, y_val)
            models.append(model)
            
            # Log Model specifically for this fold
            mlflow.lightgbm.log_model(model, artifact_path=f"model_fold_{fold+1}")
            
        # Optional: Save locally as well
        if models:
            os.makedirs("models", exist_ok=True)
            joblib.dump(models, "models/lightgbm_cv_models.pkl")
            print("Training complete. Models logged to MLflow and saved locally.")
        else:
            print("No models trained. Data might be too small for the Purged CV split sizes.")

if __name__ == "__main__":
    run()
