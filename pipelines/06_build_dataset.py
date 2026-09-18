import os

import polars as pl


def run():
    print("Running pipeline step: 06_build_dataset.py")

    try:
        df_labels = pl.read_parquet("data/gold/dataset_v1.parquet")
    except FileNotFoundError:
        print("Run 05_labels.py first.")
        return

    # Drop datetime and any non-feature columns
    features_df = df_labels.drop(["datetime", "label"])
    target_df = df_labels.select(["label"])

    # Simple imputation for NaN/Null values
    features_df = features_df.fill_null(0.0).fill_nan(0.0)

    os.makedirs("data/ml", exist_ok=True)
    features_df.write_parquet("data/ml/X.parquet")
    target_df.write_parquet("data/ml/y.parquet")

    print(f"ML dataset built. X shape: {features_df.shape}, y shape: {target_df.shape}")


if __name__ == "__main__":
    run()
