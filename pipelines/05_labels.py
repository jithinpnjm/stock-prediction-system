import os

import polars as pl

from src.labels.triple_barrier import apply_triple_barrier_labels


def run():
    print("Running pipeline step: 05_labels.py")

    try:
        df_features = pl.read_parquet("data/silver/5m_features.parquet")
    except FileNotFoundError:
        print("Run 04_features.py first.")
        return

    df_labels = apply_triple_barrier_labels(
        df_features, pt_sl_ratio=200.0 / 70.0, stop_loss_pts=70.0
    )

    os.makedirs("data/gold", exist_ok=True)
    df_labels.write_parquet("data/gold/dataset_v1.parquet")
    print(f"Labels applied. Shape: {df_labels.shape}")


if __name__ == "__main__":
    run()
