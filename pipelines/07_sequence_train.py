from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.mlops.lineage import Lineage
from src.mlops.mlflow_utils import log_lineage, log_resolved_config
from src.models.sequence_data import build_sequences
from src.models.sequence_training import sequence_predict_proba, train_sequence_classifier
from src.models.tcn import build_tcn
from src.models.transformer import build_transformer
from src.validation.purged_cv import PurgedTimeSeriesSplit

# Feature families available from bar 1 (or close to it) of every
# session. support_resistance (needs 48 bars) and historical_intraday
# (needs the first 20 trading days) are deliberately excluded here:
# both null out exactly the early-session bars that this sequence
# model most needs to see to learn first-candle/opening-structure
# behavior. structure and multi_timeframe are excluded because the
# feature funnel (pipelines/07_feature_funnel.py) found they added no
# incremental value over volatility alone on the same row population.
SEQUENCE_FEATURE_COLUMNS = [
    "f_range",
    "f_body",
    "f_return_1",
    "f_upper_wick",
    "f_lower_wick",
    "f_close_location",
    "f_body_pct",
    "f_upper_wick_pct",
    "f_lower_wick_pct",
    "f_direction",
    "f_range_pct",
    *[
        f"f_cluster_{n}_{stat}"
        for n in range(1, 11)
        for stat in (
            "return",
            "range",
            "body",
            "direction_balance",
            "up_count",
            "down_count",
            "high",
            "low",
            "efficiency",
            # volume_mean deliberately excluded: raw volume is 0 for
            # the whole dataset before 2025-07-01.
        )
    ],
    "f_atr_6",
    "f_abs_return_6",
    "f_tr_std_6",
    "f_range_to_atr_6",
    "f_natr_6_bps",
    "f_atr_14",
    "f_abs_return_14",
    "f_tr_std_14",
    "f_range_to_atr_14",
    "f_natr_14_bps",
    "f_volatility_shock",
    "f_cusum_event",
    "f_cusum_threshold",
    "f_hour",
    "f_minute",
    "f_weekday",
    "f_minutes_from_open",
    "f_minutes_to_close",
    "f_session_bar_index",
    "f_session_phase",
    "f_time_sin",
    "f_time_cos",
    "f_gap_from_prev_close",
    "f_gap_from_prev_close_pct",
    "f_distance_prev_high",
    "f_distance_prev_low",
    "f_prev_day_range_position",
    "f_prev_day_return",
    "f_prev_day_range",
    "f_intraday_high_pullback",
    "f_intraday_low_rebound",
    "f_or3_range",
    "f_or3_position",
    "f_distance_to_or3_high",
    "f_distance_to_or3_low",
    "f_or6_range",
    "f_or6_position",
    "f_distance_to_or6_high",
    "f_distance_to_or6_low",
    # f_1m_path_volume deliberately excluded: raw volume is 0 for the
    # whole dataset before 2025-07-01.
    "f_1m_range_sum",
    "f_1m_range_mean",
    "f_1m_range_std",
    "f_1m_up_count",
    "f_1m_down_count",
    "f_1m_count",
    "f_1m_path_net_move",
    "f_1m_path_range",
    "f_1m_path_high_rejection",
    "f_1m_path_low_rebound",
    "f_1m_up_ratio",
]


def run(model_family: str = "tcn"):
    cfg = yaml.safe_load(Path("configs/models/sequence.yaml").read_text())
    dataset = pl.read_parquet("data/gold/dataset_v1.parquet").sort("timestamp")
    names = [c for c in SEQUENCE_FEATURE_COLUMNS if c in dataset.columns]
    usable = dataset.drop_nulls(subset=names + ["label", "event_end_timestamp"]).sort("timestamp")

    batch = build_sequences(
        usable.select(names).to_numpy(),
        usable["label"].to_numpy(),
        usable["timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        usable["event_end_timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        sequence_length=int(cfg["sequence_length"]),
    )
    embargo_ns = int(cfg.get("embargo_bars", 75)) * 5 * 60 * 1_000_000_000
    split = PurgedTimeSeriesSplit(n_splits=int(cfg.get("n_splits", 4)), embargo=embargo_ns)
    folds = list(split.split(batch.timestamps, batch.event_end))
    if not folds:
        raise RuntimeError("no valid sequence validation split")
    n_features = batch.X.shape[2]

    mlflow.set_experiment("BankNifty_Sequence_Models")
    with mlflow.start_run(run_name=f"{model_family}_full_5y"):
        lineage = Lineage.create(
            dataset_id="banknifty_5m_v1",
            feature_version="sequence_early_session_v1",
            label_version="triple_barrier_200_70_v1",
            validation_version="purged_time_series_v1",
        )
        log_lineage(lineage.to_dict())
        log_resolved_config({"sequence": cfg, "feature_columns": names})
        mlflow.log_params(
            {
                "model_family": model_family,
                "n_features": n_features,
                "sequence_length": int(cfg["sequence_length"]),
                "n_sequences": batch.X.shape[0],
                "n_folds_configured": int(cfg.get("n_splits", 4)),
            }
        )
        device_used = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device_used)

        rows = []
        n_folds_used = 0
        for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
            if len(train_idx) < int(cfg.get("min_train_rows", 500)):
                continue
            n_folds_used += 1

            # Standardize per feature, fit on this fold's training
            # split only, so raw-scale features don't blow up the
            # network's activations/loss the way they can for tree
            # models -- and so no information from a later fold's
            # validation period leaks into an earlier fold's scaling.
            scaler = StandardScaler().fit(batch.X[train_idx].reshape(-1, n_features))
            scaled_X = (
                scaler.transform(batch.X.reshape(-1, n_features))
                .reshape(batch.X.shape)
                .astype(np.float32)
            )

            if model_family == "tcn":
                model = build_tcn(n_features)
                train_X = scaled_X.transpose(0, 2, 1)
                result = train_sequence_classifier(
                    model,
                    train_X[train_idx],
                    batch.y[train_idx],
                    train_X[val_idx],
                    batch.y[val_idx],
                    epochs=int(cfg["epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    learning_rate=float(cfg["learning_rate"]),
                    patience=int(cfg["patience"]),
                    device=None if cfg.get("device", "auto") == "auto" else cfg["device"],
                )
                probs = sequence_predict_proba(result.model, train_X[val_idx])
            elif model_family == "transformer":
                model = build_transformer(n_features)
                result = train_sequence_classifier(
                    model,
                    scaled_X[train_idx],
                    batch.y[train_idx],
                    scaled_X[val_idx],
                    batch.y[val_idx],
                    epochs=int(cfg["epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    learning_rate=float(cfg["learning_rate"]),
                    patience=int(cfg["patience"]),
                    device=None if cfg.get("device", "auto") == "auto" else cfg["device"],
                )
                probs = sequence_predict_proba(result.model, scaled_X[val_idx])
            else:
                raise ValueError("model_family must be tcn or transformer")

            mlflow.log_metric(f"fold_{fold_num}_train_loss", result.train_loss)
            mlflow.log_metric(f"fold_{fold_num}_val_loss", result.val_loss)
            print(
                f"{model_family} fold={fold_num} epoch={result.best_epoch} "
                f"train_loss={result.train_loss:.5f} val_loss={result.val_loss:.5f} "
                f"n_train={len(train_idx)} n_val={len(val_idx)}"
            )
            for idx, row in zip(val_idx, probs):
                rows.append(
                    {
                        "timestamp": int(batch.timestamps[idx]),
                        "event_end_timestamp": int(batch.event_end[idx]),
                        "p_short": float(row[0]),
                        "p_none": float(row[1]),
                        "p_long": float(row[2]),
                        "label": int(batch.y[idx]),
                        "fold": fold_num,
                    }
                )

        if not rows:
            raise RuntimeError("no valid sequence folds were produced")
        mlflow.log_metric("n_folds_used", n_folds_used)
        out = (
            pl.DataFrame(rows)
            .with_columns(
                pl.col("timestamp").cast(pl.Datetime("ns")).dt.replace_time_zone("Asia/Kolkata"),
                pl.col("event_end_timestamp")
                .cast(pl.Datetime("ns"))
                .dt.replace_time_zone("Asia/Kolkata"),
            )
            .sort("timestamp")
        )
        mlflow.log_metric("n_oof", out.height)
        mlflow.log_metric("max_p_long", float(out["p_long"].max()))
        mlflow.log_metric("max_p_short", float(out["p_short"].max()))
        print(
            f"{model_family}: folds_used={n_folds_used} n_oof={out.height} "
            f"max_p_long={out['p_long'].max():.3f} max_p_short={out['p_short'].max():.3f} "
            f"device={device_used}"
        )
        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        out.write_parquet(f"data/predictions/{model_family}_oof.parquet")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "tcn")
