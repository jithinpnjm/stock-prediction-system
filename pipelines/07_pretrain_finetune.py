"""Pretrain a TemporalTransformer encoder on next-bar-return prediction
(self-supervised, using every bar in the unlabeled history) then
fine-tune the same weights on the triple-barrier classification task.
Trained and evaluated per purged walk-forward fold: pretraining data
for a given fold is bounded to strictly before that fold's embargoed
test start, so no fold's evaluation ever benefits from data it
wouldn't have had live.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
import yaml
from sklearn.preprocessing import RobustScaler

from src.mlops.lineage import Lineage
from src.mlops.mlflow_utils import log_lineage, log_resolved_config
from src.models.sequence_data import build_sequences
from src.models.sequence_pretrain import pretrain_encoder
from src.models.sequence_training import sequence_predict_proba, train_sequence_classifier
from src.models.transformer import build_transformer
from src.validation.purged_cv import PurgedTimeSeriesSplit

_spec = importlib.util.spec_from_file_location(
    "seq_train", Path(__file__).parent / "07_sequence_train.py"
)
_seq_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seq_train)
SEQUENCE_FEATURE_COLUMNS = _seq_train.SEQUENCE_FEATURE_COLUMNS


def run():
    cfg = yaml.safe_load(Path("configs/models/sequence.yaml").read_text())
    tuned = cfg.get("transformer", {})
    clip_z = float(cfg.get("clip_z", 8.0))
    sequence_length = int(cfg["sequence_length"])

    # Fine-tune pool: labeled triple-barrier events.
    finetune_df = pl.read_parquet("data/gold/dataset_v1.parquet").sort("timestamp")
    names = [c for c in SEQUENCE_FEATURE_COLUMNS if c in finetune_df.columns]
    finetune_usable = finetune_df.drop_nulls(subset=names + ["label", "event_end_timestamp"]).sort(
        "timestamp"
    )

    ft_batch = build_sequences(
        finetune_usable.select(names).to_numpy(),
        finetune_usable["label"].to_numpy(),
        finetune_usable["timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        finetune_usable["event_end_timestamp"].dt.epoch(time_unit="ns").to_numpy(),
        sequence_length=sequence_length,
    )
    n_features = ft_batch.X.shape[2]

    # Pretrain pool: every bar (pre-labeling), no triple-barrier
    # requirement, much larger than the labeled set. Target is the
    # ATR-normalized return realized one bar later.
    pretrain_df = pl.read_parquet("data/silver/5m_features.parquet").sort("timestamp")
    pretrain_df = pretrain_df.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    pretrain_df = pretrain_df.with_columns(
        (
            (pl.col("close").shift(-1).over("_session_date") - pl.col("close"))
            / (pl.col("f_atr_14") + 1e-9)
        ).alias("_next_return")
    )
    pretrain_names = [c for c in SEQUENCE_FEATURE_COLUMNS if c in pretrain_df.columns]
    pretrain_usable = pretrain_df.drop_nulls(subset=pretrain_names + ["_next_return"]).sort(
        "timestamp"
    )
    pretrain_ts = pretrain_usable["timestamp"].dt.epoch(time_unit="ns").to_numpy()

    embargo_ns = int(cfg.get("embargo_bars", 75)) * 5 * 60 * 1_000_000_000
    split = PurgedTimeSeriesSplit(int(cfg.get("n_splits", 4)), embargo=embargo_ns)
    folds = list(split.split(ft_batch.timestamps, ft_batch.event_end))
    if not folds:
        raise RuntimeError("no valid fine-tune fold split")

    mlflow.set_experiment("BankNifty_Pretrain_Finetune")
    device_used = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    n_folds_used = 0
    final_model = None
    final_scaler = None

    with mlflow.start_run(run_name="transformer_pretrain_finetune_full_5y"):
        log_lineage(
            Lineage.create(
                dataset_id="banknifty_5m_v1",
                feature_version="pretrain_finetune_v1",
                label_version="triple_barrier_200_70_v1",
                validation_version="purged_time_series_v1",
            ).to_dict()
        )
        log_resolved_config({"sequence": cfg, "feature_columns": names})
        mlflow.log_param("device", device_used)
        mlflow.log_param("n_pretrain_rows_total", pretrain_usable.height)
        mlflow.log_param("n_finetune_sequences_total", ft_batch.X.shape[0])

        for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
            if len(train_idx) < int(cfg.get("min_train_rows", 500)):
                continue
            n_folds_used += 1

            # Bound pretraining strictly before this fold's embargoed
            # test start -- no fold ever pretrains on data it wouldn't
            # have had live.
            fold_test_start_ns = int(ft_batch.timestamps[val_idx].min())
            cutoff_ns = fold_test_start_ns - embargo_ns
            pt_mask = pretrain_ts < cutoff_ns
            pt_df = pretrain_usable.filter(pl.Series(pt_mask))
            if pt_df.height < sequence_length * 20:
                print(f"fold={fold_num}: not enough pretrain data ({pt_df.height} rows), skipping")
                continue

            pt_batch = build_sequences(
                pt_df.select(pretrain_names).to_numpy(),
                pt_df["_next_return"].to_numpy(),
                pt_df["timestamp"].dt.epoch(time_unit="ns").to_numpy(),
                sequence_length=sequence_length,
            )
            # simple chronological 90/10 split within the pretrain pool
            # for early stopping (not evaluated/reported -- purely to
            # know when to stop pretraining).
            n_pt = pt_batch.X.shape[0]
            pt_split = int(n_pt * 0.9)
            pt_train_idx = np.arange(0, pt_split)
            pt_val_idx = np.arange(pt_split, n_pt)

            pt_scaler = RobustScaler().fit(pt_batch.X[pt_train_idx].reshape(-1, n_features))
            pt_scaled = (
                np.clip(pt_scaler.transform(pt_batch.X.reshape(-1, n_features)), -clip_z, clip_z)
                .reshape(pt_batch.X.shape)
                .astype(np.float32)
            )

            model = build_transformer(
                n_features,
                d_model=int(tuned.get("d_model", 64)),
                n_heads=int(tuned.get("n_heads", 2)),
                n_layers=int(tuned.get("n_layers", 2)),
                dropout=float(tuned.get("dropout", 0.1)),
            )
            pt_result = pretrain_encoder(
                model,
                pt_scaled[pt_train_idx],
                pt_batch.y[pt_train_idx].astype(np.float32),
                pt_scaled[pt_val_idx],
                pt_batch.y[pt_val_idx].astype(np.float32),
                epochs=int(cfg.get("pretrain_epochs", 30)),
                batch_size=int(tuned.get("batch_size", 512)),
                learning_rate=float(cfg.get("pretrain_learning_rate", 1e-3)),
                patience=int(cfg.get("pretrain_patience", 6)),
            )
            mlflow.log_metric(f"fold_{fold_num}_pretrain_train_loss", pt_result.train_loss)
            mlflow.log_metric(f"fold_{fold_num}_pretrain_val_loss", pt_result.val_loss)
            print(
                f"fold={fold_num} pretrain: n_rows={pt_df.height} epoch={pt_result.best_epoch} "
                f"train_mse={pt_result.train_loss:.5f} val_mse={pt_result.val_loss:.5f}"
            )

            # Fine-tune the pretrained weights on this fold's labeled
            # triple-barrier data, scaled with this fold's own scaler
            # (fit on the fine-tune training split, same as the
            # baseline sequence trainer).
            ft_scaler = RobustScaler().fit(ft_batch.X[train_idx].reshape(-1, n_features))
            ft_scaled = (
                np.clip(ft_scaler.transform(ft_batch.X.reshape(-1, n_features)), -clip_z, clip_z)
                .reshape(ft_batch.X.shape)
                .astype(np.float32)
            )
            ft_result = train_sequence_classifier(
                pt_result.model,
                ft_scaled[train_idx],
                ft_batch.y[train_idx],
                ft_scaled[val_idx],
                ft_batch.y[val_idx],
                epochs=int(cfg["epochs"]),
                batch_size=int(tuned.get("batch_size", 512)),
                learning_rate=float(tuned.get("learning_rate", cfg["learning_rate"])),
                patience=int(cfg["patience"]),
            )
            probs = sequence_predict_proba(ft_result.model, ft_scaled[val_idx])

            mlflow.log_metric(f"fold_{fold_num}_finetune_train_loss", ft_result.train_loss)
            mlflow.log_metric(f"fold_{fold_num}_finetune_val_loss", ft_result.val_loss)
            print(
                f"fold={fold_num} finetune: epoch={ft_result.best_epoch} "
                f"train_loss={ft_result.train_loss:.5f} val_loss={ft_result.val_loss:.5f} "
                f"n_train={len(train_idx)} n_val={len(val_idx)}"
            )

            for idx, row in zip(val_idx, probs):
                rows.append(
                    {
                        "timestamp": int(ft_batch.timestamps[idx]),
                        "event_end_timestamp": int(ft_batch.event_end[idx]),
                        "p_short": float(row[0]),
                        "p_none": float(row[1]),
                        "p_long": float(row[2]),
                        "label": int(ft_batch.y[idx]),
                        "fold": fold_num,
                    }
                )
            final_model = ft_result.model
            final_scaler = ft_scaler

        if not rows:
            raise RuntimeError("no valid pretrain/finetune folds were produced")
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
            f"pretrain_finetune: folds_used={n_folds_used} n_oof={out.height} "
            f"max_p_long={out['p_long'].max():.3f} max_p_short={out['p_short'].max():.3f} "
            f"device={device_used}"
        )
        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        out.write_parquet("data/predictions/transformer_pretrained_oof.parquet")

        if final_model is not None:
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": final_model.state_dict(),
                    "scaler_center": final_scaler.center_,
                    "scaler_scale": final_scaler.scale_,
                    "feature_columns": names,
                    "sequence_length": sequence_length,
                    "clip_z": clip_z,
                    "d_model": int(tuned.get("d_model", 64)),
                    "n_heads": int(tuned.get("n_heads", 2)),
                    "n_layers": int(tuned.get("n_layers", 2)),
                    "dropout": float(tuned.get("dropout", 0.1)),
                },
                "models/transformer_pretrained_final.pt",
            )
            mlflow.log_artifact("models/transformer_pretrained_final.pt")
            print("saved models/transformer_pretrained_final.pt (last fold's fine-tuned weights)")


if __name__ == "__main__":
    run()
