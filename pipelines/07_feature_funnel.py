from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import accuracy_score, log_loss

from src.mlops.lineage import Lineage
from src.mlops.mlflow_utils import log_lineage, log_resolved_config
from src.models.lightgbm import predict_proba, train_lightgbm_classifier
from src.validation.calibration import brier_score, expected_calibration_error
from src.validation.purged_cv import PurgedTimeSeriesSplit

# Ordered per docs/trading-research-plan.md section 5. VWAP is listed
# there but is not currently generated (enable_vwap=false in
# configs/features/default.yaml) so it is skipped; multi-timeframe
# context is a real, wired feature family the plan doesn't name
# explicitly, inserted after volatility since it's a similar
# higher-timeframe-context signal.
FUNNEL_STAGES: list[tuple[str, list[str]]] = [
    (
        "cluster",
        [
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
                    "volume_mean",
                )
            ],
        ],
    ),
    (
        "volatility",
        [
            *[f"f_atr_{p}" for p in (6, 14, 30)],
            *[f"f_abs_return_{p}" for p in (6, 14, 30)],
            *[f"f_tr_std_{p}" for p in (6, 14, 30)],
            *[f"f_range_to_atr_{p}" for p in (6, 14, 30)],
            *[f"f_natr_{p}_bps" for p in (6, 14, 30)],
            "f_volatility_shock",
            "f_cusum_event",
            "f_cusum_threshold",
        ],
    ),
    (
        "multi_timeframe",
        [
            f"f_mtf_{tf}_{stat}"
            for tf in (15, 30, 60)
            for stat in ("distance_ma", "range_position", "return")
        ],
    ),
    (
        "structure",
        [
            "f_swing_high_candidate",
            "f_swing_low_candidate",
            "f_breakout_above_prior_high",
            "f_breakdown_below_prior_low",
            "f_structure_return",
            "f_last_swing_high",
            "f_last_swing_low",
            "f_higher_high",
            "f_lower_high",
            "f_higher_low",
            "f_lower_low",
            "f_structure_trend",
            "f_position_vs_structure_high",
            "f_position_vs_structure_low",
            "f_break_of_structure_up",
            "f_break_of_structure_down",
            "f_trend_context",
            "f_short_structure_range",
        ],
    ),
    (
        "support_resistance",
        [
            "f_resistance",
            "f_support",
            "f_distance_to_resistance",
            "f_distance_to_support",
            "f_resistance_touches",
            "f_support_touches",
            "f_position_in_sr_range",
        ],
    ),
    (
        "session_opening_range",
        [
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
            "f_session_high_so_far",
            "f_session_low_so_far",
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
        ],
    ),
    (
        "historical_intraday",
        [
            "f_historical_slot_return_mean",
            "f_historical_slot_return_std",
            "f_historical_slot_up_rate",
        ],
    ),
    (
        "microstructure_1m",
        [
            "f_1m_path_high",
            "f_1m_path_low",
            "f_1m_path_volume",
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
        ],
    ),
]


def _fold_metrics(dataset: pl.DataFrame, feature_columns: list[str], vcfg: dict, mcfg: dict):
    available = [c for c in feature_columns if c in dataset.columns]
    missing = [c for c in feature_columns if c not in dataset.columns]
    usable = dataset.drop_nulls(subset=available + ["label", "event_end_timestamp"]).sort(
        "timestamp"
    )
    if usable.is_empty():
        return None

    X = usable.select(available).to_numpy()
    y = usable["label"].to_numpy()
    timestamps = usable["timestamp"].dt.epoch("ns").to_numpy()
    event_end = usable["event_end_timestamp"].dt.epoch("ns").to_numpy()

    embargo_ns = int(vcfg["embargo_bars"]) * 5 * 60 * 1_000_000_000
    splitter = PurgedTimeSeriesSplit(int(vcfg["n_splits"]), embargo=embargo_ns)

    params = dict(mcfg)
    rounds = int(params.pop("num_boost_round", 1000))
    params.pop("early_stopping_rounds", None)

    rows = []
    for train_idx, val_idx in splitter.split(timestamps, event_end):
        if len(train_idx) < int(vcfg["min_train_rows"]):
            continue
        model = train_lightgbm_classifier(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            params=params,
            num_boost_round=rounds,
        )
        p = predict_proba(model, X[val_idx])
        for idx, row in zip(val_idx, p):
            rows.append(
                {"label": int(y[idx]), "p_short": row[0], "p_none": row[1], "p_long": row[2]}
            )

    if not rows:
        return None

    oof = pl.DataFrame(rows)
    y_oof = oof["label"].to_numpy()
    p_oof = oof.select(["p_short", "p_none", "p_long"]).to_numpy()
    pred = np.asarray([-1, 0, 1], dtype=np.int8)[np.argmax(p_oof, axis=1)]

    return {
        "n_features": len(available),
        "missing_features": missing,
        "n_rows_available": usable.height,
        "n_oof": oof.height,
        "accuracy": float(accuracy_score(y_oof, pred)),
        "log_loss": float(log_loss(y_oof, p_oof, labels=[-1, 0, 1])),
        "ece_long": expected_calibration_error((y_oof == 1).astype(int), p_oof[:, 2]),
        "ece_short": expected_calibration_error((y_oof == -1).astype(int), p_oof[:, 0]),
        "brier_long": brier_score((y_oof == 1).astype(int), p_oof[:, 2]),
        "brier_short": brier_score((y_oof == -1).astype(int), p_oof[:, 0]),
        "max_p_long": float(oof["p_long"].max()),
        "max_p_short": float(oof["p_short"].max()),
        "rows_p_long_ge_055": int(oof.filter(pl.col("p_long") >= 0.55).height),
        "rows_p_short_ge_055": int(oof.filter(pl.col("p_short") >= 0.55).height),
    }


def run():
    dataset = pl.read_parquet("data/gold/dataset_v1.parquet").sort("timestamp")
    vcfg = yaml.safe_load(Path("configs/validation/default.yaml").read_text())
    mcfg = yaml.safe_load(Path("configs/models/lightgbm.yaml").read_text())

    mlflow.set_experiment("BankNifty_Feature_Funnel")
    cumulative: list[str] = []
    results = []
    for stage_name, stage_columns in FUNNEL_STAGES:
        cumulative = cumulative + stage_columns
        with mlflow.start_run(run_name=f"funnel_{stage_name}"):
            lineage = Lineage.create(
                dataset_id="banknifty_5m_v1",
                feature_version=f"funnel_{stage_name}",
                label_version="triple_barrier_200_70_v1",
                validation_version="purged_time_series_v1",
            )
            log_lineage(lineage.to_dict())
            log_resolved_config({"model": mcfg, "validation": vcfg, "stage": stage_name})
            metrics = _fold_metrics(dataset, cumulative, vcfg, mcfg)
            if metrics is None:
                print(f"{stage_name}: no usable OOF folds, skipping")
                continue
            mlflow.log_params(
                {
                    "stage": stage_name,
                    "n_features": metrics["n_features"],
                    "n_rows_available": metrics["n_rows_available"],
                }
            )
            for key in (
                "accuracy",
                "log_loss",
                "ece_long",
                "ece_short",
                "brier_long",
                "brier_short",
                "max_p_long",
                "max_p_short",
                "rows_p_long_ge_055",
                "rows_p_short_ge_055",
            ):
                mlflow.log_metric(key, metrics[key])
            record = {"stage": stage_name, **metrics}
            results.append(record)
            print(
                f"{stage_name:22s} features={metrics['n_features']:3d} "
                f"rows={metrics['n_oof']:6d} log_loss={metrics['log_loss']:.4f} "
                f"max_p_long={metrics['max_p_long']:.3f} max_p_short={metrics['max_p_short']:.3f} "
                f"trades>=0.55={metrics['rows_p_long_ge_055'] + metrics['rows_p_short_ge_055']}"
            )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/feature_funnel.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    run()
