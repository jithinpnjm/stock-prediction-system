from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.research.move_reverse_engineering import analyze_moves


def run(threshold_points: float = 200.0):
    df5 = pl.read_parquet("data/silver/5m_canonical.parquet")
    moves = analyze_moves(df5, threshold_points=threshold_points)
    if moves.is_empty():
        print("no moves found")
        return

    Path("reports").mkdir(exist_ok=True)
    moves.write_parquet("reports/move_analysis.parquet")

    n = moves.height
    n_days = moves["date"].n_unique()
    zone_rate = moves.filter(pl.col("near_zone").is_not_null()).height / n
    compression_rate = moves["pre_move_compression"].mean()
    aligned_rate = moves["pre_move_aligned_with_move"].mean()
    up_rate = (moves["direction"] == "up").mean()

    summary = {
        "threshold_points": threshold_points,
        "n_moves": n,
        "n_distinct_days": n_days,
        "avg_moves_per_day": n / n_days,
        "up_fraction": up_rate,
        "avg_magnitude_points": float(moves["magnitude_points"].mean()),
        "avg_duration_bars": float(moves["duration_bars"].mean()),
        "started_near_a_zone_fraction": zone_rate,
        "zone_breakdown": moves.filter(pl.col("near_zone").is_not_null())["near_zone"]
        .value_counts()
        .sort("count", descending=True)
        .to_dicts(),
        "pre_move_was_compression_fraction": compression_rate,
        "pre_move_aligned_with_eventual_move_fraction": aligned_rate,
        "time_of_day_histogram": (
            moves.with_columns((pl.col("minutes_from_open") // 30 * 30).alias("bucket"))
            .group_by("bucket")
            .agg(pl.len().alias("count"))
            .sort("bucket")
            .to_dicts()
        ),
    }
    Path("reports/move_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    run()
