from __future__ import annotations

from dataclasses import replace

import polars as pl

from src.common.contracts import LabelConfig
from src.labels.triple_barrier import apply_triple_barrier_labels


def build_target_ladder(
    events: pl.DataFrame,
    one_minute: pl.DataFrame,
    *,
    targets: tuple[float, ...] = (100.0, 150.0, 200.0, 250.0, 300.0),
    stop_points: float = 70.0,
    horizon_bars: int = 75,
) -> dict[float, pl.DataFrame]:
    return {
        target: apply_triple_barrier_labels(
            events,
            one_minute,
            replace(
                LabelConfig(
                    target_points=target,
                    stop_points=stop_points,
                    horizon_bars=horizon_bars,
                )
            ),
        )
        for target in targets
    }
