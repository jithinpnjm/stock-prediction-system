from __future__ import annotations

import polars as pl

from .triple_barrier import apply_triple_barrier_labels


def build_target_ladder(
    events_5m:pl.DataFrame,
    bars_1m:pl.DataFrame,
    targets:tuple[float,...]=(200,250,300,350,400,500),
    stop_points:float=70,
    max_horizon_minutes:int=375,
)->pl.DataFrame:
    outputs=[]
    base_cols=["timestamp"]
    for target in targets:
        labeled=apply_triple_barrier_labels(
            events_5m,bars_1m,target_pts=target,stop_pts=stop_points,
            max_horizon_minutes=max_horizon_minutes
        ).select([
            *base_cols,
            "label","mfe_points","mae_points"
        ]).rename({
            "label":f"label_{int(target)}",
            "mfe_points":f"mfe_{int(target)}",
            "mae_points":f"mae_{int(target)}",
        })
        outputs.append(labeled)
    out=outputs[0]
    for item in outputs[1:]:
        out=out.join(item,on="timestamp",how="left")
    return out
