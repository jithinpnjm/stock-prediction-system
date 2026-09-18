import polars as pl


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds time-based features from the datetime column.
    """
    return df.with_columns(
        [
            pl.col("datetime").dt.hour().alias("f_hour"),
            pl.col("datetime").dt.minute().alias("f_minute"),
            pl.col("datetime").dt.weekday().alias("f_weekday"),
            # Minutes since session open (assuming 09:15 open)
            (
                (pl.col("datetime").dt.hour() - 9) * 60
                + pl.col("datetime").dt.minute()
                - 15
            ).alias("f_minutes_from_open"),
            # Minutes to session close (assuming 15:30 close)
            (
                (15 - pl.col("datetime").dt.hour()) * 60
                + 30
                - pl.col("datetime").dt.minute()
            ).alias("f_minutes_to_close"),
        ]
    ).with_columns(
        [
            # Ensure non-negative bounds in case of slight out-of-hours data
            pl.when(pl.col("f_minutes_from_open") < 0)
            .then(0)
            .otherwise(pl.col("f_minutes_from_open"))
            .alias("f_minutes_from_open"),
            pl.when(pl.col("f_minutes_to_close") < 0)
            .then(0)
            .otherwise(pl.col("f_minutes_to_close"))
            .alias("f_minutes_to_close"),
        ]
    )
