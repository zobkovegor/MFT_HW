import pandas as pd
import numpy as np

def add_high_volume_breakout_signal(bars, lookback=4):
    df_15_min = bars.copy()
    df_15_min["prior_high"] = (
        df_15_min
        .groupby("session_block")["high"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(window=lookback, min_periods=lookback)
                .max()
            )
        )
    )

    df_15_min["prior_low"] = (
        df_15_min
        .groupby("session_block")["low"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(window=lookback, min_periods=lookback)
                .min()
            )
        )
    )
    df_15_min["prior_mean_volume"] = (
        df_15_min
        .groupby("session_block")["volume"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(window=lookback, min_periods=lookback)
                .mean()
            )
        )
    )
    df_15_min['volume_ratio'] = df_15_min['volume'] / df_15_min['prior_mean_volume']
    df_15_min['breakout_signal'] = np.select(
        [
            df_15_min['close'] > df_15_min['prior_high'],
            df_15_min['close'] < df_15_min['prior_low'],
        ],
        [
            1,
            -1
        ],
        default=0
    )
    df_15_min["strategy_signal"] = np.where(
        df_15_min["volume_ratio"] > 1,
        df_15_min["breakout_signal"],
        0,
    )
    return df_15_min
