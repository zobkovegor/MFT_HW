import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_15min_bars(df):
    df = df.copy()
    df['group_15min'] = (
    df.groupby('session_block')['date']
      .transform(lambda x: ((x - x.min()).dt.total_seconds() // (15 * 60)).astype(int))
    )
    df_15_min = df.groupby(['session_block', 'group_15min']).agg(
        start=('date', 'min'),
        end=('date', 'max'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        money=('money','sum'),
        open_interest=('open_interest', 'last'),
        symbol=('symbol', 'first'),
        number_of_minutes=('date', 'count')
    )
    df_15_min = df_15_min[df_15_min['number_of_minutes'] == 15]
    df_15_min = df_15_min.reset_index()
    return df_15_min