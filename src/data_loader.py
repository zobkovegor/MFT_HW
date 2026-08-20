import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def load_futures_csv(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df


def clean_futures_data(df):
    df_copy = df.copy()
    df_copy = df_copy.drop_duplicates(subset=['date'], keep='first').sort_values('date')
    df_copy = df_copy.reset_index(drop=True)
    return df_copy


def add_clean_minute_returns(df):
    df_copy = df.copy()

    prev_symbol = df_copy["symbol"].shift(1)
    prev_close = df_copy["close"].shift(1)
    prev_date = df_copy["date"].shift(1)

    df_copy["is_valid_1m_pair"] = (
        prev_date.notna()
        & (df_copy["date"] - prev_date).eq(
            pd.Timedelta(minutes=1)
        )
        & df_copy["symbol"].eq(prev_symbol)
    )

    raw_return = df_copy["close"] / prev_close - 1

    df_copy["clean_return_1m"] = raw_return.where(
        df_copy["is_valid_1m_pair"]
    )

    return df_copy

def add_session_blocks(df):
    df_copy = df.copy()
    new_block = ~df_copy["is_valid_1m_pair"]
    df_copy["session_block"] = (
        new_block.cumsum() - 1
    )
    return df_copy


def prepare_futures_data(path):
    df = load_futures_csv(path)
    df = clean_futures_data(df)
    df = add_clean_minute_returns(df)
    df = add_session_blocks(df)
    return df