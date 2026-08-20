"""Research utilities for the index-futures opening-move continuation strategy.

The signal is deliberately simple and causal:

1. Measure the first 15-minute close relative to the previous trading day's close.
2. Estimate an instrument-specific threshold using training data only.
3. After an unusually negative opening move, enter short at the next bar's open.
4. Exit at the final daytime bar close.

Roll days are retained while lagged fields are built, then removed from the
tradable sample. This prevents a return from being calculated across contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.bar_builder import build_15min_bars
from src.data_loader import prepare_futures_data


@dataclass(frozen=True)
class InstrumentSpec:
    path: str
    multiplier: int
    margin_rate: float = 0.12
    tick_size: float = 0.2


INSTRUMENT_SPECS = {
    "IF": InstrumentSpec("data/IF9999.CCFX.csv", multiplier=300),
    "IC": InstrumentSpec("data/IC9999.CCFX.csv", multiplier=200),
    "IH": InstrumentSpec("data/IH9999.CCFX.csv", multiplier=300),
}

PERIODS = {
    "train": (pd.Timestamp("2015-01-01"), pd.Timestamp("2018-12-31 23:59:59")),
    "validation": (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31 23:59:59")),
    "test": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31 23:59:59")),
}


def _first_row(group: pd.DataFrame) -> pd.Series:
    return group.iloc[0]


def _second_row(group: pd.DataFrame) -> pd.Series:
    return group.iloc[1] if len(group) >= 2 else group.iloc[0] * np.nan


def _last_row(group: pd.DataFrame) -> pd.Series:
    return group.iloc[-1]


def build_opening_setups(path: str | Path, instrument: str) -> pd.DataFrame:
    """Create one causal opening setup per trading date for one instrument."""
    minute = prepare_futures_data(str(path))
    bars = build_15min_bars(minute).copy()
    bars = bars.sort_values(["start", "end"]).reset_index(drop=True)

    # CFFEX index futures have no night session. 15:15 occurred in the older
    # sample, while the later sample normally ends at 15:00.
    start_time = bars["start"].dt.time
    end_time = bars["end"].dt.time
    daytime = bars.loc[
        (start_time >= pd.Timestamp("09:00").time())
        & (end_time <= pd.Timestamp("15:15").time())
    ].copy()
    daytime["date"] = daytime["end"].dt.normalize()
    daytime = daytime.sort_values(["date", "start"]).reset_index(drop=True)
    daytime["bar_number"] = daytime.groupby("date").cumcount()

    grouped = daytime.groupby("date", sort=True)
    first = grouped.apply(_first_row, include_groups=False)
    second = grouped.apply(_second_row, include_groups=False)
    last = grouped.apply(_last_row, include_groups=False)

    setup = pd.DataFrame(index=first.index)
    setup["date"] = setup.index
    setup["instrument"] = instrument
    setup["bars_in_day"] = grouped.size()
    setup["first_close"] = first["close"]
    setup["first_symbol"] = first["symbol"]
    setup["entry_time"] = second["start"]
    setup["entry_price"] = second["open"]
    setup["entry_symbol"] = second["symbol"]
    setup["exit_time"] = last["end"]
    setup["exit_price"] = last["close"]
    setup["exit_symbol"] = last["symbol"]

    # Shift before removing roll dates. Therefore the day after a roll is
    # compared with the new contract's actual close on the roll date.
    setup["prev_day_close"] = setup["exit_price"].shift(1)
    setup["prev_day_symbol"] = setup["exit_symbol"].shift(1)
    setup["symbols_match_intraday"] = (
        setup["first_symbol"].eq(setup["entry_symbol"])
        & setup["first_symbol"].eq(setup["exit_symbol"])
    )
    setup["same_contract_as_previous_day"] = setup["first_symbol"].eq(
        setup["prev_day_symbol"]
    )
    setup["valid_setup"] = (
        setup["bars_in_day"].ge(2)
        & setup["symbols_match_intraday"]
        & setup["same_contract_as_previous_day"]
        & setup["prev_day_close"].notna()
    )
    setup["opening_move"] = setup["first_close"] / setup["prev_day_close"] - 1
    setup["gross_return"] = 1 - setup["exit_price"] / setup["entry_price"]
    setup["year"] = setup["date"].dt.year
    return setup.reset_index(drop=True)


def add_costs(
    trades: pd.DataFrame,
    spec: InstrumentSpec,
    slippage_ticks: float = 1.0,
    contracts: int = 1,
) -> pd.DataFrame:
    """Add CFFEX fees, round-trip slippage, margin and net PnL."""
    result = trades.copy()
    multiplier = spec.multiplier
    open_fee_rate = 0.23 / 10_000
    close_today_fee_rate = 3.45 / 10_000

    result["contracts"] = contracts
    result["gross_pnl_cny"] = (
        (result["entry_price"] - result["exit_price"])
        * multiplier
        * contracts
    )
    result["entry_fee_cny"] = (
        result["entry_price"] * multiplier * open_fee_rate * contracts
    )
    result["exit_fee_cny"] = (
        result["exit_price"] * multiplier * close_today_fee_rate * contracts
    )
    result["slippage_cny"] = (
        spec.tick_size * multiplier * slippage_ticks * contracts
    )
    result["total_cost_cny"] = (
        result["entry_fee_cny"]
        + result["exit_fee_cny"]
        + result["slippage_cny"]
    )
    result["net_pnl_cny"] = result["gross_pnl_cny"] - result["total_cost_cny"]
    result["notional_cny"] = result["entry_price"] * multiplier * contracts
    result["required_margin_cny"] = result["notional_cny"] * spec.margin_rate
    result["net_payoff"] = result["net_pnl_cny"] / result["notional_cny"]
    return result


def _summary(trades: pd.DataFrame) -> pd.Series:
    payoff = trades["net_payoff"].dropna()
    if payoff.empty:
        return pd.Series(
            {
                "trades": 0,
                "mean_net_payoff_bps": np.nan,
                "median_net_payoff_bps": np.nan,
                "win_rate_pct": np.nan,
                "standard_error_bps": np.nan,
                "t_statistic": np.nan,
                "total_net_pnl_cny": 0.0,
            }
        )
    standard_error = payoff.std(ddof=1) / np.sqrt(len(payoff))
    return pd.Series(
        {
            "trades": len(payoff),
            "mean_net_payoff_bps": payoff.mean() * 10_000,
            "median_net_payoff_bps": payoff.median() * 10_000,
            "win_rate_pct": (payoff > 0).mean() * 100,
            "standard_error_bps": standard_error * 10_000,
            "t_statistic": payoff.mean() / standard_error if standard_error > 0 else np.nan,
            "total_net_pnl_cny": trades.loc[payoff.index, "net_pnl_cny"].sum(),
        }
    )


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    start, end = PERIODS[period]
    return frame["date"].between(start, end)


def create_trades(
    setups_by_instrument: dict[str, pd.DataFrame],
    quantile: float = 0.80,
    slippage_ticks: float = 1.0,
    contracts: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-only thresholds and create train/validation/test trades."""
    threshold_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    for instrument, setup in setups_by_instrument.items():
        valid = setup.loc[setup["valid_setup"]].copy()
        train = valid.loc[_period_mask(valid, "train")]
        threshold = train["opening_move"].abs().quantile(quantile)
        threshold_rows.append(
            {
                "instrument": instrument,
                "quantile": quantile,
                "opening_move_threshold": threshold,
                "threshold_pct": threshold * 100,
                "train_valid_days": len(train),
            }
        )

        for period in PERIODS:
            sample = valid.loc[_period_mask(valid, period)].copy()
            # One-sided continuation rule: only unusually negative openings.
            sample = sample.loc[sample["opening_move"] < -threshold].copy()
            sample["period"] = period
            sample["signal"] = -1
            sample = add_costs(
                sample,
                INSTRUMENT_SPECS[instrument],
                slippage_ticks=slippage_ticks,
                contracts=contracts,
            )
            trade_frames.append(sample)

    thresholds = pd.DataFrame(threshold_rows)
    trades = pd.concat(trade_frames, ignore_index=True)
    trades = trades.sort_values(["date", "instrument"]).reset_index(drop=True)
    return thresholds, trades


def summarize_trades(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    instrument_period = (
        trades.groupby(["period", "instrument"], sort=False)
        .apply(_summary, include_groups=False)
        .reset_index()
    )
    combined_period = (
        trades.groupby("period", sort=False)
        .apply(_summary, include_groups=False)
        .reset_index()
    )
    yearly = (
        trades.groupby(["year", "instrument"], sort=True)
        .apply(_summary, include_groups=False)
        .reset_index()
    )
    yearly_combined = (
        trades.groupby("year", sort=True)
        .apply(_summary, include_groups=False)
        .reset_index()
    )
    return {
        "instrument_period": instrument_period,
        "combined_period": combined_period,
        "yearly_instrument": yearly,
        "yearly_combined": yearly_combined,
    }


def calculate_equity(
    trades: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    initial_capital: float = 3_000_000,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for period, (start, end) in PERIODS.items():
        dates = market_dates[(market_dates >= start.normalize()) & (market_dates <= end.normalize())]
        pnl = (
            trades.loc[trades["period"].eq(period)]
            .groupby("date")["net_pnl_cny"]
            .sum()
            .reindex(dates, fill_value=0.0)
        )
        daily = pd.DataFrame({"net_pnl_cny": pnl})
        daily["equity_cny"] = initial_capital + daily["net_pnl_cny"].cumsum()
        daily["daily_return"] = daily["net_pnl_cny"] / daily["equity_cny"].shift(1).fillna(initial_capital)
        daily["running_max"] = daily["equity_cny"].cummax()
        daily["drawdown"] = daily["equity_cny"] / daily["running_max"] - 1
        curves[period] = daily

        n_days = len(daily)
        total_return = daily["equity_cny"].iloc[-1] / initial_capital - 1 if n_days else np.nan
        annualized_return = (
            (1 + total_return) ** (252 / n_days) - 1
            if n_days and total_return > -1
            else np.nan
        )
        std = daily["daily_return"].std(ddof=1)
        sharpe = (
            daily["daily_return"].mean() / std * np.sqrt(252)
            if std and std > 0
            else np.nan
        )
        rows.append(
            {
                "period": period,
                "trading_days": n_days,
                "start_equity_cny": initial_capital,
                "end_equity_cny": daily["equity_cny"].iloc[-1] if n_days else initial_capital,
                "total_return_pct": total_return * 100,
                "annualized_return_pct": annualized_return * 100,
                "annualized_sharpe": sharpe,
                "max_drawdown_pct": daily["drawdown"].min() * 100 if n_days else np.nan,
                "profitable_days_pct": (daily["net_pnl_cny"] > 0).mean() * 100 if n_days else np.nan,
            }
        )
    return curves, pd.DataFrame(rows)


def robustness_grid(
    setups: dict[str, pd.DataFrame],
    quantiles: tuple[float, ...] = (0.70, 0.80, 0.90),
    slippage_ticks: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    rows: list[dict] = []
    for quantile in quantiles:
        for ticks in slippage_ticks:
            _, trades = create_trades(setups, quantile=quantile, slippage_ticks=ticks)
            for period in ("validation", "test"):
                stats = _summary(trades.loc[trades["period"].eq(period)])
                rows.append(
                    {
                        "quantile": quantile,
                        "slippage_ticks": ticks,
                        "period": period,
                        **stats.to_dict(),
                    }
                )
    return pd.DataFrame(rows)


def margin_summary(trades: pd.DataFrame, initial_capital: float = 3_000_000) -> pd.Series:
    events: list[tuple[pd.Timestamp, int, float]] = []
    for row in trades.itertuples():
        events.append((row.entry_time, 1, row.required_margin_cny))
        # Exits are processed before entries occurring at the same timestamp.
        events.append((row.exit_time, -1, -row.required_margin_cny))
    events.sort(key=lambda item: (item[0], item[1]))

    active_positions = 0
    active_margin = 0.0
    max_positions = 0
    max_margin = 0.0
    for _, direction, margin_change in events:
        active_positions += direction
        active_margin += margin_change
        max_positions = max(max_positions, active_positions)
        max_margin = max(max_margin, active_margin)
    return pd.Series(
        {
            "maximum_simultaneous_positions": max_positions,
            "maximum_required_margin_cny": max_margin,
            "maximum_margin_usage_pct": max_margin / initial_capital * 100,
            "maximum_margin_per_trade_cny": trades["required_margin_cny"].max(),
            "median_margin_per_trade_cny": trades["required_margin_cny"].median(),
        }
    )


def run_opening_move_research(
    base_dir: str | Path = ".",
    quantile: float = 0.80,
    slippage_ticks: float = 1.0,
    initial_capital: float = 3_000_000,
) -> dict:
    """Run the full reproducible experiment and return all result tables."""
    base_dir = Path(base_dir)
    setups = {
        instrument: build_opening_setups(base_dir / spec.path, instrument)
        for instrument, spec in INSTRUMENT_SPECS.items()
    }
    thresholds, trades = create_trades(
        setups,
        quantile=quantile,
        slippage_ticks=slippage_ticks,
    )
    summaries = summarize_trades(trades)
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(frame.loc[frame["valid_setup"], "date"] for frame in setups.values())))
    )
    curves, equity_summary = calculate_equity(
        trades,
        all_dates,
        initial_capital=initial_capital,
    )
    return {
        "setups": setups,
        "thresholds": thresholds,
        "trades": trades,
        **summaries,
        "equity_curves": curves,
        "equity_summary": equity_summary,
        "robustness": robustness_grid(setups),
        "margin_summary": margin_summary(trades, initial_capital=initial_capital),
    }
