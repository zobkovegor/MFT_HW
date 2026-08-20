"""Final research engine: next-bar fills, explicit costs, constrained sizing and MTM.

The roll/completeness filter is an ex-post data-quality screen, not a live
roll-selection policy. Fee/margin history is a documented proxy; no broker
contract-level history, quotes, settlement prices or order book is available.
"""
from __future__ import annotations

from heapq import heappop, heappush
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import prepare_futures_data
from src.bar_builder import build_15min_bars

INSTRUMENTS = {
    "IF": {"contract_multiplier": 300, "tick_size": .2},
    "IC": {"contract_multiplier": 200, "tick_size": .2},
    "IH": {"contract_multiplier": 300, "tick_size": .2},
}
PERIODS = {
    "train": (pd.Timestamp("2015-01-01"), pd.Timestamp("2018-12-31 23:59:59")),
    "validation": (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31 23:59:59")),
    "test": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31 23:59:59")),
}

def period_for_date(value):
    return next((p for p, (s, e) in PERIODS.items() if s <= value <= e), None)

def build_setups(minute, bars, instrument):
    """Lag raw daily closes BEFORE screening; keep excluded dates for the audit."""
    minute = minute.copy()
    minute["date_day"] = minute["date"].dt.normalize()
    raw = minute.groupby("date_day").agg(
        raw_start=("date", "min"), raw_end=("date", "max"),
        minutes=("date", "size"), symbol_count=("symbol", "nunique"),
        day_close=("close", "last"), day_symbol=("symbol", "last"),
    )
    raw["prev_day_close"] = raw["day_close"].shift()
    raw["prev_day_symbol"] = raw["day_symbol"].shift()
    close_return = (raw["day_close"] / raw["prev_day_close"] - 1).where(
        raw["day_symbol"].eq(raw["prev_day_symbol"])
    )
    # At the opening of day t this estimate contains returns through t-1 only.
    raw["prior_daily_vol"] = close_return.rolling(20, min_periods=15).std().shift()
    bars = bars.sort_values("start").copy()
    bars["date"] = bars["end"].dt.normalize()
    bars["bar_number"] = bars.groupby("date").cumcount()
    first = bars[bars.bar_number.eq(0)].set_index("date")
    second = bars[bars.bar_number.eq(1)].set_index("date")
    last = bars.groupby("date").tail(1).set_index("date")
    setup = raw.copy()
    for field, frame, column in [
        ("first_close", first, "close"), ("first_symbol", first, "symbol"),
        ("first_bar_end", first, "end"), ("first_bar_start", first, "start"),
        ("entry_open", second, "open"), ("entry_symbol", second, "symbol"),
        ("entry_time", second, "start"), ("exit_close", last, "close"),
        ("exit_symbol", last, "symbol"), ("exit_time", last, "end"),
    ]:
        setup[field] = frame[column]
    setup["bars_in_day"] = bars.groupby("date").size().reindex(setup.index, fill_value=0)
    # Accept both observed session layouts without fitting the layout on returns.
    clock_start = setup["raw_start"].dt.strftime("%H:%M")
    clock_end = setup["raw_end"].dt.strftime("%H:%M")
    old_layout = (clock_start.eq("09:16") & clock_end.eq("15:15")
                  & setup.minutes.eq(270) & setup.bars_in_day.eq(18))
    new_layout = (clock_start.eq("09:31") & clock_end.eq("15:00")
                  & setup.minutes.eq(240) & setup.bars_in_day.eq(16))
    setup["complete_day"] = old_layout | new_layout
    setup["symbols_match"] = (
        setup.symbol_count.eq(1) & setup.first_symbol.eq(setup.entry_symbol)
        & setup.first_symbol.eq(setup.exit_symbol)
    )
    setup["same_contract_as_previous_day"] = setup.first_symbol.eq(setup.prev_day_symbol)
    setup["entry_is_next_minute"] = (
        setup.entry_time - setup.first_bar_end
    ).eq(pd.Timedelta(minutes=1))
    setup["valid_setup"] = (
        setup.complete_day & setup.symbols_match
        & setup.same_contract_as_previous_day & setup.entry_is_next_minute
        & setup.first_bar_start.eq(setup.raw_start)
        & setup.exit_time.eq(setup.raw_end) & setup.prev_day_close.gt(0)
    )
    setup["opening_move"] = setup.first_close / setup.prev_day_close - 1
    setup["opening_move_abs"] = setup.opening_move.abs()
    setup["gross_return"] = 1 - setup.exit_close / setup.entry_open
    setup["instrument"] = instrument
    setup["date"] = setup.index
    setup["period"] = setup["date"].map(period_for_date)
    return setup.reset_index(drop=True), bars

def fee_parameters(date, instrument, scenario="historical_proxy"):
    """Rates in bps (one bps=1e-4). Reconstructed history, not broker invoices."""
    date = pd.Timestamp(date)
    if scenario == "legacy_constant":
        return .23, 3.45, .12, 20
    if scenario == "workbook_per_mille":
        # Literal workbook: 0.23 per mille=2.3 bps; 3.45 per mille=34.5 bps.
        return 2.3, 34.5, .40, 20
    if scenario != "historical_proxy":
        raise ValueError(f"Unknown fee scenario: {scenario}")
    opening = .25 if date < pd.Timestamp("2015-08-03") else .23
    closing = opening
    for effective, value in [
        ("2015-08-26", 1.15), ("2015-09-07", 23.0),
        ("2017-02-17", 9.2), ("2017-09-18", 6.9),
        ("2018-12-03", 4.6), ("2019-04-22", 3.45),
    ]:
        if date >= pd.Timestamp(effective):
            closing = value
    # Deliberately conservative 40% throughout the early regime.
    margin = .40
    if date >= pd.Timestamp("2017-02-20"):
        margin = .30 if instrument == "IC" else .20
    if date >= pd.Timestamp("2017-09-19"):
        margin = .30 if instrument == "IC" else .15
    if date >= pd.Timestamp("2018-12-04"):
        margin = .15 if instrument == "IC" else .10
    if date >= pd.Timestamp("2019-04-23"):
        margin = .12 if instrument == "IC" else .10
    # Conservative opening count caps, per instrument per calendar trading day.
    cap = 20
    if pd.Timestamp("2015-09-07") <= date < pd.Timestamp("2017-02-17"):
        cap = 10
    return opening, closing, margin, cap

def add_trade_costs(trades, slippage_ticks=1, fee_scenario="historical_proxy"):
    """Slippage ticks are PER SIDE, on both entry and exit; two submitted orders."""
    result = trades.copy()
    specs = result.instrument.map(INSTRUMENTS)
    result["multiplier"] = specs.map(lambda s: s["contract_multiplier"]).astype(float)
    result["tick_size"] = specs.map(lambda s: s["tick_size"]).astype(float)
    parameters = [fee_parameters(r.date, r.instrument, fee_scenario)
                  for r in result.itertuples()]
    for i, name in enumerate(["open_fee_bps", "close_fee_bps", "margin_rate", "daily_cap"]):
        result[name] = [p[i] for p in parameters]
    result["entry_fill"] = result.entry_open - result.tick_size * slippage_ticks
    result["exit_fill"] = result.exit_close + result.tick_size * slippage_ticks
    result["gross_pnl_cny"] = (result.entry_open - result.exit_close) * result.multiplier
    result["entry_fee_cny"] = result.entry_fill * result.multiplier * result.open_fee_bps / 10000
    result["exit_fee_cny"] = result.exit_fill * result.multiplier * result.close_fee_bps / 10000
    # Proxy: 1 CNY per submitted order since 2015-08-03, two orders per trade.
    result["order_fee_cny"] = np.where(result.date.ge(pd.Timestamp("2015-08-03")), 2., 0.)
    result["slippage_cny"] = 2 * result.tick_size * result.multiplier * slippage_ticks
    result["total_cost_cny"] = (
        result.entry_fee_cny + result.exit_fee_cny + result.order_fee_cny + result.slippage_cny
    )
    result["net_pnl_cny"] = result.gross_pnl_cny - result.total_cost_cny
    result["net_return"] = result.net_pnl_cny / (result.entry_open * result.multiplier)
    result["required_margin_cny"] = result.entry_open * result.multiplier * result.margin_rate
    return result

def select_opening(setups, quantile=.8):
    frames, thresholds = [], []
    for instrument, frame in setups.groupby("instrument", sort=True):
        valid = frame[frame.valid_setup & frame.period.notna()].copy()
        train = valid[valid.period.eq("train")]
        if train.empty:
            raise ValueError(f"No eligible training data for {instrument}")
        threshold = train.opening_move_abs.quantile(quantile)
        thresholds.append(dict(instrument=instrument, strategy="opening",
                               quantile=quantile, threshold=threshold,
                               threshold_pct=threshold*100, train_observations=len(train)))
        selected = valid[valid.opening_move.lt(-threshold)].copy()
        selected["signal_time"] = selected.first_bar_end
        selected["strategy"] = "opening"
        selected["threshold"] = threshold
        frames.append(selected)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(thresholds)

def select_momentum(bars_by_instrument, setups, quantile=.8):
    """Comparable IF/IC/IH 60-minute momentum; next-bar open, four-bar holding."""
    frames, thresholds = [], []
    for instrument, source in bars_by_instrument.items():
        b = source.copy()
        groups = b.groupby("session_block", sort=False)
        b["past_move"] = b.close / groups.open.shift(3) - 1
        b["entry_open"] = groups.open.shift(-1)
        b["entry_time"] = groups.start.shift(-1)
        b["exit_close"] = groups.close.shift(-4)
        b["exit_time"] = groups.end.shift(-4)
        b["period"] = b["date"].map(period_for_date)
        b["signal_time"] = b.end
        b["instrument"] = instrument
        b = b.merge(setups[setups.instrument.eq(instrument)][
            ["date", "prior_daily_vol", "complete_day", "symbol_count"]
        ], on="date", how="left", validate="many_to_one")
        # Training distribution only uses observable past moves; no future return.
        eligible = b.complete_day & b.symbol_count.eq(1) & b.period.notna()
        train = b[eligible & b.period.eq("train") & b.past_move.notna()]
        threshold = train.past_move.abs().quantile(quantile)
        thresholds.append(dict(instrument=instrument, strategy="momentum",
                               quantile=quantile, threshold=threshold,
                               threshold_pct=threshold*100, train_observations=len(train)))
        signal = b[eligible & b.past_move.lt(-threshold) & b.exit_time.notna()
                   & (b.entry_time-b.signal_time).eq(pd.Timedelta(minutes=1))].copy()
        signal["strategy"] = "momentum"
        signal["threshold"] = threshold
        frames.append(signal)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(thresholds)

def summarize_trades(trades, pnl_column="net_pnl_cny"):
    values = trades.net_return.dropna()
    # A descriptive trade statistic; correlated same-day instruments are NOT IID.
    se = values.std() / np.sqrt(len(values)) if len(values)>1 else np.nan
    return pd.Series(dict(
        trades=len(values), mean_net_payoff_bps=values.mean()*10000,
        median_net_payoff_bps=values.median()*10000,
        win_rate_pct=values.gt(0).mean()*100,
        total_net_pnl_cny=trades[pnl_column].sum(),
        naive_trade_t_statistic=values.mean()/se if se>0 else np.nan,
    ))

def daily_equity(trades, calendar, initial_capital=3_000_000, pnl_column="portfolio_pnl_cny"):
    pnl = trades.groupby("date")[pnl_column].sum()
    daily = pd.DataFrame(index=pd.DatetimeIndex(calendar).sort_values().unique())
    daily.index.name = "date"
    daily["net_pnl_cny"] = pnl.reindex(daily.index, fill_value=0.)
    daily["equity_cny"] = initial_capital + daily.net_pnl_cny.cumsum()
    previous = daily.equity_cny.shift().fillna(initial_capital)
    daily["daily_return"] = daily.net_pnl_cny / previous
    daily["drawdown"] = daily.equity_cny / daily.equity_cny.cummax().clip(lower=initial_capital) - 1
    return daily

def equity_statistics(daily, trades, initial_capital):
    if daily.empty:
        raise ValueError("No observed trading days for equity")
    total = daily.equity_cny.iloc[-1] / initial_capital - 1
    elapsed_years = max(((daily.index[-1]-daily.index[0]).days+1)/365.25, 1/365.25)
    std = daily.daily_return.std()
    return pd.Series(dict(
        first_date=daily.index[0], last_date=daily.index[-1], trading_days=len(daily),
        start_equity_cny=initial_capital, end_equity_cny=daily.equity_cny.iloc[-1],
        total_return_pct=total*100,
        annualized_return_pct=((1+total)**(1/elapsed_years)-1)*100 if total>-1 else -100,
        annualized_sharpe=daily.daily_return.mean()/std*np.sqrt(252) if std>0 else np.nan,
        max_drawdown_pct=daily.drawdown.min()*100,
        trades=len(trades), win_rate_pct=trades.net_return.gt(0).mean()*100,
    ))

def simulate_portfolio(candidates, minute_by_instrument, calendar, initial_capital=3_000_000,
                       risk_fraction=.005, max_trade_margin_fraction=.20,
                       max_portfolio_margin_fraction=.60, max_contracts=20,
                       max_portfolio_risk_fraction=.015, fixed_contracts=None):
    """Event simulation. No pyramiding; simultaneous entries ordered by instrument.

    Estimated risk=prior daily sigma*entry notional; NOT a guaranteed loss or stop.
    Entry caps use min(initial capital,current marked equity). Marks use minute
    closes; margin is revalued with current price. Cap breaches after entry are
    audited, not concealed by assuming forced liquidation.
    """
    allocated = candidates.sort_values(["entry_time", "instrument", "strategy"]).reset_index(drop=True).copy()
    allocated["contracts"] = 0
    allocated["skip_reason"] = ""
    allocated["entry_equity_cny"] = np.nan
    allocated["estimated_risk_per_contract_cny"] = (
        allocated.prior_daily_vol.clip(lower=.001) * allocated.entry_open * allocated.multiplier
    )
    paths = {name: frame.set_index("date")[["close", "symbol"]].sort_index()
             for name, frame in minute_by_instrument.items()}
    event_heap = []
    # Heap types: exit=0, entry=1, mark=2. A same-instrument signal on exit is skipped.
    for i, row in allocated.iterrows():
        heappush(event_heap, (row.entry_time, 1, i, 0., 0.))
    active, used_per_day = {}, {}
    equity, history = float(initial_capital), []
    last_exit = {}
    while event_heap:
        time, kind, i, cumulative, marked_margin = heappop(event_heap)
        row = allocated.loc[i]
        if kind == 1:
            allocated.at[i, "entry_equity_cny"] = equity
            if row.instrument in active or time <= last_exit.get(row.instrument, pd.Timestamp.min):
                allocated.at[i, "skip_reason"] = "overlapping_instrument"
                continue
            risk = row.estimated_risk_per_contract_cny
            if fixed_contracts is None and (not np.isfinite(risk) or risk<=0):
                allocated.at[i, "skip_reason"] = "volatility_warmup"
                continue
            budget = max(0., min(initial_capital, equity))
            used_margin = sum(p["margin"] for p in active.values())
            used_risk = sum(p["risk"] for p in active.values())
            used_count = used_per_day.get((row.date, row.instrument), 0)
            entry_cost_per_contract = row.entry_fee_cny + row.slippage_cny/2 + row.order_fee_cny/2
            desired = fixed_contracts if fixed_contracts is not None else int(
                budget*risk_fraction/(risk+risk_fraction*entry_cost_per_contract))
            portfolio_risk_cap = max_contracts if fixed_contracts is not None else int(
                max(0., budget*max_portfolio_risk_fraction-used_risk)
                /(risk+max_portfolio_risk_fraction*entry_cost_per_contract))
            n = max(0, min(
                desired, max_contracts, portfolio_risk_cap,
                int(budget*max_trade_margin_fraction/(row.required_margin_cny
                    +max_trade_margin_fraction*entry_cost_per_contract)),
                int(max(0., budget*max_portfolio_margin_fraction-used_margin)
                    /(row.required_margin_cny+max_portfolio_margin_fraction*entry_cost_per_contract)),
                int(row.daily_cap)-used_count,
            ))
            if n == 0:
                allocated.at[i, "skip_reason"] = "risk_margin_or_daily_cap"
                continue
            path = paths[row.instrument].loc[row.entry_time:row.exit_time]
            if path.empty or path.index[-1] != row.exit_time or path.symbol.nunique()!=1:
                raise AssertionError("Invalid execution path")
            allocated.at[i, "contracts"] = n
            used_per_day[(row.date, row.instrument)] = used_count+n
            # The order levy is per submitted order, NOT per filled contract.
            # Allocation above reserves a conservative per-contract fee buffer.
            entry_cost = n*(row.entry_fee_cny+row.slippage_cny/2)+row.order_fee_cny/2
            equity -= entry_cost
            active[row.instrument] = dict(pnl=-entry_cost, margin=n*row.required_margin_cny,
                                          risk=n*risk if np.isfinite(risk) else 0., index=i)
            # Exit includes both side costs and must reconcile to trade net PnL.
            for mark_time, price in path.close.items():
                is_exit = mark_time == row.exit_time
                cumulative = n*(row.entry_open-price)*row.multiplier-entry_cost
                if is_exit:
                    cumulative = n*row.net_pnl_cny+(n-1)*row.order_fee_cny
                margin = n*price*row.multiplier*row.margin_rate
                heappush(event_heap, (mark_time, 0 if is_exit else 2, i, cumulative, margin))
        else:
            state = active[row.instrument]
            equity += cumulative - state["pnl"]
            state["pnl"], state["margin"] = cumulative, marked_margin
            if kind == 0:
                del active[row.instrument]
                last_exit[row.instrument] = time
        margin = sum(p["margin"] for p in active.values())
        history.append((time, equity, margin))
    allocated["portfolio_pnl_cny"] = (
        allocated.contracts*allocated.net_pnl_cny
        +(allocated.contracts-1).clip(lower=0)*allocated.order_fee_cny
    )
    allocated["one_contract_net_return"] = allocated.net_return
    has_position = allocated.contracts.gt(0)
    allocated.loc[has_position,"net_return"] = (
        allocated.loc[has_position,"portfolio_pnl_cny"]
        /(allocated.loc[has_position,"contracts"]*allocated.loc[has_position,"entry_open"]
          *allocated.loc[has_position,"multiplier"])
    )
    allocated["allocated_margin_cny"] = allocated.contracts * allocated.required_margin_cny
    accepted = allocated[allocated.contracts.gt(0)].copy()
    marks = pd.DataFrame(history, columns=["time", "equity_cny", "active_margin_cny"])
    if not marks.empty:
        # Simultaneous minute closes form one portfolio mark. Intermediate
        # per-instrument updates are bookkeeping, not observed price paths.
        marks = marks.groupby("time", as_index=False, sort=True).last()
        marks["drawdown"] = marks.equity_cny / marks.equity_cny.cummax().clip(lower=initial_capital)-1
        marks["margin_usage_pct"] = marks.active_margin_cny/marks.equity_cny*100
        assert np.isclose(equity-initial_capital, accepted.portfolio_pnl_cny.sum(), atol=1e-6)
    daily = daily_equity(accepted, calendar, initial_capital)
    summary = equity_statistics(daily, accepted, initial_capital)
    summary["minute_close_max_drawdown_pct"] = marks.drawdown.min()*100 if len(marks) else 0.
    summary["maximum_margin_usage_pct"] = marks.margin_usage_pct.max() if len(marks) else 0.
    summary["margin_cap_breach_events"] = int(marks.margin_usage_pct.gt(
        max_portfolio_margin_fraction*100+1e-8).sum()) if len(marks) else 0
    return allocated, daily, marks, summary

def run_opening_move_research(data_dir="data", signal_quantile=.80, slippage_ticks=1,
                             initial_capital=3_000_000, risk_fraction=.005,
                             max_trade_margin_fraction=.20, max_portfolio_margin_fraction=.60,
                             max_contracts=20, fee_scenario="historical_proxy",
                             include_comparison=True):
    if not 0<signal_quantile<1 or not 0<risk_fraction<=.1 or initial_capital<=0:
        raise ValueError("Invalid quantile, risk budget or capital")
    if slippage_ticks<0 or max_contracts<1:
        raise ValueError("Invalid slippage or contract cap")
    minutes, bars, frames = {}, {}, []
    coverage = []
    for instrument in INSTRUMENTS:
        path = Path(data_dir)/f"{instrument}9999.CCFX.csv"
        minute = prepare_futures_data(str(path))
        if not minute.date.is_monotonic_increasing or minute.date.duplicated().any():
            raise AssertionError("Timestamps must be unique and increasing")
        if (minute[["open","high","low","close"]]<=0).any().any():
            raise ValueError("Nonpositive prices")
        if minute[["date","open","high","low","close","volume","symbol"]].isna().any().any():
            raise ValueError("Missing market observations")
        inconsistent = (
            minute.high.lt(minute[["open","close"]].max(axis=1))
            | minute.low.gt(minute[["open","close"]].min(axis=1))
            | minute.low.gt(minute.high) | minute.volume.lt(0)
        )
        if inconsistent.any():
            raise ValueError(f"Invalid OHLC/volume for {instrument}")
        built = build_15min_bars(minute)
        setup, b = build_setups(minute, built, instrument)
        minutes[instrument], bars[instrument] = minute, b
        frames.append(setup)
        coverage.append(dict(instrument=instrument, rows=len(minute), bars=len(b),
                             first_date=minute.date.min(), last_date=minute.date.max()))
    setups = pd.concat(frames, ignore_index=True)
    candidates, thresholds = select_opening(setups, signal_quantile)
    trades = add_trade_costs(candidates, slippage_ticks, fee_scenario)
    calendar = pd.DatetimeIndex(sorted(setups.loc[setups.period.notna(), "date"].unique()))
    simulations, curves, marks, summary_rows = {}, {}, {}, {}
    for period, (start, end) in PERIODS.items():
        day_index = calendar[(calendar>=start)&(calendar<=end)]
        a, d, m, s = simulate_portfolio(
            trades[trades.period.eq(period)], minutes, day_index, initial_capital,
            risk_fraction, max_trade_margin_fraction, max_portfolio_margin_fraction, max_contracts)
        simulations[period], curves[period], marks[period], summary_rows[period] = a, d, m, s
    allocated = pd.concat(simulations.values(), ignore_index=True)
    accepted = allocated[allocated.contracts.gt(0)].copy()
    equity_summary = pd.DataFrame(summary_rows).T
    summary = lambda keys: accepted.groupby(keys).apply(
        lambda g: summarize_trades(g, "portfolio_pnl_cny"), include_groups=False)
    trade_summary = summary("period").reindex(PERIODS)
    instrument_summary = summary(["period","instrument"])
    yearly_summary = accepted.assign(year=accepted.date.dt.year).groupby(
        ["period","year"]).apply(lambda g: summarize_trades(g,"portfolio_pnl_cny"),include_groups=False)
    audit = setups[setups.period.notna()].groupby(["period","instrument"]).agg(
        observed_days=("date","size"), eligible_days=("valid_setup","sum"),
        mixed_contract_days=("symbol_count",lambda v:v.gt(1).sum()),
        incomplete_days=("complete_day",lambda v:(~v).sum()))
    audit["raw_signals"] = trades.groupby(["period","instrument"]).size()
    audit["accepted_trades"] = accepted.groupby(["period","instrument"]).size()
    audit = audit.fillna(0)
    robustness_rows = []
    for q in [.7,.8,.9]:
        selected, _ = select_opening(setups,q)
        for ticks in [1,2,3]:
            priced = add_trade_costs(selected,ticks,fee_scenario)
            for period in PERIODS:
                stat = summarize_trades(priced[priced.period.eq(period)])
                robustness_rows.append(dict(quantile=q,slippage_ticks=ticks,period=period,**stat))
    cost_rows = []
    for scenario in ["legacy_constant","historical_proxy","workbook_per_mille"]:
        priced = add_trade_costs(candidates,slippage_ticks,scenario)
        for period in PERIODS:
            cost_rows.append(dict(fee_scenario=scenario,period=period,
                                 **summarize_trades(priced[priced.period.eq(period)])))
    # Daily clustering incorporates cross-instrument contemporaneous dependence.
    rng = np.random.default_rng(42)
    confidence = []
    for period, daily in curves.items():
        values = daily.net_pnl_cny.to_numpy()
        n = len(values)
        boot_means = []
        # Circular 5-trading-day blocks, including zero-trade days.
        for _ in range(1000):
            starts = rng.integers(0,n,size=int(np.ceil(n/5)))
            sampled = values[((starts[:,None]+np.arange(5))%n).ravel()[:n]]
            boot_means.append(sampled.mean())
        lo, hi = np.quantile(boot_means,[.025,.975])
        confidence.append(dict(period=period,mean_daily_pnl_cny=values.mean(),
                               block_bootstrap_low_cny=lo,block_bootstrap_high_cny=hi))
    comparison_rows, comparison_curves, comparison_trades = [], {}, {}
    if include_comparison:
        momentum, momentum_thresholds = select_momentum(bars,setups,signal_quantile)
        momentum = add_trade_costs(momentum,slippage_ticks,fee_scenario)
        thresholds = pd.concat([thresholds,momentum_thresholds],ignore_index=True)
        for book, signals in [("opening",trades),("momentum_indices",momentum),
                              ("combined_indices",pd.concat([trades,momentum],ignore_index=True))]:
            for period,(start,end) in PERIODS.items():
                if book=="opening":
                    a,d,s = simulations[period],curves[period],summary_rows[period]
                else:
                    a,d,_,s = simulate_portfolio(
                        signals[signals.period.eq(period)],minutes,
                        calendar[(calendar>=start)&(calendar<=end)],initial_capital,
                        risk_fraction,max_trade_margin_fraction,max_portfolio_margin_fraction,max_contracts)
                comparison_rows.append(dict(strategy=book,period=period,**s))
                comparison_curves[(book,period)] = d
                comparison_trades[(book,period)] = a
    goal_check = equity_summary.copy()
    goal_check["return_above_20pct"] = goal_check.annualized_return_pct.astype(float)>20
    goal_check["sharpe_above_1_5"] = goal_check.annualized_sharpe.astype(float)>1.5
    goal_check["drawdown_below_15pct"] = goal_check.minute_close_max_drawdown_pct.astype(float).abs()<15
    goal_check["win_rate_above_50pct"] = goal_check.win_rate_pct.astype(float)>50
    goal_check["required_goals_pass"] = goal_check[
        ["return_above_20pct","sharpe_above_1_5","drawdown_below_15pct"]].all(axis=1)
    return dict(
        setups=setups, trades=trades, allocated_trades=allocated, accepted_trades=accepted,
        thresholds=thresholds, trade_summary=trade_summary, period_summary=trade_summary,
        instrument_period_summary=instrument_summary, yearly_summary=yearly_summary,
        equity_curves=curves,daily_curves=curves,minute_equity=marks,equity_summary=equity_summary,
        data_audit=audit,coverage=pd.DataFrame(coverage),robustness=pd.DataFrame(robustness_rows),
        cost_sensitivity=pd.DataFrame(cost_rows),confidence_intervals=pd.DataFrame(confidence),
        comparison=pd.DataFrame(comparison_rows),comparison_curves=comparison_curves,
        comparison_trades=comparison_trades,goal_check=goal_check,
        config=dict(signal_quantile=signal_quantile,slippage_ticks_per_side=slippage_ticks,
                    initial_capital=initial_capital,risk_fraction=risk_fraction,
                    max_trade_margin_fraction=max_trade_margin_fraction,
                    max_portfolio_margin_fraction=max_portfolio_margin_fraction,
                    max_contracts=max_contracts,fee_scenario=fee_scenario,
                    max_portfolio_risk_fraction=.015,volatility_window=20,
                    volatility_min_observations=15,volatility_floor=.001,
                    bootstrap_seed=42,bootstrap_repetitions=1000,bootstrap_block_days=5),
    )
