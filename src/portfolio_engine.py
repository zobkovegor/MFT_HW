"""Integer-lot multi-asset accounting with explicit liquidity and margin scenarios.

Known signal-bar volume/OI constrain sizing; future returns never enter sizing.
Close marks are not executable quotes. Intraminute extremes, settlement/broker
margin calls, queues and market impact cannot be recovered from these OHLCV.
"""
from heapq import heappop, heappush
from pathlib import Path
import numpy as np
import pandas as pd
from src.universe_research import load_bars, load_specs, make_candidate, price_trades
from src.daily_universe_research import daily_candidates, seasonal_candidates

DEFAULT_RISK = dict(initial_capital=3_000_000., risk_fraction=.01,
                    per_asset_margin=.30, total_margin=.60,
                    volume_participation=.01, oi_participation=.01,
                    volatility_floor=.005, margin_rate=.20)


def candidate_from_config(config, spec, slippage_ticks=1):
    c=dict(config); symbol=c['symbol']
    path=next(Path('data').glob(symbol+'9999*.csv'))
    bars=load_bars(path,int(c['timeframe']))
    family=c['family']
    if family.startswith('daily_'):
        t,threshold=daily_candidates(bars,spec,int(c['lookback']),float(c['quantile']),
                                     family[6:],c['side'])
    elif family=='seasonal':
        t=seasonal_candidates(bars,spec,c['clock'],int(c['direction']));threshold=0.
    else:
        t,threshold=make_candidate(bars,spec,int(c['lookback']),int(c['holding']),
                                  float(c['quantile']),family,c['side'])
    if t.empty:return t,bars,threshold
    # Volatility at entry uses only completed previous calendar days.
    day=bars.groupby('date').agg(close=('close','last'),symbol=('symbol','last'))
    returns=day.close.pct_change(fill_method=None).where(day.symbol.eq(day.symbol.shift()))
    prior=returns.rolling(20,min_periods=10).std().shift()
    t=t.copy();t['prior_vol']=t.date.map(prior)
    t['instrument']=symbol
    t['candidate_id']=c.get('candidate_id',family)
    t=price_trades(t,spec,slippage_ticks)
    # data quality restriction: no sizing on missing history or stale signals.
    t=t[t.prior_vol.notna()&t.volume.gt(0)&t.open_interest.ge(100)].copy()
    return t,bars,threshold


def lot_count(row, spec, equity, used_margin, risk):
    capital=min(risk['initial_capital'],equity)
    if capital<=0:return 0
    notional=row.entry_fill*spec['multiplier']
    unit_margin=notional*risk['margin_rate']
    unit_entry_fee=notional*spec['open_fee_rate']+spec['open_fee_fixed']
    vol=max(row.prior_vol,risk['volatility_floor'])
    limits=[capital*risk['risk_fraction']/(notional*vol),
            capital*risk['per_asset_margin']/(unit_margin+unit_entry_fee),
            max(0,capital*risk['total_margin']-used_margin)/(unit_margin+unit_entry_fee),
            row.volume*risk['volume_participation'],
            row.open_interest*risk['oi_participation']]
    return max(0,int(np.floor(min(limits))))


def simulate(trades, specs, start, end, risk=None, minute_frames=None):
    """Release exits before entries; entry time must be later than signal time.

    minute_frames optional: dict of raw date/close/symbol frames. All marks are
    close-of-minute, never used for the earlier sizing decision. Fees deducted
    at their respective fill. Equal timestamps are consolidated before metrics.
    """
    risk={**DEFAULT_RISK,**(risk or {})}
    if trades.empty:
        trades=pd.DataFrame(columns=['entry_time','exit_time','instrument','candidate_id'])
        trades['entry_time']=pd.to_datetime(trades['entry_time'])
        trades['exit_time']=pd.to_datetime(trades['exit_time'])
    t=trades[trades.entry_time.between(start,end)&trades.exit_time.le(end)].copy()
    t=t.sort_values(['entry_time','instrument','candidate_id']).reset_index(drop=True)
    events=[]
    for i,r in t.iterrows():
        if r.entry_time<=r.signal_time:raise ValueError('Noncausal entry')
        if r.exit_time<r.entry_time:raise ValueError('Exit before entry')
        heappush(events,(r.entry_time.value,1,i,0.))
        heappush(events,(r.exit_time.value,0,i,0.))
    cash=float(risk['initial_capital']);active={};logs=[];marks=[];last_exit={};log_slots={}
    rows=list(t.itertuples(index=False))
    def state():
        pnl=0.;margin=0.
        for p in active.values():
            spec=specs[p['instrument']]
            pnl+=p['lots']*p['direction']*(p['mark']-p['entry_fill'])*spec['multiplier']
            margin+=p['lots']*p['mark']*spec['multiplier']*risk['margin_rate']
        return cash+pnl,margin
    def close_position(i,time,forced=False):
        nonlocal cash
        p=active.pop(i);r=rows[i];spec=specs[r.instrument];n=p['lots']
        slips=r.slippage_cny/(2*spec['tick']*spec['multiplier'])
        exit_close=p['mark'] if forced else r.exit_close
        fill=exit_close-r.direction*spec['tick']*slips if forced else r.exit_fill
        entry_fee=r.entry_fill*spec['multiplier']*spec['open_fee_rate']+spec['open_fee_fixed']
        exit_fee=fill*spec['multiplier']*spec['close_fee_rate']+spec['close_fee_fixed']
        unit_pnl=r.direction*(fill-r.entry_fill)*spec['multiplier']-entry_fee-exit_fee
        cash+=n*(unit_pnl+entry_fee)
        log=logs[log_slots[i]]
        log.update(account_pnl_cny=n*unit_pnl,net_pnl_cny=unit_pnl,
                   net_return=unit_pnl/(r.entry_open*spec['multiplier']),
                   gross_pnl_cny=r.direction*(exit_close-r.entry_open)*spec['multiplier'],
                   exit_time=pd.Timestamp(time),exit_close=exit_close,exit_fill=fill,
                   fees_cny=entry_fee+exit_fee,forced_margin_exit=forced)
    while events:
        time,kind,i,price=heappop(events);r=rows[i];spec=specs[r.instrument]
        if kind==0:
            if i in active:
                close_position(i,time)
        elif kind==1:
            equity,margin=state()
            overlapping=(any(p['instrument']==r.instrument for p in active.values())
                         or r.entry_time<=last_exit.get(r.instrument,pd.Timestamp.min))
            n=0 if overlapping else lot_count(r,spec,equity,margin,risk)
            log=r._asdict();log.update(lots=n,accepted=n>0,equity_at_entry=equity,
                                      skip_reason='overlap' if overlapping else ('size_zero' if n==0 else ''))
            log['account_pnl_cny']=0.;log['forced_margin_exit']=False
            log_slots[i]=len(logs);logs.append(log)
            if n:
                last_exit[r.instrument]=r.exit_time
                entry_fee=r.entry_fill*spec['multiplier']*spec['open_fee_rate']+spec['open_fee_fixed']
                cash-=n*entry_fee
                active[i]=dict(lots=n,instrument=r.instrument,direction=r.direction,
                               entry_fill=r.entry_fill,mark=r.entry_open,last_mark=time)
                if minute_frames is not None:
                    m=minute_frames[r.instrument]
                    left=m.date.searchsorted(r.entry_time,side='left')
                    right=m.date.searchsorted(r.exit_time,side='left')
                    path=m.iloc[left:right]
                    path=path[path.symbol.eq(r.symbol)]
                    for mr in path[['date','close']].itertuples(index=False):
                        heappush(events,(mr.date.value,2,i,float(mr.close)))
        else:
            if i in active:active[i].update(mark=price,last_mark=time)
        equity,margin=state()
        # Consolidate same-minute marks, then enforce the margin ceiling.
        # Liquidate only an asset with a current observable close. Stale quotes
        # cannot be used to invent an executable price during its lunch break.
        if not events or events[0][0]!=time:
            while active and margin>max(0,equity)*risk['total_margin']:
                available=[k for k,p in active.items() if p['last_mark']==time]
                if not available:break
                victim=max(available,key=lambda k:(active[k]['lots']*active[k]['mark']*
                            specs[active[k]['instrument']]['multiplier'],active[k]['instrument']))
                close_position(victim,time,forced=True)
                equity,margin=state()
        marks.append(dict(timestamp=pd.Timestamp(time),equity=equity,margin=margin,
                          margin_fraction=margin/equity if equity>0 else np.inf))
    if active:raise AssertionError('Unclosed positions')
    columns=['timestamp','equity','margin','margin_fraction']
    mark=pd.DataFrame(marks,columns=columns).drop_duplicates('timestamp',keep='last')
    if mark.empty:
        mark=pd.DataFrame([dict(timestamp=pd.Timestamp(start),equity=cash,margin=0.,margin_fraction=0.)])
    # Include all weekdays, not just signal dates. Zero-position days are zeros.
    calendar=pd.bdate_range(pd.Timestamp(start).normalize(),pd.Timestamp(end).normalize())
    closes=mark.assign(day=mark.timestamp.dt.normalize()).groupby('day').equity.last()
    # Include weekend night cash-flow days rather than discarding their PnL.
    all_days=calendar.union(closes.index).sort_values()
    daily=closes.reindex(all_days).ffill().fillna(risk['initial_capital']).to_frame('equity')
    daily['return']=daily.equity.pct_change(fill_method=None)
    daily.iloc[0,daily.columns.get_loc('return')]=daily.equity.iloc[0]/risk['initial_capital']-1
    daily.index.name='date'
    ledger=pd.DataFrame(logs)
    total=ledger.account_pnl_cny.sum() if len(ledger) else 0.
    if not np.isclose(cash,risk['initial_capital']+total,atol=1e-6):
        raise AssertionError('Cash ledger reconciliation failed')
    return ledger,daily,mark


def performance(ledger,daily,marks,initial_capital=3_000_000.):
    r=daily['return'];elapsed=(daily.index[-1]-daily.index[0]).days+1
    growth=daily.equity.iloc[-1]/initial_capital
    annual=growth**(365.25/elapsed)-1 if growth>0 else -1.
    std=r.std(ddof=1)
    # Annualization uses observations/year; calendar includes holidays as zeros.
    observations_per_year=len(r)*365.25/elapsed
    sharpe=r.mean()/std*np.sqrt(observations_per_year) if std>0 else np.nan
    eq=np.r_[initial_capital,marks.equity.to_numpy()]
    dd=eq/np.maximum.accumulate(eq)-1
    accepted=ledger[ledger.accepted] if len(ledger) else ledger
    win=(accepted.account_pnl_cny>0).mean() if len(accepted) else np.nan
    result=dict(days=len(r),elapsed_days=elapsed,final_equity_cny=daily.equity.iloc[-1],
                total_return_pct=(growth-1)*100,annualized_return_pct=annual*100,
                sharpe=sharpe,max_drawdown_pct=-dd.min()*100,
                trades=len(accepted),win_rate_pct=win*100,
                max_margin_pct=marks.margin_fraction.max()*100,
                margin_breaches=int(marks.margin_fraction.gt(.60).sum()),
                forced_margin_exits=int(accepted.forced_margin_exit.sum()) if len(accepted) else 0,
                drawdown_marks='minute_close' if len(marks)>len(ledger)*2 else 'fills_only')
    result['primary_pass']=bool(annual>.20 and sharpe>1.5)
    result['all_required_pass']=bool(result['primary_pass'] and result['max_drawdown_pct']<15)
    return result
