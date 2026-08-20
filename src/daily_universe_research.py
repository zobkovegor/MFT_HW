"""Independent lower-turnover day strategies and clock-based session baselines."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.universe_research import load_bars,load_specs,price_trades,candidate_statistics


def daily_candidates(bars,spec,lookback=5,quantile=.5,family='trend',side='both'):
    day=bars[bars.start.dt.hour.ge(9)&bars.start.dt.hour.lt(16)].copy()
    grouped=day.groupby('date',sort=True)
    first=grouped.head(1).set_index('date')
    second=grouped.nth(1).set_index('date')
    last=grouped.tail(1).set_index('date')
    d=first.copy()
    d['entry_open']=second.open;d['entry_time']=second.start
    d['exit_close']=last.close;d['exit_time']=last.end
    d['prev_close']=last.close.shift();d['prev_symbol']=last.symbol.shift()
    d['intraday_symbols']=grouped.symbol.nunique()
    r=np.log(last.close/last.close.shift()).where(last.symbol.eq(last.symbol.shift()))
    # t-1 and older only for trend; opening move is known at first-bar close.
    if family=='trend':
        signal=r.rolling(lookback,min_periods=max(2,int(.8*lookback))).sum().shift()
    elif family=='reversal':
        signal=-r.rolling(lookback,min_periods=max(2,int(.8*lookback))).sum().shift()
    else:
        signal=(d.close/d.prev_close-1)*(1 if family=='opening_momentum' else -1)
    known_valid=d.symbol.eq(d.prev_symbol)&d.volume.gt(0)&d.open_interest.ge(100)
    threshold=signal[d.period.eq('train')&known_valid].abs().quantile(quantile)
    selected=known_valid&signal.abs().gt(threshold)
    direction=np.sign(signal)
    if side=='long':selected&=direction.eq(1)
    if side=='short':selected&=direction.eq(-1)
    d['direction']=direction;d['signal_value']=signal;d['signal_time']=d.end
    # Mixed-contract dates are flagged as an ex-post data-quality restriction.
    d=d[selected&d.intraday_symbols.eq(1)&d.entry_open.notna()&d.exit_close.notna()]
    d['date']=d.index
    return price_trades(d.reset_index(drop=True),spec),threshold


def seasonal_candidates(bars,spec,clock,direction):
    g=bars.groupby('block',sort=False)
    first=g.cumcount().eq(0)
    b=bars.copy();b['entry_open']=g.open.shift(-1);b['entry_time']=g.start.shift(-1)
    b['exit_close']=g.close.transform('last');b['exit_time']=g.end.transform('last')
    b['signal_time']=b.end;b['direction']=direction;b['signal_value']=0.
    mask=first&b.start.dt.strftime('%H:%M').eq(clock)&b.volume.gt(0)&b.open_interest.ge(100)
    return price_trades(b[mask&b.entry_open.notna()&b.period.ne('')],spec)


def scan_daily():
    out=Path('reports/daily_universe');out.mkdir(parents=True,exist_ok=True)
    specs=load_specs();rows=[]
    for symbol,spec in sorted(specs.items()):
        paths=list(Path('data').glob(symbol+'9999*'))
        if not paths:continue
        bars=load_bars(paths[0],5)
        for family in ['trend','reversal','opening_momentum','opening_reversal']:
            for lookback in ([5,20,60] if family in ['trend','reversal'] else [1]):
                for q in [.5,.8,.95]:
                    for side in ['long','short','both']:
                        t,threshold=daily_candidates(bars,spec,lookback,q,family,side)
                        train=candidate_statistics(t,'train');v=candidate_statistics(t,'validation')
                        rows.append(dict(symbol=symbol,timeframe=5,family='daily_'+family,
                                         lookback=lookback,quantile=q,side=side,holding=0,threshold=threshold,
                                         **{'train_'+k:x for k,x in train.items()},
                                         **{'validation_'+k:x for k,x in v.items()}))
        for clock in ['09:01','09:16','09:31','10:31','13:01','13:31','21:01']:
            for direction in [-1,1]:
                t=seasonal_candidates(bars,spec,clock,direction)
                train=candidate_statistics(t,'train');v=candidate_statistics(t,'validation')
                rows.append(dict(symbol=symbol,timeframe=5,family='seasonal',clock=clock,
                                 direction=direction,holding=0,lookback=1,quantile=0,
                                 side='long' if direction==1 else 'short',threshold=0.,
                                 **{'train_'+k:x for k,x in train.items()},
                                 **{'validation_'+k:x for k,x in v.items()}))
        pd.DataFrame(rows).to_csv(out/'candidate_registry.csv',index=False)
        print(symbol,len(rows),flush=True)
    return pd.DataFrame(rows)


if __name__=='__main__':
    x=scan_daily()
    eligible=x[(x.train_trades>=100)&(x.validation_trades>=30)&
               (x.train_mean_bps>0)&(x.validation_mean_bps>0)].copy()
    eligible['selection_score']=eligible[['train_t_stat','validation_t_stat']].min(axis=1)
    shortlist=eligible.sort_values('selection_score',ascending=False).drop_duplicates('symbol').head(12)
    shortlist.to_csv('reports/daily_universe/shortlist.csv',index=False)
    print(shortlist.to_string(index=False))
