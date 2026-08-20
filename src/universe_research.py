"""Research a predeclared momentum/reversal grid on the supplied futures universe.

Signals are known at bar close; fill at next open. Search uses TRAIN only,
shortlisting uses 2019. No test-year results are used by the selection function.
Workbook fees are interpreted literally (per mille is never silently changed).
Constant commodity margins are stress assumptions, not historical exchange data.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Multiplier, minimum price tick. All contracts lacking metadata are excluded,
# not assigned a guessed multiplier or commission. Margins are a 20% scenario.
SPECS = {
 'FG':(20,1),'BU':(10,2),'I':(100,.5),'JD':(10,1),'RU':(10,5),
 'RB':(10,1),'HC':(10,1),'SR':(10,1),'TA':(5,2),'CF':(5,5),
 'MA':(10,1),'RM':(10,1),'OI':(10,1),'M':(10,1),'Y':(10,2),
 'P':(10,2),'A':(10,1),'B':(10,1),'C':(10,1),'CS':(10,1),
 'L':(5,5),'V':(5,5),'PP':(5,1),'J':(100,.5),'JM':(60,.5),
 'AG':(15,1),'AU':(1000,.02),'CU':(5,10),'AL':(5,5),'ZN':(5,5),
 'PB':(5,5),'NI':(1,10),'SN':(1,10),'SP':(10,2),'SS':(5,5),
 'AP':(10,1),'CJ':(5,5),'CY':(5,5),'SF':(5,2),'SM':(5,2),
 'SA':(20,1),'PF':(5,2),'UR':(20,1),'EG':(10,1),'EB':(5,1),
 'PG':(20,1),'SC':(1000,.1),'NR':(10,5),'LU':(10,1),'BC':(5,10),
 'LH':(16,5),'PK':(5,2),'RR':(10,1),'RS':(10,1),'RI':(20,1),
 'WH':(20,1),'PM':(50,1),'ZC':(100, .2),'FU':(10,1),
 'IF':(300,.2),'IH':(300,.2),'IC':(200,.2),
 'T':(10000,.005),'TF':(10000,.005),'TS':(20000,.005),
}
PERIODS={'train':('2015-01-01','2018-12-31 23:59:59'),
         'validation':('2019-01-01','2019-12-31 23:59:59'),
         'test':('2020-01-01','2021-12-31 23:59:59')}

def parse_fee(value):
    value=str(value).strip()
    number=float(re.search(r'[-+]?\d*\.?\d+',value).group())
    if '‰' in value: return number/1000,0.
    if 'CNY' in value: return 0.,number
    raise ValueError(f'Unrecognized fee: {value}')

def load_specs():
    book=pd.read_excel('Futures Commission Schedule (All Exchanges).xlsx')
    result={}
    for row in book.to_dict('records'):
        symbol=row['Symbol']
        if symbol not in SPECS:continue
        multiplier,tick=SPECS[symbol]
        op_rate,op_fixed=parse_fee(row['Open (per lot)'])
        cl_rate,cl_fixed=parse_fee(row['Close Today (per lot)'])
        result[symbol]=dict(multiplier=multiplier,tick=tick,margin_rate=.20,
                            open_fee_rate=op_rate,open_fee_fixed=op_fixed,
                            close_fee_rate=cl_rate,close_fee_fixed=cl_fixed)
    return result

def load_bars(path, timeframe=5, cache_dir='artifacts/bars'):
    """Full observed continuous sessions, including night; no cross-symbol bars."""
    path=Path(path); cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    stamp=f'{path.stat().st_size}_{path.stat().st_mtime_ns}'
    target=cache/f'{path.stem}_{timeframe}_{stamp}_v1.pkl'
    if target.exists():return pd.read_pickle(target)
    m=pd.read_csv(path,parse_dates=['date'])
    duplicate_count=int(m.date.duplicated().sum())
    m=m.drop_duplicates('date',keep='first').sort_values('date').reset_index(drop=True)
    valid_pair=m.date.diff().eq(pd.Timedelta(minutes=1)) & m.symbol.eq(m.symbol.shift())
    m['session']=(~valid_pair).cumsum()
    first=m.groupby('session').date.transform('first')
    m['bar']=((m.date-first).dt.total_seconds()//(timeframe*60)).astype(int)
    b=m.groupby(['session','bar']).agg(
        start=('date','min'),end=('date','max'),open=('open','first'),
        high=('high','max'),low=('low','min'),close=('close','last'),
        volume=('volume','sum'),open_interest=('open_interest','last'),
        symbol=('symbol','first'),minutes=('date','size'))
    b=b[b.minutes.eq(timeframe)].reset_index()
    b.attrs['raw_rows']=len(m)+duplicate_count
    b.attrs['duplicate_timestamps']=duplicate_count
    b['date']=b.end.dt.normalize()
    # Every position is intraday within a continuous block. Calendar date
    # remains a cash-flow label here; holding cannot cross midnight.
    b['block']=b.groupby(['session','date'],sort=False).ngroup()
    b['period']=''
    for period,(start,end) in PERIODS.items():
        b.loc[b.end.between(start,end),'period']=period
    b.to_pickle(target)
    return b

def price_trades(frame, spec, slippage_ticks=1):
    t=frame.copy()
    direction=t.direction
    t['entry_fill']=t.entry_open+direction*spec['tick']*slippage_ticks
    t['exit_fill']=t.exit_close-direction*spec['tick']*slippage_ticks
    t['gross_pnl_cny']=direction*(t.exit_close-t.entry_open)*spec['multiplier']
    t['fees_cny']=(t.entry_fill*spec['multiplier']*spec['open_fee_rate']+spec['open_fee_fixed']
                   +t.exit_fill*spec['multiplier']*spec['close_fee_rate']+spec['close_fee_fixed'])
    t['slippage_cny']=2*spec['tick']*spec['multiplier']*slippage_ticks
    t['net_pnl_cny']=t.gross_pnl_cny-t.fees_cny-t.slippage_cny
    t['net_return']=t.net_pnl_cny/(t.entry_open*spec['multiplier'])
    return t

def make_candidate(bars, spec, lookback=1, holding=1, quantile=.85,
                   family='reversal', side='both', slippage_ticks=1):
    b=bars.copy();g=b.groupby('block',sort=False)
    signal=b.close/g.open.shift(lookback-1)-1
    eligible=b.volume.gt(0)&b.open_interest.ge(100)
    if family.startswith('opening_'):
        eligible &= g.cumcount().eq(0) & b.symbol.eq(b.symbol.shift())
        signal=b.close/b.close.shift()-1
    train=signal[b.period.eq('train') & eligible].dropna()
    if len(train)<1000:return pd.DataFrame(),np.nan
    threshold=train.abs().quantile(quantile)
    b['direction']=np.sign(signal)*(1 if 'momentum' in family else -1)
    selected=signal.abs().gt(threshold) & eligible
    if side=='long':selected &= b.direction.eq(1)
    if side=='short':selected &= b.direction.eq(-1)
    b['signal_time']=b.end
    b['entry_open']=g.open.shift(-1)
    b['entry_time']=g.start.shift(-1)
    b['exit_close']=g.close.shift(-holding) if holding else g.close.transform('last')
    b['exit_time']=g.end.shift(-holding) if holding else g.end.transform('last')
    b['signal_value']=signal
    t=b[selected & b.period.ne('') & b.exit_close.notna() & b.entry_time.notna()].copy()
    t=t[(t.entry_time-t.signal_time).eq(pd.Timedelta(minutes=1))]
    # Greedy non-overlap, ordered by timestamp; signals at exit are skipped.
    accepted=[];last_exit=pd.Timestamp.min
    for row in t[['entry_time','exit_time']].itertuples():
        if row.entry_time>last_exit:
            accepted.append(row.Index);last_exit=row.exit_time
    t=t.loc[accepted]
    if t.empty:return t,threshold
    return price_trades(t,spec,slippage_ticks),threshold

def candidate_statistics(trades, period):
    v=trades.loc[trades.period.eq(period)] if len(trades) else trades
    if len(v)==0:return dict(trades=0,mean_bps=np.nan,t_stat=np.nan,win_pct=np.nan)
    r=v.net_return;se=r.std()/np.sqrt(len(r))
    return dict(trades=len(r),mean_bps=r.mean()*10000,
                t_stat=r.mean()/se if se>0 else np.nan,win_pct=r.gt(0).mean()*100)

def scan(symbols=None,output_dir='reports/universe',timeframes=(5,15),
         lookbacks=(1,4),holdings=(1,4),quantiles=(.8,.9,.95),
         families=('reversal','momentum'),sides=('long','short')):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    specs=load_specs();paths=list(Path('data').glob('*.csv'))
    rows=[];inventory=[]
    for path in sorted(paths):
        symbol=path.name.split('9999')[0]
        if symbols is not None and symbol not in symbols:continue
        if symbol not in specs:
            inventory.append(dict(symbol=symbol,status='excluded_missing_metadata'));continue
        for timeframe in timeframes:
            bars=load_bars(path,timeframe)
            inventory.append(dict(symbol=symbol,timeframe=timeframe,status='included',
                                  first_date=bars.start.min(),last_date=bars.end.max(),
                                  bars=len(bars),**bars.attrs))
            for lookback in lookbacks:
                for holding in holdings:
                    for quantile in quantiles:
                        for family in families:
                            for side in sides:
                                trades,threshold=make_candidate(bars,specs[symbol],lookback,holding,
                                                               quantile,family,side)
                                train=candidate_statistics(trades,'train')
                                # All rows are logged; no test summaries in the search registry.
                                validation=candidate_statistics(trades,'validation')
                                rows.append(dict(symbol=symbol,timeframe=timeframe,lookback=lookback,
                                                 holding=holding,quantile=quantile,family=family,side=side,
                                                 threshold=threshold,
                                                 **{'train_'+k:v for k,v in train.items()},
                                                 **{'validation_'+k:v for k,v in validation.items()}))
        pd.DataFrame(rows).to_csv(out/'candidate_registry.csv',index=False)
        pd.DataFrame(inventory).to_csv(out/'data_inventory.csv',index=False)
        print(symbol,'completed',len(rows),'configurations',flush=True)
    frame=pd.DataFrame(rows)
    eligible=frame[(frame.train_trades>=100)&(frame.validation_trades>=30)
                   &(frame.train_mean_bps>0)&(frame.validation_mean_bps>0)].copy()
    # Require training profitability first. One configuration per instrument.
    eligible['selection_score']=np.minimum(eligible.train_t_stat,eligible.validation_t_stat)
    selected=eligible.sort_values('selection_score',ascending=False).drop_duplicates('symbol').head(12)
    selected.to_csv(out/'shortlist.csv',index=False)
    config=dict(timeframes=timeframes,lookbacks=lookbacks,holdings=holdings,quantiles=quantiles,
                families=families,sides=sides,slippage_ticks_per_side=1,
                selection='positive train/validation, minimum t-stat, one per instrument',
                test_used_for_selection=False,fee_source='literal supplied workbook (2026 snapshot)')
    (out/'search_config.json').write_text(json.dumps(config,indent=2))
    print(selected.to_string(index=False),flush=True)
    return frame,selected

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbols',nargs='*')
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    if args.quick:
        scan(args.symbols,output_dir='reports/universe_probe',timeframes=(5,),lookbacks=(1,),
             holdings=(1,),quantiles=(.8,.85,.9,.95,.99))
    else:scan(args.symbols,timeframes=(5,15,30),lookbacks=(1,4),
              holdings=(1,4,0),quantiles=(.8,.95))
