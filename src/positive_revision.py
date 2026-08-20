"""Post-feedback research: cost stress screening and opening-range breakouts.

All choices use pre-2020 statistics, but hypotheses were revised AFTER viewing
the earlier test. No output from this module is a new untouched holdout.
Original frozen portfolio and reports are deliberately not overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.universe_research import load_bars, load_specs, price_trades, PERIODS
from src.portfolio_engine import candidate_from_config, simulate, performance, DEFAULT_RISK
from src.project_report import markdown_table

OUT=Path('reports/positive_revision')
CONFIG=Path('configs/positive_revision.json')
RISK={**DEFAULT_RISK,'risk_fraction':.005,'per_asset_margin':.20,'total_margin':.40}
# Liquid markets chosen by market coverage, not by their 2020/21 PnL.
BREAKOUT_MARKETS=['AU','AL','CF','CU','RU','RB','M','SR','NI','SN','FG','OI']


def attach_risk(trades,bars,instrument):
    daily=bars.groupby('date').agg(close=('close','last'),symbol=('symbol','last'))
    returns=daily.close.pct_change(fill_method=None).where(daily.symbol.eq(daily.symbol.shift()))
    vol=returns.rolling(20,min_periods=10).std().shift()
    t=trades.copy();t['prior_vol']=t.date.map(vol)
    t['instrument']=instrument
    return t[t.prior_vol.notna()].copy()


def range_breakout(bars,spec,instrument,range_minutes=30,trend_days=0,buffer=0.,slips=1):
    """One/day, next-open fill, scheduled 15:00 exit (never a gap-defined exit).

    Require the first N minutes to start at 09:01 and be contiguous. Morning
    high/low are known before a breakout is observed. No stop/target ambiguity.
    Mixed-contract and missing scheduled exit exclusions remain ex-post CSV
    quality screens, explicitly reported, not assumed to be live-known.
    """
    b=bars[bars.start.dt.hour.ge(9)&bars.start.dt.hour.lt(16)].copy()
    g=b.groupby('date',sort=True)
    b['day_bar']=g.cumcount()
    n=range_minutes//5
    opening=b[b.day_bar.lt(n)].groupby('date').agg(
        range_high=('high','max'),range_low=('low','min'),range_start=('start','min'),
        range_end=('end','max'),range_bars=('start','size'))
    valid=(opening.range_start.dt.strftime('%H:%M').eq('09:01') & opening.range_bars.eq(n)
           & (opening.range_end-opening.range_start).eq(pd.Timedelta(minutes=range_minutes-1)))
    b['range_high']=b.date.map(opening.range_high)
    b['range_low']=b.date.map(opening.range_low)
    width=b.range_high-b.range_low
    b['direction']=np.where(b.close.gt(b.range_high+buffer*width),1,
                            np.where(b.close.lt(b.range_low-buffer*width),-1,0))
    previous=g.close.last();previous_symbol=g.symbol.last()
    logs=np.log(previous/previous.shift()).where(previous_symbol.eq(previous_symbol.shift()))
    trend=logs.rolling(trend_days,min_periods=max(2,int(.8*trend_days))).sum().shift() if trend_days else None
    b['signal_time']=b.end;b['signal_value']=b.direction*(b.close/b.range_high-1)
    b['entry_open']=g.open.shift(-1);b['entry_time']=g.start.shift(-1)
    exit_rows=b[b.end.dt.strftime('%H:%M').eq('15:00')].set_index('date')
    b['exit_close']=b.date.map(exit_rows.close) if len(exit_rows) else np.nan
    b['exit_time']=b.date.map(exit_rows.end) if len(exit_rows) else pd.NaT
    b['candidate_id']=f'{instrument}:ORB:{range_minutes}:{trend_days}:{buffer}'
    mask=(b.date.map(valid).fillna(False)&b.day_bar.ge(n)&b.direction.ne(0)
          &b.volume.ge(100)&b.open_interest.ge(1000)&b.entry_open.notna()&b.exit_close.notna()
          &(b.entry_time-b.signal_time).eq(pd.Timedelta(minutes=1))
          &b.entry_time.le(b.exit_time-pd.Timedelta(minutes=30))
          &b.date.map(g.symbol.nunique()).eq(1))
    if trend_days:mask &= b.direction.eq(np.sign(b.date.map(trend)))
    # First qualifying breakout only, in the direction known at that moment.
    t=b[mask].sort_values('signal_time').groupby('date',sort=False).head(1)
    t=attach_risk(t,bars,instrument)
    return price_trades(t,spec,slips)


def capital_stats(t,specs,period,slips=1):
    repriced=pd.concat([price_trades(g,specs[s],slips) for s,g in t.groupby('instrument')],ignore_index=True) if len(t) else t
    l,d,m=simulate(repriced,specs,*PERIODS[period],risk=RISK)
    a=l[l.accepted].copy() if len(l) else l
    years={}
    if len(a):
        a['year']=pd.to_datetime(a.entry_time).dt.year
        years=a.groupby('year').account_pnl_cny.sum().to_dict()
    return performance(l,d,m),years


def research():
    OUT.mkdir(parents=True,exist_ok=True);specs=load_specs();rows=[];choices=[]
    old=json.loads(Path('configs/final_portfolio.json').read_text())
    # Revision A: every original candidate is assessed, no test-PnL filter.
    for c in old['candidates']:
        t,b,_=candidate_from_config(c,specs[c['symbol']])
        train,years=capital_stats(t,specs,'train',3);val,_=capital_stats(t,specs,'validation',3)
        accepted=(train['total_return_pct']>0 and val['total_return_pct']>0
                  and sum(v>0 for v in years.values())>=2 and len(years)>=3
                  and val['trades']>=20)
        rows.append(dict(revision='cost_screen',symbol=c['symbol'],accepted=accepted,
                         train_stress_return_pct=train['total_return_pct'],
                         validation_stress_return_pct=val['total_return_pct'],
                         positive_train_years=sum(v>0 for v in years.values()),
                         validation_trades=val['trades']))
        if accepted:choices.append(dict(**c,revision='cost_screen'))
    # Revision B: small declared grid; ranking uses yearly TRAIN stability and
    # validation cash PnL after TWO ticks per side, rather than a single t-stat.
    breakout_rows=[];breakout_trades={}
    for instrument in BREAKOUT_MARKETS:
        path=next(Path('data').glob(instrument+'9999*'));bars=load_bars(path,5)
        for minutes in [30,60]:
            for trend_days in [0,20]:
                for buffer in [0.,.1]:
                    c=dict(symbol=instrument,family='range_breakout',timeframe=5,
                           range_minutes=minutes,trend_days=trend_days,buffer=buffer,
                           revision='range_breakout',candidate_id=f'{instrument}:ORB:{minutes}:{trend_days}:{buffer}')
                    t=range_breakout(bars,specs[instrument],instrument,minutes,trend_days,buffer)
                    train,years=capital_stats(t,specs,'train',2)
                    val,_=capital_stats(t,specs,'validation',2)
                    accept=(train['total_return_pct']>0 and val['total_return_pct']>0
                            and sum(v>0 for v in years.values())>=3
                            and train['trades']>=100 and val['trades']>=30)
                    score=min(train['sharpe'],val['sharpe'])
                    row=dict(**c,accepted=accept,score=score,
                             train_stress_return_pct=train['total_return_pct'],
                             validation_stress_return_pct=val['total_return_pct'],
                             positive_train_years=sum(v>0 for v in years.values()),
                             validation_trades=val['trades'])
                    breakout_rows.append(row)
                    breakout_trades[c['candidate_id']]=t
        print('ORB screened',instrument,flush=True)
    grid=pd.DataFrame(breakout_rows)
    valid=grid[grid.accepted].sort_values('score',ascending=False).drop_duplicates('symbol')
    for c in valid.to_dict('records'):
        choices.append({k:c[k] for k in ['symbol','family','timeframe','range_minutes','trend_days','buffer','revision','candidate_id']})
    pd.DataFrame(rows).to_csv(OUT/'cost_selection_audit.csv',index=False)
    grid.to_csv(OUT/'breakout_registry.csv',index=False)
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                retrospective=True,earlier_test_seen=True,test_pnl_filter=False,
                cost_screen='original candidates; positive train+2019 with 3 ticks per side; '
                            '>=2 positive train years and >=3 available years; >=20 validation trades',
                breakout_screen='12 markets x 2 ranges x 2 trend filters x 2 buffers; '
                                'positive train+2019 with 2 ticks; >=3 positive train years; >=100/30 trades',
                risk=RISK,candidates=choices)
    CONFIG.write_text(json.dumps(config,indent=2,ensure_ascii=False))
    print('REVISION CONFIG FROZEN',config['candidates'],flush=True)
    return config


def build_trades(config,specs):
    groups={};minutes={};coverage=[]
    for c in config['candidates']:
        s=c['symbol']
        if c['family']=='range_breakout':
            path=next(Path('data').glob(s+'9999*'));b=load_bars(path,5)
            t=range_breakout(b,specs[s],s,c['range_minutes'],c['trend_days'],c['buffer'])
        else:t,b,_=candidate_from_config(c,specs[s])
        groups.setdefault(c['revision'],[]).append(t)
        if s not in minutes:
            path=next(Path('data').glob(s+'9999*'))
            raw=pd.read_csv(path,usecols=['date','close','symbol','volume'],parse_dates=['date'])
            minutes[s]=raw.drop_duplicates('date').sort_values('date')
            coverage.append(dict(symbol=s,path=str(path),last=b.end.max(),sha256=digest(path)))
    return {k:pd.concat(v,ignore_index=True) for k,v in groups.items()},minutes,coverage


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def evaluate():
    config=json.loads(CONFIG.read_text());specs=load_specs()
    groups,minute,coverage=build_trades(config,specs)
    if not groups:raise ValueError('No candidates passed pre-2020 selection; do not relax based on test')
    end=min(pd.Timestamp(x['last']) for x in coverage)
    periods={**PERIODS,'2020':('2020-01-01','2020-12-31 23:59:59'),
             '2021_partial':('2021-01-01',str(end))}
    summary=[];curves={};audits=[]
    for book,trades in groups.items():
        for period,(start,finish) in periods.items():
            finish=min(pd.Timestamp(finish),end)
            l,d,m=simulate(trades,specs,start,finish,risk=config['risk'],minute_frames=minute)
            result=dict(book=book,period=period,**performance(l,d,m))
            summary.append(result);curves[book,period]=(d,m)
            l.to_csv(OUT/f'trades_{book}_{period}.csv',index=False)
            d.to_csv(OUT/f'equity_{book}_{period}.csv')
            m.to_csv(OUT/f'marks_{book}_{period}.csv',index=False)
            print(result,flush=True)
        for slips in [2,3]:
            repriced=pd.concat([price_trades(t,specs[s],slips) for s,t in trades.groupby('instrument')],ignore_index=True)
            for period in ['validation','2020','test']:
                start,finish=periods[period];finish=min(pd.Timestamp(finish),end)
                l,d,m=simulate(repriced,specs,start,finish,risk=config['risk'],minute_frames=minute)
                audits.append(dict(book=book,period=period,slippage_ticks_per_side=slips,**performance(l,d,m)))
    result=pd.DataFrame(summary);result.to_csv(OUT/'performance.csv',index=False)
    sensitivity=pd.DataFrame(audits);sensitivity.to_csv(OUT/'cost_stress.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for i,period in enumerate(['train','validation','test']):
        for book in groups:
            d,m=curves[book,period];eq=np.r_[config['risk']['initial_capital'],m.equity.to_numpy()]
            axes[i,0].plot(d.index,d.equity/1e6,label=book)
            axes[i,1].plot(m.timestamp,(eq/np.maximum.accumulate(eq)-1)[1:]*100,label=book)
        for ax in axes[i]:ax.set_title(period);ax.grid(alpha=.2);ax.legend()
        axes[i,0].set_ylabel('Equity, million CNY');axes[i,1].set_ylabel('Minute-close drawdown, %')
    fig.tight_layout();fig.savefig(OUT/'equity_drawdown.png',dpi=140);plt.close(fig)
    cols=['book','period','annualized_return_pct','sharpe','max_drawdown_pct','trades','win_rate_pct']
    text='''# Попытка получить положительный результат: revision 2

Это **ретроспективная переработка после просмотра прежнего test**, не новая
независимая проверка. Отбор функций не использует 2020/2021 PnL, но выбор новых
критериев после отрицательного test сам по себе создаёт selection bias.
Прежний портфель, конфигурация и отрицательный результат сохранены отдельно.

Цель этой итерации — положительная доходность после расходов, а не заявление о
выполнении исходных 20%/Sharpe 1,5. Капитал 3 млн CNY. Risk budget 0,5%, маржа
20% notional (сценарий), лимиты 20% на инструмент/40% суммарно, лоты не больше
1% известного сигнального объёма/OI. Комиссии Excel 2026 буквальные, baseline
1 tick НА СТОРОНУ; отсутствие quote-level fill/исторических тарифов сохраняется.

## Правила

**cost_screen**: каждое правило прежней конфигурации проверяется на train и 2019
при 3 ticks на сторону. Требуются положительный PnL обоих периодов, минимум
2 прибыльных train года из хотя бы 3 доступных и 20 validation сделок. Такая
проверка не означает, что последующие годы выдержат те же затраты.

**range_breakout**: двусторонний пробой диапазона high/low первых 30/60 минут
дневной сессии, при необходимости с буфером 10% ширины диапазона и фильтром
направления прошлой 20-дневной доходности. Первый пробой по close, вход по
open следующего 5-минутного бара, выход в **заранее заданные 15:00**, одна
сделка на рынок в день. Отбор на train/2019 при 2 ticks: прибыль обоих периодов,
3 прибыльных train года, минимум 100/30 сделок. 96 конфигураций, все в журнале.
Первый диапазон начинается 09:01 и обязан быть непрерывным. Mixed-symbol дни
и отсутствие данных для выхода — ex-post data-quality ограничения, не live
фильтры. Размер по прошлой волатильности не является стоп-лоссом.

## Фактические результаты

'''+markdown_table(result[cols])+'''\n
Каждый период начинает отдельный счёт. 2021 неполный; test заканчивается
2021-10-18. DD по minute-close, не по внутриминутным экстремумам. Sharpe по
дневным доходностям счёта, нулевые дни включены, ставка 0, календарь приближённый.

## Затраты выше baseline

'''+markdown_table(sensitivity[['book','period','slippage_ticks_per_side','annualized_return_pct','sharpe','max_drawdown_pct']])+'''\n
Плюс при одном tick не следует представлять как устойчивый плюс при любом
исполнении. Любые отрицательные строки сохраняются; стратегии не выбираются
по лучшему test. Для независимого подтверждения нужны новые данные/paper trading.

Запуск: `python -m src.positive_revision` (замороженная revision).
Повторить train/validation исследование: `python -m src.positive_revision --research`.
Все исходные результаты основной работы находятся в reports/final_portfolio/.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=config,
                  data=coverage,code_sha256={str(p):digest(p) for p in
                       [Path(__file__),Path('src/portfolio_engine.py'),Path('src/universe_research.py'),
                        Path('src/daily_universe_research.py'),CONFIG]})
    # Timestamp serialization for coverage; the data content is still SHA256'd.
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--research',action='store_true')
    args=p.parse_args()
    if args.research:research()
    evaluate()
