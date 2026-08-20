"""Causal shadow-PnL guard on fixed rules, researched after earlier test was seen.

Only shadow trades whose exit is STRICTLY before the signal enter the trailing
mean. Blocked trades remain in shadow history, avoiding irreversible lockout.
The 20-trade/10-observation parameters are fixed, not optimized on evaluation.
"""
from collections import deque
from heapq import heappop, heappush
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.portfolio_engine import candidate_from_config, simulate, performance
from src.universe_research import load_specs, price_trades, PERIODS
from src.positive_revision import RISK,digest
from src.project_report import markdown_table
from src.final_project import confidence

OUT=Path('reports/online_guard')


def guard(trades,window=20,min_observations=10):
    """Filter at signal time; hypothetical future labels are released at exit.

    DataFrame can contain future net_return for accounting, but it is never
    queried for eligibility before that row's exit timestamp has passed.
    """
    if window<min_observations or min_observations<1:raise ValueError('Invalid history lengths')
    t=trades.sort_values(['signal_time','instrument','candidate_id']).reset_index(drop=True).copy()
    pending=[];history={};allowed=[];edges=[];counts=[]
    for i,row in enumerate(t.itertuples(index=False)):
        key=(row.instrument,row.candidate_id)
        while pending and pending[0][0]<row.signal_time.value:
            _,j,pastkey=heappop(pending)
            history.setdefault(pastkey,deque(maxlen=window)).append(float(t.net_return.iloc[j]))
        h=history.get(key,[]);mean=float(np.mean(h)) if len(h) else np.nan
        allow=len(h)>=min_observations and mean>0
        allowed.append(allow);edges.append(mean);counts.append(len(h))
        if row.exit_time<row.signal_time:raise ValueError('Exit before signal')
        heappush(pending,(row.exit_time.value,i,key))
    t['guard_allowed']=allowed;t['prior_shadow_mean']=edges;t['prior_shadow_count']=counts
    return t[t.guard_allowed].copy(),t


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path('configs/final_portfolio.json').read_text())
    reduced=json.loads(Path('configs/positive_revision.json').read_text())
    specs=load_specs();minute={};book_trades={};coverage=[]
    selected={c['symbol']:c for c in original['candidates']}
    for s,c in selected.items():
        t,b,_=candidate_from_config(c,specs[s]);book_trades[s]=t
        path=next(Path('data').glob(s+'9999*'))
        m=pd.read_csv(path,usecols=['date','close','symbol','volume'],parse_dates=['date'])
        minute[s]=m.drop_duplicates('date').sort_values('date')
        coverage.append(dict(symbol=s,last=str(b.end.max()),file=str(path),sha256=digest(path)))
    # Freeze guard definition before computing any new book performance.
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),retrospective=True,
                earlier_test_seen=True,parameters_optimized_on_test=False,
                window=20,min_observations=10,threshold=0,risk=RISK,
                books={'original_guarded':[c['symbol'] for c in original['candidates']],
                       'cost_guarded':[c['symbol'] for c in reduced['candidates'] if c['revision']=='cost_screen']},
                rule='positive mean of last 20 completed shadow net returns; >=10 observations; exit < signal')
    Path('configs/online_guard.json').write_text(json.dumps(config,indent=2))
    end=min(pd.Timestamp(c['last']) for c in coverage)
    periods={**PERIODS,'2020':('2020-01-01','2020-12-31 23:59:59'),
             '2021_partial':('2021-01-01',str(end))}
    stats=[];sensitivity=[];curves={};intervals=[]
    for book,symbols in config['books'].items():
        # Includes pre-2015 observations where the original rule's train-fitted
        # threshold is available; this is an explicitly retrospective warm-up.
        all_trades=pd.concat([book_trades[s] for s in symbols],ignore_index=True)
        filtered,audit=guard(all_trades,config['window'],config['min_observations'])
        audit.to_csv(OUT/f'guard_audit_{book}.csv',index=False)
        for p,(start,finish) in periods.items():
            finish=min(pd.Timestamp(finish),end)
            l,d,m=simulate(filtered,specs,start,finish,risk=RISK,minute_frames=minute)
            z=dict(book=book,period=p,**performance(l,d,m));stats.append(z)
            curves[book,p]=(d,m)
            intervals.append(dict(book=book,period=p,**confidence(d)))
            l.to_csv(OUT/f'trades_{book}_{p}.csv',index=False)
            d.to_csv(OUT/f'equity_{book}_{p}.csv')
            m.to_csv(OUT/f'marks_{book}_{p}.csv',index=False)
            print(z,flush=True)
        # Guard eligibility fixed from the one-tick estimated cost model;
        # stress changes actual fill costs, not selection after observing losses.
        for ticks in [2,3]:
            stress=pd.concat([price_trades(g,specs[s],ticks) for s,g in filtered.groupby('instrument')],ignore_index=True)
            for p in ['validation','2020','test']:
                start,finish=periods[p];finish=min(pd.Timestamp(finish),end)
                l,d,m=simulate(stress,specs,start,finish,risk=RISK,minute_frames=minute)
                sensitivity.append(dict(book=book,period=p,ticks_per_side=ticks,**performance(l,d,m)))
    result=pd.DataFrame(stats);result.to_csv(OUT/'performance.csv',index=False)
    stress=pd.DataFrame(sensitivity);stress.to_csv(OUT/'cost_stress.csv',index=False)
    ci=pd.DataFrame(intervals);ci.to_csv(OUT/'confidence.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for i,p in enumerate(['train','validation','test']):
        for book in config['books']:
            d,m=curves[book,p];eq=np.r_[RISK['initial_capital'],m.equity.to_numpy()]
            axes[i,0].plot(d.index,d.equity/1e6,label=book)
            axes[i,1].plot(m.timestamp,(eq/np.maximum.accumulate(eq)-1)[1:]*100,label=book)
        for ax in axes[i]:ax.set_title(p);ax.legend();ax.grid(alpha=.2)
        axes[i,0].set_ylabel('CNY million');axes[i,1].set_ylabel('Minute-close drawdown, %')
    fig.tight_layout();fig.savefig(OUT/'equity_drawdown.png',dpi=140);plt.close(fig)
    text='''# Положительный результат: контроль ухудшения сигналов

**Ретроспективная revision после просмотра прежнего test. Это не независимый holdout.**
Правило не использует будущий PnL, но гипотеза контроля выбрана после наблюдения
неудачи прежнего портфеля. Состав исходного и cost-screen портфелей не подбирался
по новой оценке; показываются оба результата и все периоды.

## Правило

До каждого сигнала проверяется средняя net-return последних **20 завершённых
виртуальных сделок** этого правила. Если наблюдений минимум 10 и среднее >0,
сигнал разрешён; иначе пропуск. Сделка входит в историю только если её exit_time
**строго раньше signal_time** текущего сигнала. Виртуальные сделки рассчитываются
даже когда реальные не разрешены: правило может включиться снова. Это оценка
при сценарном одном tick на сторону, а не информация о будущей доходности.

Капитал 3 млн CNY; риск-бюджет 0,5%, не стоп; 20% капитала на маржу инструмента,
40% суммарно при 20%-ной марже notional; лоты по прошлой волатильности и максимум
1% уже известного объёма/OI. Целые контракты, ни одной принудительной минимальной
позиции. Комиссии пользовательского Excel 2026 интерпретированы буквально;
baseline 1 tick **на каждой стороне**, не 1 tick round-trip. Ограничения
исторической маржи, continuous CSV, ex-post data-quality и отсутствия bid/ask
те же, что в первоначальном отчёте.

20/10 и threshold=0 зафиксированы до этой оценки, не выбирались по лучшему 2020.
Train с исходными train-fitted порогами остаётся in-sample. Данные до 2015
дают retrospective warm-up для доступных дневных правил; не следует объявлять
такой train исторической полностью online-торговлей.

## Доходность капитала

'''+markdown_table(result[['book','period','annualized_return_pct','sharpe','max_drawdown_pct','trades','win_rate_pct']])+'''\n
Для каждого периода отдельный стартовый счёт. Полный год — 2020, 2021 частичный
до 18 октября. CAGR, дневной Sharpe с нулевыми днями и minute-close DD; внутриминутная
просадка и историческое биржевое исполнение не известны. Проверять исходные цели
20% / 1,5 / 15% отдельно: просто положительная доходность не равна их выполнению.

## Стоимость исполнения

'''+markdown_table(stress[['book','period','ticks_per_side','annualized_return_pct','sharpe','max_drawdown_pct']])+'''\n
Eligibility guard остаётся по базовой модели издержек. Стресс ухудшает фактические
цены fill, а не переотбирает сделки задним числом.

## Интервалы, условные на выбранных правилах

'''+markdown_table(ci)+'''\n
Bootstrap: 5-дневные блоки, 1000 повторов, seed 42. Интервалы не учитывают весь
предыдущий перебор. Для подтверждения нужна новая не просмотренная история.

Запуск: `python -m src.online_guard`. Audit каждого shadow наблюдения и причин
пропуска, ledger, minute/daily curves и SHA256 сохранены рядом с отчётом.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    manifest=dict(config=config,data=coverage,code_sha256={str(p):digest(p) for p in
                  [Path(__file__),Path('src/positive_revision.py'),Path('src/portfolio_engine.py'),
                   Path('src/universe_research.py'),Path('src/daily_universe_research.py'),
                   Path('configs/online_guard.json'),Path('configs/positive_revision.json')]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return result


if __name__=='__main__':evaluate()
