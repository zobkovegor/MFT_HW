"""Reduce each cost-screen leg's risk by its worst pre-2020 stressed year.

Primary book: training-loss risk-weighted, maximum 0.5% risk budget, no leverage
increase. A single defensive comparison selects maximin TRAIN, not best test.
Everything is retrospective because the failed earlier test has been seen.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.universe_research import load_specs, price_trades, PERIODS
from src.portfolio_engine import candidate_from_config,simulate,performance
from src.positive_revision import RISK,capital_stats,digest
from src.final_project import confidence
from src.project_report import markdown_table

OUT=Path('reports/defensive_allocation')
CONFIG=Path('configs/defensive_allocation.json')


def training_weights(candidates,specs):
    rows=[]
    for c in candidates:
        t,_,_=candidate_from_config(c,specs[c['symbol']])
        _,years=capital_stats(t,specs,'train',3)
        worst=min(years.values())
        rows.append(dict(symbol=c['symbol'],worst_train_year_pnl_cny=worst,
                         loss_severity=max(-worst,1.),train_years=years))
    floor=min(r['loss_severity'] for r in rows)
    for r in rows:r['risk_weight']=float(floor/r['loss_severity'])
    return rows


def apply_weights(trades,weights):
    """Equivalent to reducing risk fraction only, without editing base engine.

    Budget / (notional * max(vol,floor) / weight) = weight * budget / risk.
    Actual observed vol is preserved separately; sizing prior_vol is explicitly
    a risk-penalized estimate, not falsely advertised as raw volatility.
    """
    t=trades.copy();t['observed_prior_vol']=t.prior_vol
    t['risk_weight']=t.instrument.map(weights)
    if t.risk_weight.isna().any() or not t.risk_weight.between(0,1,inclusive='right').all():
        raise ValueError('Risk weights must be in (0,1]')
    t['prior_vol']=t.observed_prior_vol.clip(lower=RISK['volatility_floor'])/t.risk_weight
    return t


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True);specs=load_specs()
    revision=json.loads(Path('configs/positive_revision.json').read_text())
    candidates=[c for c in revision['candidates'] if c['revision']=='cost_screen']
    if not candidates:raise ValueError('No cost-stable candidates')
    allocation=training_weights(candidates,specs)
    weights={r['symbol']:r['risk_weight'] for r in allocation}
    defensive=max(allocation,key=lambda r:(r['worst_train_year_pnl_cny'],r['symbol']))['symbol']
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),retrospective=True,
                earlier_test_seen=True,test_used_in_weight_function=False,
                primary_book='training_loss_weighted',comparison_book='maximin_single',
                single_symbol=defensive,risk=RISK,weights=weights,candidates=candidates,
                allocation='weight = smallest worst-year loss severity / asset worst-year loss severity; '
                           'train 2015-18 at 3 ticks per side; weights <= 1; no risk increase')
    CONFIG.write_text(json.dumps(config,indent=2))
    pd.DataFrame(allocation).drop(columns='train_years').to_csv(OUT/'training_allocation.csv',index=False)
    print('WEIGHTS FROZEN BEFORE THIS EVALUATION',weights,'maximin:',defensive,flush=True)
    minute={};frames=[];coverage=[]
    for c in candidates:
        s=c['symbol'];t,b,_=candidate_from_config(c,specs[s]);frames.append(t)
        path=next(Path('data').glob(s+'9999*'))
        m=pd.read_csv(path,usecols=['date','close','symbol','volume'],parse_dates=['date'])
        minute[s]=m.drop_duplicates('date').sort_values('date')
        coverage.append(dict(symbol=s,last=str(b.end.max()),file=str(path),sha256=digest(path)))
    all_trades=pd.concat(frames,ignore_index=True)
    books={'training_loss_weighted':apply_weights(all_trades,weights),
           'maximin_single':all_trades[all_trades.instrument.eq(defensive)].copy()}
    end=min(pd.Timestamp(c['last']) for c in coverage)
    periods={**PERIODS,'2020':('2020-01-01','2020-12-31 23:59:59'),
             '2021_partial':('2021-01-01',str(end))}
    rows=[];stresses=[];curves={};intervals=[]
    for book,t in books.items():
        for p,(start,finish) in periods.items():
            finish=min(pd.Timestamp(finish),end)
            l,d,m=simulate(t,specs,start,finish,risk=RISK,minute_frames=minute)
            z=dict(book=book,period=p,**performance(l,d,m));rows.append(z)
            intervals.append(dict(book=book,period=p,**confidence(d)))
            curves[book,p]=(d,m)
            l.to_csv(OUT/f'trades_{book}_{p}.csv',index=False)
            d.to_csv(OUT/f'equity_{book}_{p}.csv')
            m.to_csv(OUT/f'marks_{book}_{p}.csv',index=False)
            print(z,flush=True)
        for ticks in [2,3]:
            stress=pd.concat([price_trades(g,specs[s],ticks) for s,g in t.groupby('instrument')],ignore_index=True)
            for p in ['validation','2020','test']:
                start,finish=periods[p];finish=min(pd.Timestamp(finish),end)
                l,d,m=simulate(stress,specs,start,finish,risk=RISK,minute_frames=minute)
                stresses.append(dict(book=book,period=p,ticks_per_side=ticks,**performance(l,d,m)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'performance.csv',index=False)
    stress=pd.DataFrame(stresses);stress.to_csv(OUT/'cost_stress.csv',index=False)
    ci=pd.DataFrame(intervals);ci.to_csv(OUT/'confidence.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for i,p in enumerate(['train','validation','test']):
        for book in books:
            d,m=curves[book,p];eq=np.r_[RISK['initial_capital'],m.equity.to_numpy()]
            axes[i,0].plot(d.index,d.equity/1e6,label=book)
            axes[i,1].plot(m.timestamp,(eq/np.maximum.accumulate(eq)-1)[1:]*100,label=book)
        for ax in axes[i]:ax.set_title(p);ax.legend();ax.grid(alpha=.2)
        axes[i,0].set_ylabel('CNY million');axes[i,1].set_ylabel('Minute-close drawdown, %')
    fig.tight_layout();fig.savefig(OUT/'equity_drawdown.png',dpi=140);plt.close(fig)
    text='''# Защитное распределение риска: попытка сохранить положительный PnL

**Ретроспективное исследование после просмотра отрицательного test. Не новый holdout.**
Основной вариант — `training_loss_weighted`; одиночный `maximin_single` служит
сравнением и тоже выбирается только по train. Показаны все годы и издержки;
из таблиц не удаляются минусовые сценарии.

Из исходных 14 правил cost-screen оставил положительные на train и validation
при трёх ticks на сторону, с хотя бы двумя положительными train годами и 20
validation сделками. Затем риск каждого правила уменьшен пропорционально
величине худшего годового train-убытка при трёх ticks. Максимальный вес 1,
остальные <=1; ни один риск не увеличивается ради прибыли. Формула:
`weight_i = min_j(loss_severity_j) / loss_severity_i`,
`loss_severity_i = max(-worst_train_year_PnL_i, 1 CNY)`.
Для `maximin_single` выбирается инструмент с наибольшим минимальным годовым
train PnL, **не с наибольшим test PnL**. Тем не менее выбор самой новой политики
после уже просмотренного test создаёт research/selection bias.

## Распределение, вычисленное только на 2015–2018

'''+markdown_table(pd.DataFrame(allocation).drop(columns='train_years'))+'''\n
Базовый риск-бюджет 0,5% капитала, не стоп; вес умножает только этот бюджет.
Для точной совместимости с прежним engine сохраняется `observed_prior_vol`,
а в `prior_vol` для sizing записывается `max(observed_vol, 0.005) / weight`.
Это штраф оценки риска, не новая оценка наблюдаемой волатильности. Все веса
доступны до 2020. Маржа 20% notional — сценарий; лимиты 20% на инструмент/40%
суммарно, объем/OI известного бара 1%, целые лоты, без обязательного одного лота.

Правила, комиссия, время входа/выхода не менялись: NI short при отрицательной
прошлой 20-дневной сумме лог-доходностей, AU long при отрицательной прошлой
60-дневной сумме; пороги — train-квантиль 50% абсолютных сигналов. Вход на open
второго дневного 5-минутного бара, выход последнего дневного close, без переноса
через ночь. Комиссии Excel 2026 буквальные; baseline 1 tick НА КАЖДОЙ стороне.
Исторические комиссии/маржа не восстановлены, ex-post quality screens и
отсутствие bid/ask ограничивают интерпретацию результатов.

## Результаты капитала 3 млн CNY

'''+markdown_table(result[['book','period','total_return_pct','annualized_return_pct','sharpe','max_drawdown_pct','trades','win_rate_pct']])+'''\n
Отдельный капитал в начале каждого периода, их нельзя склеивать. 2020 полный;
2021 заканчивается 18 октября. CAGR за фактический период, Sharpe дневного
счёта с нулями и нулевой ставкой, DD на всех доступных minute-close marks.
Не заменяет внутриминутную просадку и биржевую модель исполнения.

## Проскальзывание выше baseline

'''+markdown_table(stress[['book','period','ticks_per_side','annualized_return_pct','sharpe','max_drawdown_pct']])+'''\n
Если плюс исчезает при более высоких издержках, его нельзя называть устойчивым.
Положительный baseline сам по себе не доказывает исходные цели >20% / >1,5 / <15%.

## Неопределённость

'''+markdown_table(ci)+'''\n
95% bootstrap: блок 5 дней, 1000 повторов, seed 42; условен на выбранных правилах
и не учитывает весь перебор. Новые данные или paper trading нужны для независимой
проверки. Все прошлые отрицательные результаты сохранены, включая online guard.

Воспроизвести: `python -m src.defensive_allocation`. Конфигурация и веса:
configs/defensive_allocation.json. Это вычисляет заданную train-формулу, не ищет
веса с наибольшей test доходностью. CSV, графики, ledger и SHA256 — рядом с отчётом.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=config,data=coverage,
                  code_sha256={str(p):digest(p) for p in [Path(__file__),Path('src/positive_revision.py'),
                       Path('src/portfolio_engine.py'),Path('src/universe_research.py'),
                       Path('src/daily_universe_research.py'),CONFIG,Path('configs/positive_revision.json')]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return result


if __name__=='__main__':evaluate()
