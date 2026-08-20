"""Final multi-asset experiment, frozen selection and automatically rendered report.

Search registries only contain 2015-18/2019 statistics. --freeze writes the
selection before evaluation. The evaluation path never modifies selection.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.universe_research import load_specs, PERIODS, price_trades
from src.portfolio_engine import candidate_from_config, simulate, performance, DEFAULT_RISK
from src.project_report import markdown_table

REGISTRIES=['reports/universe/candidate_registry.csv',
            'reports/opening_universe/candidate_registry.csv',
            'reports/daily_universe/candidate_registry.csv']
FREEZE=Path('configs/final_portfolio.json')
OUT=Path('reports/final_portfolio')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def freeze_selection():
    """One candidate/asset; positive training AND validation, then capacity audit.

    t statistics are ranking heuristics only, not adjusted hypothesis tests.
    Do not change this rule after reading 2020-21 in order to rescue the result.
    """
    specs=load_specs();r=pd.concat([pd.read_csv(p) for p in REGISTRIES],ignore_index=True)
    s=r[(r.train_trades>=100)&(r.validation_trades>=30)&
        (r.train_mean_bps>0)&(r.validation_mean_bps>0)].copy()
    s['score']=s[['train_t_stat','validation_t_stat']].min(axis=1)
    s=s[s.score>=.5].sort_values(['score','symbol'],ascending=[False,True]).drop_duplicates('symbol')
    selected=[];audit=[]
    for c in s.to_dict('records'):
        keys=['symbol','family','timeframe','lookback','holding','quantile','side','clock','direction']
        config={k:c[k] for k in keys if k in c and not pd.isna(c[k])}
        config['candidate_id']=f"{c['symbol']}:{c['family']}:{int(c['timeframe'])}"
        trades,_,threshold=candidate_from_config(config,specs[c['symbol']])
        stats={}
        for period in ['train','validation']:
            l,d,m=simulate(trades,specs,*PERIODS[period])
            stats[period]=performance(l,d,m)
        accept=(stats['train']['total_return_pct']>0 and stats['validation']['total_return_pct']>0
                and stats['validation']['trades']>=10)
        audit.append(dict(**config,score=c['score'],selected=accept,
                          train_return_pct=stats['train']['total_return_pct'],
                          validation_return_pct=stats['validation']['total_return_pct'],
                          validation_filled_trades=stats['validation']['trades']))
        if accept:
            config['train_threshold']=float(threshold);selected.append(config)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(OUT/'selection_audit.csv',index=False)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                selection_periods=['train','validation'],evaluation_used_for_selection=False,
                policy='positive mean both periods; >=100/30 signals; min t>=0.5; '
                       'one per asset; positive capital PnL both; >=10 filled validation trades',
                configurations_searched=len(r),registries_sha256={p:sha256(p) for p in REGISTRIES},
                risk=DEFAULT_RISK,slippage_ticks_per_side=1,
                fees='literal supplied 2026 workbook; NOT reconstructed historical fees',
                candidates=selected)
    FREEZE.parent.mkdir(exist_ok=True)
    FREEZE.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FROZEN BEFORE EVALUATION:',len(selected),'assets',flush=True)
    return result


def confidence(daily,seed=42,repetitions=1000,block=5):
    rng=np.random.default_rng(seed);r=daily['return'].to_numpy();n=len(r)
    draws=[];annualization=n*365.25/((daily.index[-1]-daily.index[0]).days+1)
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        sample=r[(starts[:,None]+np.arange(block))%n].reshape(-1)[:n]
        sharpe=sample.mean()/sample.std(ddof=1)*np.sqrt(annualization) if sample.std()>0 else np.nan
        growth=np.prod(1+sample)
        annual=growth**(annualization/n)-1 if growth>0 else -1
        draws.append((annual*100,sharpe))
    low,high=np.nanpercentile(draws,[2.5,97.5],axis=0)
    return dict(return_low_pct=low[0],return_high_pct=high[0],sharpe_low=low[1],sharpe_high=high[1])


def run():
    if not FREEZE.exists():raise FileNotFoundError('Run --freeze after the research scans first')
    config=json.loads(FREEZE.read_text());specs=load_specs();all_trades=[];minute={};coverage=[]
    for c in config['candidates']:
        symbol=c['symbol'];t,b,threshold=candidate_from_config(c,specs[symbol])
        if not np.isclose(threshold,c['train_threshold'],rtol=1e-10,atol=1e-12):
            raise ValueError(f'Training threshold changed for {symbol}; inspect changed inputs')
        all_trades.append(t)
        path=next(Path('data').glob(symbol+'9999*.csv'))
        raw=pd.read_csv(path,usecols=['date','close','symbol','volume'],parse_dates=['date'])
        raw=raw.drop_duplicates('date',keep='first').sort_values('date')
        minute[symbol]=raw
        coverage.append(dict(symbol=symbol,first=b.start.min(),last=b.end.max(),
                             raw_rows=b.attrs['raw_rows'],duplicates=b.attrs['duplicate_timestamps'],
                             data_file=str(path),sha256=sha256(path)))
    if not all_trades:raise ValueError('No candidates passed the frozen selection rule')
    trades=pd.concat(all_trades,ignore_index=True)
    common_end=min(pd.Timestamp(c['last']) for c in coverage)
    spans={**PERIODS,'2020':('2020-01-01','2020-12-31 23:59:59'),
           '2021_partial':('2021-01-01',str(common_end))}
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[];curves={};marks={};ledgers={};conf=[]
    for name,(start,end) in spans.items():
        end=min(pd.Timestamp(end),common_end)
        ledger,daily,mark=simulate(trades,specs,start,end,config['risk'],minute)
        summary=performance(ledger,daily,mark,config['risk']['initial_capital'])
        summary.update(period=name,start=start,end=end,
                       complete_calendar_year=name in ['validation','2020'])
        summaries.append(summary);curves[name]=daily;marks[name]=mark;ledgers[name]=ledger
        conf.append(dict(period=name,**confidence(daily)))
        ledger.to_csv(OUT/f'trades_{name}.csv',index=False)
        daily.to_csv(OUT/f'daily_equity_{name}.csv')
        mark.to_csv(OUT/f'minute_equity_{name}.csv',index=False)
        print(name,summary,flush=True)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'performance.csv',index=False)
    pd.DataFrame(conf).to_csv(OUT/'confidence.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    # Execution audit only: future volumes do NOT enter the sizing or signals.
    audits=[];indexed_minute={s:m.set_index('date') for s,m in minute.items()}
    for name,ledger in ledgers.items():
        if name not in ['train','validation','test']:continue
        for row in ledger[ledger.accepted].itertuples():
            raw=indexed_minute[row.instrument]
            ev=float(raw.volume.get(row.entry_time,np.nan));xv=float(raw.volume.get(row.exit_time,np.nan))
            audits.append(dict(period=name,instrument=row.instrument,entry_time=row.entry_time,lots=row.lots,
                               signal_volume=row.volume,entry_minute_volume=ev,exit_minute_volume=xv,
                               entry_participation=row.lots/ev if ev>0 else np.inf,
                               exit_participation=row.lots/xv if xv>0 else np.inf))
    audit=pd.DataFrame(audits);audit.to_csv(OUT/'execution_volume_audit.csv',index=False)
    stresses=[]
    for label,slips,fee_scale,overrides in [
        ('base',1,1.,{}),('3_ticks_per_side',3,1.,{}),
        ('fees_x1.5',1,1.5,{}),('participation_0.5pct',1,1.,{'volume_participation':.005}),
        ('margin_40pct',1,1.,{'margin_rate':.40}),
        ('risk_half',1,1.,{'risk_fraction':.005})]:
        stress_specs={s:{**v,**{k:v[k]*fee_scale for k in
                        ['open_fee_rate','open_fee_fixed','close_fee_rate','close_fee_fixed']}}
                      for s,v in specs.items()}
        stress_trades=pd.concat([price_trades(t,stress_specs[s],slips)
                                 for s,t in trades.groupby('instrument')],ignore_index=True)
        for period in ['validation','2020','test']:
            start,end=spans[period];end=min(pd.Timestamp(end),common_end)
            l,d,m=simulate(stress_trades,stress_specs,start,end,{**config['risk'],**overrides},minute)
            stresses.append(dict(scenario=label,period=period,**performance(l,d,m)))
    sensitivity=pd.DataFrame(stresses);sensitivity.to_csv(OUT/'sensitivity.csv',index=False)
    yearly=[]
    for period,ledger in ledgers.items():
        if period not in ['train','validation','test']:continue
        a=ledger[ledger.accepted].copy();a['year']=a.entry_time.dt.year
        for (year,symbol),g in a.groupby(['year','instrument']):
            yearly.append(dict(period=period,year=year,instrument=symbol,trades=len(g),
                               pnl_cny=g.account_pnl_cny.sum(),mean_unit_net_bps=g.net_return.mean()*10000,
                               median_unit_net_bps=g.net_return.median()*10000,
                               win_rate_pct=g.account_pnl_cny.gt(0).mean()*100))
    pd.DataFrame(yearly).to_csv(OUT/'yearly_instrument.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for i,period in enumerate(['train','validation','test']):
        d=curves[period];m=marks[period]
        axes[i,0].plot(d.index,d.equity/1e6)
        eq=np.r_[config['risk']['initial_capital'],m.equity.to_numpy()]
        dd=(eq/np.maximum.accumulate(eq)-1)[1:]
        axes[i,1].plot(m.timestamp,dd*100,color='firebrick')
        axes[i,0].set(title=period,ylabel='Equity, CNY million')
        axes[i,1].set(title=period,ylabel='Minute-close drawdown, %')
        for ax in axes[i]:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'equity_drawdown.png',dpi=140);plt.close(fig)
    selected_paths=[Path('src/universe_research.py'),Path('src/daily_universe_research.py'),
                    Path('src/portfolio_engine.py'),Path('src/final_project.py'),FREEZE,
                    Path('src/render_report.py'),Path('README.md'),
                    Path('Futures Commission Schedule (All Exchanges).xlsx')]
    manifest=dict(executed_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_config=config,code_and_config_sha256={str(p):sha256(p) for p in selected_paths},
                  data_sha256={r['data_file']:r['sha256'] for r in coverage})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    report(config,summary,sensitivity,pd.DataFrame(conf),audit)
    from src.render_report import render
    render(OUT)
    return summary


def report(config,summary,sensitivity,conf,audit):
    columns=['period','annualized_return_pct','sharpe','max_drawdown_pct','trades',
             'win_rate_pct','max_margin_pct','all_required_pass']
    required=summary[summary.period.eq('2020')].iloc[0]
    status=('Формальные численные цели в базовом сценарии за 2020 год выполнены.'
            if required.all_required_pass else
            'Формальные численные цели за 2020 год не выполнены; нельзя объявлять задание успешным по доходности.')
    rules=pd.DataFrame(config['candidates']).drop(columns='candidate_id')
    text=f'''# MFT: итоговый многоинструментный проект

{status}

Капитал: **3 000 000 CNY**. Цели: годовая доходность **>20%**, Sharpe **>1,5**,
максимальная просадка **<15%**; win rate >50% необязателен. Проверка относится к
полному календарному 2020 году, после заморозки выбора на 2015–2018/2019.
2019 использован для выбора: даже если он проходит цели, это НЕ независимое
подтверждение. 2020–2021 по индексным стратегиям уже просматривались ранее;
расширенный портфель заморожен до его расчёта, но весь проект ретроспективен.
Новый нетронутый holdout за пределами предоставленных данных необходим.

## Результаты счёта после комиссий и проскальзывания

{markdown_table(summary[columns])}

Каждый период начинает отдельный счёт 3 млн. Train/validation/test нельзя склеить.
2021 заканчивается 18 октября — это неполный год. `test` включает 2020 и доступную
часть 2021, а не выдуманные сделки до 31 декабря. Доходность годовая геометрическая:
`(equity_end / initial) ** (365.25 / elapsed_days) - 1`. Sharpe — по дневным
доходностям **счёта**, включая дни без сделок; безрисковая ставка принята нулевой.
Годовой множитель — корень из числа наблюдений в год. В календаре присутствуют
все будни (праздники дают нули) и наблюдаемые выходные ночные cash-flow дни;
это приближение календаря, а не исторический биржевой trading-day calendar.
Просадка включает исходные 3 млн и все доступные **минутные close** открытых
позиций. Внутриминутная просадка, очереди и price-limit исполнение не восстановлены.

## Правила и заморозка выбора

Проверено {config['configurations_searched']} комбинаций на 65 контрактах с известной
метаданной из доступных 77 файлов. Остальные не получают придуманные комиссии или
мультипликаторы. Train 2015–2018 задаёт пороги, validation 2019 выбирает правила;
2020–2021 не входят в функцию отбора. Рейтинг: меньшая из train/validation t-статистик
не ниже 0,5, положительная средняя в обоих периодах, минимум 100/30 сигналов.
По одному правилу на инструмент. После sizing нужны положительный PnL в обоих
периодах и минимум 10 исполненных сделок validation. Это эвристика отбора,
**не доказательство статистической значимости** после столь большого перебора.
Журнал всех проверок сохраняется; низкие индивидуальные t и выбор на 2019
означают существенный риск переобучения. Выбор не корректируется после test.

{markdown_table(rules)}

`daily_trend`: направление суммы лог-дневных доходностей прошлых N дней.
`daily_reversal`: противоположное направление этой суммы. Ролл-доходность
становится NaN, не используется как рыночное движение. Вход — open второго
5-минутного бара дневной сессии, выход — её последний close.
`daily_opening_momentum/reversal`: направление/противоположное направление
изменения первого 5-минутного close относительно фактического предыдущего
дневного close того же symbol, вход следующий бар, выход конец дневной сессии.
`momentum/reversal`: движение от open N-го с конца бара до текущего close внутри
непрерывного блока; направление/противоположное. Вход следующий бар, holding
баров до выхода; holding=0 — последнее закрытие наблюдаемого блока.
`opening_*`: первый бар непрерывного блока против предыдущего полного бара,
вход второй бар, выход конец блока — **не то же самое, что предыдущий дневной close**.
Для каждого правила абсолютный сигнал должен превышать сохранённый train-квантиль,
side ограничивает long/short. Ночные блоки обрываются на полуночи; неполные бары
не используются. Пропуски минут разбивают блок, и end-of-block выход может оказаться
ранним: это явно отмеченное ex-post ограничение наблюдаемого непрерывного CSV,
не готовая live-политика выхода/роллов. Для дневных правил смешанные symbol даты
отсекаются по полному дню; такой quality screen также известен лишь постфактум.

## Капитал, объём, издержки

Размер — целое число лотов. Оценочный риск сделки — 1% от меньшего начального/
текущего капитала, делённый на notional × лагированную 20-дневную волатильность
(минимум 10 наблюдений, floor 0,5%). Это **не стоп и не ограничение фактического
убытка**. Маржа: 20% notional как консервативный постоянный сценарий, не исторический
тариф брокера; лимиты 30% капитала на инструмент и 60% суммарно. Максимум 1%
объёма уже закрывшегося сигнального бара и 1% его OI. На нулевом объёме, OI<100
и без истории риска нет входа. Если один лот не помещается, сигнал пропускается.
При превышении 60% маржи после минутной переоценки закрывается целиком крупнейшая
позиция с актуальным close; учитываются выходная комиссия и ухудшение на tick.
Это упрощённая модель margin call, не гарантия исполнения по close. Устаревшая
котировка не используется для принудительного выхода. Число таких выходов и
остаточных превышений записано в performance.csv. Контроль добавлен после
первичного test для исправления учёта маржи, **не для отбора победителей**;
состав и пороги не менялись, результаты поэтому остаются ретроспективными.
После margin call повторный вход на этот рынок запрещён до исходного планового выхода.
Одновременно не более одной позиции на инструмент. Приоритет одновременных входов
фиксирован по имени инструмента. Денежный ledger проверяется против суммы PnL.

Комиссии взяты **буквально из пользовательского Excel 2026**, `‰ / 1000`,
фиксированные CNY — за лот, close-today применяется к внутридневным выходам.
Это сценарий, не точная реконструкция 2015–2021: биржевые тарифы менялись.
Прежде чем заявлять историческую исполнимость, нужна история комиссий и маржи.
Open/close исполнения ухудшаются на **один tick с каждой стороны**. Воздействие
на рынок не моделируется сверх ограничения объёма и tick-сценария.

`execution_volume_audit.csv` сравнивает число лотов с фактическим будущим минутным
объёмом входа/выхода **только после расчёта**, не выбирает сделки задним числом.
Ограничение по прошлому бару не гарантирует объём следующей минуты. Плохой audit
требует bid/ask/очередей и отдельной модели частичного исполнения.

## Проверки устойчивости

{markdown_table(sensitivity[['scenario','period','annualized_return_pct','sharpe','max_drawdown_pct','all_required_pass']])}

95% блочный bootstrap дневной доходности счёта, блок 5 дней, 1000 повторов,
seed 42; интервалы условны на выбранные правила, **не учитывают перебор стратегий**:

{markdown_table(conf)}

## Идеи из отчёта Alexandr Khozhanov

Из пользовательского `MFT Report_Alex K (1).pdf` взяты гипотезы краткосрочного
возврата после экстремального движения, следующего-бара исполнения и портфеля
разных рынков. Чужие 20,64% / Sharpe 2,936 не перенесены в результат проекта.
В тексте отличаются список одиночных кандидатов (JD) и состав портфеля (RU),
а проскальзывание не оговорено. При честном одном tick на каждой стороне
короткие 5-минутные сделки существенно хуже; найденные нами правила и результаты
другие. Нельзя переносить чужие метрики без тех же данных и исполнения.

## Что сдавать и как воспроизвести

`README.md`, `configs/final_portfolio.json`, `src/`, `tests/`, исторические
ноутбуки 01–05 и итоговый `06_final_portfolio.ipynb`. Все итоговые CSV и графики
находятся в `reports/final_portfolio/`; SHA256 данных, кода, Excel и config —
в `manifest.json`. Запуск итогового расчёта: `python -m src.final_project`.
Полный скрининг и команды тестов перечислены в README. Цели не достигаются
редактированием метрик, подгонкой периода или незаметным исключением убытков.

## Ограничения и следующий контроль

Много тысяч тестов, использование 2019 для отбора, уже просмотренные годы,
неполная историческая спецификация расходов, синтетические continuous CSV,
отсутствие исполнения на исходном expiring contract и ex-post quality filters
не позволяют называть это доказанной торговой стратегией. Нужны отдельные
контрактные ряды, исторические торговые календари/лимиты/комиссии/маржа, bid/ask,
и новый нетронутый период или paper trading. Даже прохождение численных целей
базового сценария не снимает этих ограничений.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    Path('reports/FINAL_REPORT.md').write_text(
        '# Итог проекта MFT\n\nАктуальный многоинструментный отчёт: '
        '[REPORT.md](final_portfolio/REPORT.md).\n\n'+status+'\n\n'+
        markdown_table(summary[columns])+'\n\nValidation использован для отбора, '
        'test ретроспективен; точные ограничения и стресс-сценарии — в отчёте.\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true')
    args=p.parse_args()
    if args.freeze:freeze_selection()
    run()
