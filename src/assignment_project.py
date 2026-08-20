"""Submission experiment: user-selected validation-year backtest, not holdout.

Reuse the frozen signal rules. Risk budget 0.5% chosen using the observed 2019
validation results. Do not relabel this as independent OOS performance.
Original unsuccessful evaluations are preserved in their original directories.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.universe_research import load_specs,price_trades
from src.portfolio_engine import candidate_from_config,simulate,performance
from src.project_report import markdown_table
from src.positive_revision import digest

OUT=Path('reports/assignment_validation')
CONFIG=Path('configs/assignment_portfolio.json')


def write_config():
    previous=json.loads(Path('configs/final_portfolio.json').read_text())
    risk={**previous['risk'],'risk_fraction':.005}
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                report_period='2019-01-01 through 2019-12-31',
                training_period='2015-01-01 through 2018-12-31',
                selected_on_validation=True,independent_holdout_claim=False,
                user_request='Use validation-year performance as the assignment backtest',
                risk=risk,slippage_ticks_per_side=1,
                fee_scenario='literal supplied Excel 2026, not historical fee reconstruction',
                candidates=previous['candidates'])
    CONFIG.write_text(json.dumps(config,indent=2,ensure_ascii=False))
    return config


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads(CONFIG.read_text()) if CONFIG.exists() else write_config()
    specs=load_specs();groups=[];minute={};coverage=[]
    for c in config['candidates']:
        s=c['symbol'];t,b,threshold=candidate_from_config(c,specs[s])
        if not np.isclose(threshold,c['train_threshold']):raise ValueError('Changed training threshold')
        groups.append(t)
        path=next(Path('data').glob(s+'9999*'))
        m=pd.read_csv(path,usecols=['date','close','symbol','volume'],parse_dates=['date'])
        minute[s]=m.drop_duplicates('date').sort_values('date')
        coverage.append(dict(symbol=s,path=str(path),first=str(b.start.min()),last=str(b.end.max()),
                             sha256=digest(path)))
    trades=pd.concat(groups,ignore_index=True)
    start='2019-01-01';end='2019-12-31 23:59:59'
    if any(pd.Timestamp(c['last'])<pd.Timestamp(end).normalize() for c in coverage):
        raise ValueError('Incomplete coverage for the selected year')
    ledger,daily,marks=simulate(trades,specs,start,end,risk=config['risk'],minute_frames=minute)
    summary=performance(ledger,daily,marks)
    summary.update(period='validation_2019',selected_on_validation=True,independent_holdout=False)
    pd.DataFrame([summary]).to_csv(OUT/'performance.csv',index=False)
    ledger.to_csv(OUT/'trades.csv',index=False);daily.to_csv(OUT/'daily_equity.csv')
    marks.to_csv(OUT/'minute_equity.csv',index=False)
    goals=pd.DataFrame([
        dict(metric='Annualized return, %',target='>20',actual=summary['annualized_return_pct'],passed=summary['annualized_return_pct']>20,required=True),
        dict(metric='Sharpe',target='>1.5',actual=summary['sharpe'],passed=summary['sharpe']>1.5,required=True),
        dict(metric='Minute-close max DD, %',target='<15',actual=summary['max_drawdown_pct'],passed=summary['max_drawdown_pct']<15,required=True),
        dict(metric='Win rate, %',target='>50 (optional)',actual=summary['win_rate_pct'],passed=summary['win_rate_pct']>50,required=False)])
    goals.to_csv(OUT/'goal_check.csv',index=False)
    accepted=ledger[ledger.accepted].copy()
    instrument=accepted.groupby('instrument').agg(trades=('lots','size'),pnl_cny=('account_pnl_cny','sum'),
        mean_unit_net_bps=('net_return',lambda x:x.mean()*10000),
        median_unit_net_bps=('net_return',lambda x:x.median()*10000),
        win_rate_pct=('account_pnl_cny',lambda x:x.gt(0).mean()*100))
    instrument.to_csv(OUT/'instrument_summary.csv')
    monthly=daily.equity.resample('ME').last()
    monthly_returns=monthly.pct_change(fill_method=None)
    monthly_returns.iloc[0]=monthly.iloc[0]/3_000_000-1
    monthly=pd.DataFrame(dict(equity_cny=monthly,return_pct=monthly_returns*100))
    monthly.to_csv(OUT/'monthly.csv')
    indexed={s:m.set_index('date') for s,m in minute.items()};audits=[]
    for r in accepted.itertuples():
        raw=indexed[r.instrument];ev=float(raw.volume.get(r.entry_time,np.nan));xv=float(raw.volume.get(r.exit_time,np.nan))
        audits.append(dict(instrument=r.instrument,entry_time=r.entry_time,lots=r.lots,
                           entry_volume=ev,exit_volume=xv,
                           entry_participation=r.lots/ev if ev>0 else np.inf,
                           exit_participation=r.lots/xv if xv>0 else np.inf))
    pd.DataFrame(audits).to_csv(OUT/'execution_volume_audit.csv',index=False)
    costs=[]
    for ticks in [1,2,3]:
        priced=pd.concat([price_trades(t,specs[s],ticks) for s,t in trades.groupby('instrument')],ignore_index=True)
        l,d,m=simulate(priced,specs,start,end,risk=config['risk'],minute_frames=minute)
        costs.append(dict(ticks_per_side=ticks,**performance(l,d,m)))
    sensitivity=pd.DataFrame(costs);sensitivity.to_csv(OUT/'cost_sensitivity.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    axes[0].plot(daily.index,daily.equity/1e6,color='#17658c');axes[0].set_ylabel('Equity, CNY million')
    eq=np.r_[3_000_000,marks.equity.to_numpy()]
    axes[1].plot(marks.timestamp,(eq/np.maximum.accumulate(eq)-1)[1:]*100,color='firebrick')
    axes[1].set_ylabel('Minute-close drawdown, %')
    axes[2].bar(monthly.index.strftime('%Y-%m'),monthly.return_pct,color=['#23785c' if v>0 else '#b64c48' for v in monthly.return_pct])
    axes[2].set_ylabel('Monthly return, %');axes[2].tick_params(axis='x',rotation=45)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Intraday portfolio: backtest 2019')
    fig.tight_layout();fig.savefig(OUT/'performance.png',dpi=140);plt.close(fig)
    text=f'''# MFT: портфель внутридневных стратегий

Отчётный период: **01.01–31.12.2019**. Начальный капитал: **3 млн CNY**.
Многоинструментная стратегия на минутных данных фьючерсов с учётом комиссий,
проскальзывания и маржинальных ограничений.

## Выполнение требований при капитале 3 млн CNY

{markdown_table(goals)}

Конечный капитал: **{summary['final_equity_cny']:,.2f} CNY**.
Чистый PnL: **{summary['final_equity_cny']-3_000_000:,.2f} CNY**.
Доходность капитала за период: **{summary['total_return_pct']:.3f}%**,
annualized CAGR: **{summary['annualized_return_pct']:.3f}%**.
Сделок: {summary['trades']}; win rate: {summary['win_rate_pct']:.3f}%.
Максимальная доля маржи: {summary['max_margin_pct']:.3f}%,
остаточных превышений 60%: {summary['margin_breaches']}.

## Логика

Многоинструментный портфель из разных семейств внутридневных сигналов.
Этапы исследования: **2015–2018 — расчёт порогов; 2019 — разработка и бэктест
портфеля**, включая состав правил и риск-бюджет. Для каждого инструмента
используется одно правило, параметры и пороги записаны в конфигурации:

{markdown_table(pd.DataFrame(config['candidates']).drop(columns='candidate_id'))}

`daily_trend`: направление суммы лог-дневных доходностей прошлых N дней.
`daily_reversal`: противоположное направление этой суммы. `daily_opening_*`:
движение первого дневного 5-минутного close против предыдущего дневного close
того же symbol, momentum следует движению, reversal торгует против.
Дневные сигналы входят на open второго 5-минутного бара, выход — последний
дневной close. Intraday momentum/reversal смотрят движение от open N-го
с конца бара текущего блока до текущего close, вход на следующем open;
holding — число баров, holding=0 означает конец наблюдаемого блока.
`opening_*` без daily — первый бар блока против предыдущего полного бара,
**не против предыдущего дневного close**. Для каждого сигнала abs превышает
train-квантиль; side ограничивает направление. Роллы не считаются обычной
доходностью. Все дневные лаги вычисляются до отбора дат.

## Размер позиций и издержки

Риск-бюджет **0,5%** от меньшего начального/текущего капитала. Делится на
notional × прошлую 20-дневную волатильность (минимум 10 наблюдений, floor 0,5%).
Это правило sizing, **не стоп-лосс**. Целые лоты; максимум 1% уже известного
объёма сигнального бара/OI, без объёма и OI<100 нет входа. Если бюджет не
вмещает один контракт, сигнал пропускается. Маржа сценарная 20% notional;
30% капитала на инструмент, 60% суммарно. По одному открытому контрактному
направлению на инструмент; одновременные входы имеют фиксированный приоритет
по имени инструмента. При превышении маржи после минутной переоценки engine
закрывает крупнейшую доступную по актуальному close позицию с расходами.

Комиссия буквально из пользовательского Excel 2026: per-mille /1000, CNY за
лот, close-today для внутридневного выхода. Это **сценарий**, не исторически
точный тариф 2019. Baseline: **один tick на каждой стороне**. Open/close fills
ухудшаются по направлению сделки, комиссии списываются на входе/выходе.
PnL ledger сверяется с изменением капитала. Нет quote-level исполнения/impact
модели, поэтому минутный объём входа/выхода отдельно проверяется в audit.

## Помесячно

{markdown_table(monthly)}

## По инструментам

{markdown_table(instrument)}

## Чувствительность к проскальзыванию

{markdown_table(sensitivity[['ticks_per_side','annualized_return_pct','sharpe','max_drawdown_pct','all_required_pass']])}

Численные цели относятся к указанному baseline, не ко всем сценариям расходов.
При увеличении стоимости исполнения доходность может стать отрицательной.

## Методика расчёта

CAGR `(equity_end / 3_000_000) ** (365.25 / elapsed_days) - 1`; Sharpe по дневным
доходностям счёта с нулевыми днями и нулевой безрисковой ставкой. Annualization
по наблюдениям/год; календарь приближённый: будни и наблюдаемые ночные выходные
cash-flow дни. DD включает исходный капитал и minute-close всех открытых позиций,
но не внутриминутные high/low экстремумы. Mixed-symbol даты, неполные блоки,
пропуски планового выхода — ex-post quality ограничения continuous CSV, не
готовая live-политика роллов. Ночные блоки обрываются на полуночи.

Поиск исходных правил включал 17 290 комбинаций внутридневных сигналов.

## Воспроизведение и сдача

`python -m src.assignment_project`; `python -m src.execute_notebook 07_assignment_validation.ipynb`.
Код src/, конфигурация configs/assignment_portfolio.json, ноутбук 07, этот отчёт
и MFT_Assignment_Validation.pdf. CSV: performance, goal_check, trades,
daily/minute equity, monthly, instrument_summary, cost_sensitivity, execution_volume_audit.
SHA256 данных и кода в manifest.json. Отчёт Alexandr Khozhanov использовался
как источник исследовательских гипотез; результаты рассчитаны собственным кодом.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    Path('reports/FINAL_REPORT.md').write_text(
        '# MFT: портфель внутридневных стратегий\n\n'
        'Отчётный период: **01.01–31.12.2019**.\n\n'
        '[Полный отчёт](assignment_validation/REPORT.md) · '
        '[PDF](assignment_validation/MFT_Assignment_Validation.pdf)\n\n'+markdown_table(goals)+
        '\n\nНачальный капитал 3 млн CNY; комиссии и 1 tick на каждую сторону учтены. '
        'Методика исследования и расчёта приведена в полном отчёте.\n',encoding='utf-8')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=config,data=coverage,
                  code_sha256={str(p):digest(p) for p in [Path(__file__),Path('src/portfolio_engine.py'),
                       Path('src/universe_research.py'),Path('src/daily_universe_research.py'),
                       Path('Futures Commission Schedule (All Exchanges).xlsx'),CONFIG,
                       Path('configs/requirements.json')]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    render_pdf(config,summary,goals,sensitivity)
    print(summary,flush=True)
    return summary


def render_pdf(config,summary,goals,sensitivity):
    import html
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=getSampleStyleSheet()
    for name in ['Normal','Title','Heading2']:
        styles[name].fontName='DejaVuBold' if name!='Normal' else 'DejaVu'
    styles['Normal'].fontSize=9;styles['Normal'].leading=12
    story=[]
    def p(text,style='Normal'):
        story.extend([Paragraph(text,styles[style]),Spacer(1,.18*cm)])
    def table(frame):
        data=[list(map(str,frame.columns))]
        for r in frame.itertuples(index=False,name=None):
            data.append([f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in r])
        cells=[[Paragraph(html.escape(v),styles['Normal']) for v in row] for row in data]
        t=Table(cells,colWidths=[17*cm/len(frame.columns)]*len(frame.columns),repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfece8')),
                               ('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP')]))
        story.extend([t,Spacer(1,.25*cm)])
    p('MFT: портфель внутридневных стратегий','Title')
    p('Егор Зобков • бэктест за 2019 год')
    p('Многоинструментный портфель на минутных данных фьючерсов. '
      'Период расчёта: 01.01–31.12.2019; начальный капитал: 3 млн CNY.')
    p('1. Цели и результат','Heading2')
    g=goals[['metric','target','actual','passed']].copy();g.columns=['Метрика','Цель','Получено','Выполнено'];table(g)
    p(f"Начальный капитал 3 000 000 CNY; конечный {summary['final_equity_cny']:,.2f} CNY. "
      f"Чистый PnL {summary['final_equity_cny']-3_000_000:,.2f} CNY, {summary['trades']} сделок. "
      f"Максимальная маржа {summary['max_margin_pct']:.2f}%, превышений 60% — {summary['margin_breaches']}.")
    p('2. Логика стратегии и методика','Heading2')
    p('Этапы исследования: 2015–2018 — расчёт порогов; 2019 — разработка и бэктест '
      'портфеля, включая состав правил и риск-бюджет. '
      'Исследовано 17 290 сочетаний momentum/reversal, открытия и дневного тренда. '
      'По одному правилу на инструмент; параметры и пороги полностью сохранены в configs/assignment_portfolio.json. '
      'Портфель: NI, AU, T, RU, OI, CY, SM, AP, ZN, JD, SN, SF, AL, CF.')
    p('Дневной тренд: сумма лог-доходностей прошлых N дней, следование/возврат. '
      'Дневное открытие: first close против фактического прошлого дневного close того же symbol. '
      'Вход по open второго 5-минутного бара, выход последний дневной close. '
      'Intraday: движение от open N-го с конца бара до текущего close внутри блока, '
      'вход следующий open, выход через holding баров или в конце блока. '
      'Opening без daily сравнивается с предыдущим полным баром, не с дневным close.')
    p('3. Капитал и затраты','Heading2')
    p('Риск-бюджет 0,5% меньшего начального/текущего капитала. Размер делится на notional '
      'и лагированную 20-дневную волатильность; это не стоп-лосс. Целые лоты, максимум '
      '1% известного объёма сигнального бара/OI, пропуск если меньше одного лота. '
      'Маржа сценарная 20% notional, бюджеты 30% на инструмент/60% суммарно, одна позиция на инструмент.')
    p('Комиссии буквально из пользовательского Excel 2026: ‰/1000 и CNY за лот, close-today. '
      'Это сценарий, не точный исторический тариф 2019. Baseline — 1 tick НА КАЖДОЙ стороне. '
      'Списания на входе/выходе, ledger сверяется с изменением капитала; позиции переоцениваются поминутно.')
    story.append(PageBreak())
    p('4. Капитал, просадка и месячная доходность','Heading2')
    story.append(Image(str(OUT/'performance.png'),width=16.0*cm,height=13.33*cm))
    p('5. Чувствительность к издержкам','Heading2')
    s=sensitivity[['ticks_per_side','annualized_return_pct','sharpe','max_drawdown_pct']].copy()
    s.columns=['Ticks на сторону','Годовая, %','Sharpe','DD, %'];table(s)
    p('Цели выполнены в указанном baseline, не гарантированы при иных расходах. '
      'Доходность — CAGR счёта, Sharpe — дневные доходности с нулевыми днями и нулевой ставкой. '
      'DD включает начальный капитал и minute-close, но не все внутриминутные экстремумы. '
      'Календарь приближённый. Mixed-symbol дни и неполные блоки — ex-post quality screens. '
      'Модель исполнения использует цены баров и сценарное проскальзывание, без bid/ask и очередей заявок.')
    p('Код и воспроизведение: python -m src.assignment_project; ноутбук 07_assignment_validation.ipynb; '
      'README.md, configs/assignment_portfolio.json, tests/. Полная логика, monthly/instrument breakdown, '
      'ledger и minute/daily curves — в reports/assignment_validation/REPORT.md и CSV. SHA256 в manifest.json.')
    def footer(canvas,doc):
        canvas.setFont('DejaVu',8);canvas.drawString(2*cm,1.1*cm,'MFT • портфель внутридневных стратегий • 2019')
        canvas.drawRightString(19*cm,1.1*cm,str(doc.page))
    SimpleDocTemplate(str(OUT/'MFT_Assignment_Validation.pdf'),leftMargin=2*cm,rightMargin=2*cm,
                      topMargin=1.5*cm,bottomMargin=1.7*cm,author='Egor Zobkov',
                      title='MFT Portfolio — 2019').build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':run()
