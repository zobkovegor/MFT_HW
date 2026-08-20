"""Generate reviewable research artifacts; run with python -m src.project_report."""
from __future__ import annotations
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.opening_move_final import INSTRUMENTS, run_opening_move_research


def markdown_table(frame):
    frame=frame.reset_index() if frame.index.name or not isinstance(frame.index,pd.RangeIndex) else frame
    def fmt(x):
        if isinstance(x,(float,np.floating)):
            return f'{x:.3f}' if np.isfinite(x) else '—'
        return str(x)
    rows=['| '+' | '.join(map(str,frame.columns))+' |',
          '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(fmt(x) for x in r)+' |' for r in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def export_results(results, output_dir='reports', data_dir='data'):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    for key in ['thresholds','coverage','data_audit','trade_summary','instrument_period_summary',
                'yearly_summary','equity_summary','goal_check','robustness','cost_sensitivity',
                'confidence_intervals','comparison','allocated_trades','accepted_trades']:
        results[key].to_csv(out/f'{key}.csv',index=not isinstance(results[key].index,pd.RangeIndex))
    for p,d in results['equity_curves'].items():
        d.to_csv(out/f'equity_{p}.csv')
        results['minute_equity'][p].to_csv(out/f'minute_equity_{p}.csv',index=False)
    for (book,p),t in results['comparison_trades'].items():
        t.to_csv(out/f'trades_{book}_{p}.csv',index=False)
        results['comparison_curves'][(book,p)].to_csv(out/f'equity_{book}_{p}.csv')
    hashes={}
    for instrument in INSTRUMENTS:
        path=Path(data_dir)/f'{instrument}9999.CCFX.csv'
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
        hashes[str(path)]=digest.hexdigest()
    code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path('src/opening_move_final.py'),Path('src/project_report.py'),
                           Path('src/data_loader.py'),Path('src/bar_builder.py')]}
    manifest={'config':results['config'],'data_sha256':hashes,'code_sha256':code_hashes,
              'packages':{p:version(p) for p in ['numpy','pandas','matplotlib','nbclient','nbformat']}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,2,figsize=(14,11))
    for i,p in enumerate(['train','validation','test']):
        d=results['equity_curves'][p]
        axes[i,0].plot(d.index,d.equity_cny/1e6,label='Opening strategy')
        axes[i,0].axhline(results['config']['initial_capital']/1e6,c='gray',ls='--',lw=1)
        axes[i,0].set(ylabel='Equity, million CNY',title=p)
        m=results['minute_equity'][p]
        axes[i,1].plot(m.time,m.drawdown*100,color='firebrick',lw=.7)
        axes[i,1].set(ylabel='Minute-close drawdown, %',title=p)
        for ax in axes[i]: ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'equity_drawdown.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    for ax,p in zip(axes,['train','validation','test']):
        for book in ['opening','momentum_indices','combined_indices']:
            d=results['comparison_curves'].get((book,p))
            if d is not None: ax.plot(d.index,d.equity_cny/1e6,label=book)
        ax.set(title=p,ylabel='Equity, million CNY');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'strategy_comparison.png',dpi=150);plt.close(fig)
    columns=['annualized_return_pct','annualized_sharpe','max_drawdown_pct',
             'minute_close_max_drawdown_pct','trades','win_rate_pct','maximum_margin_usage_pct']
    comparison=results['comparison']
    report='''# Итоговый исследовательский отчёт MFT

Проект завершён как воспроизводимое исследование стратегий. Проверенные варианты
не подтверждают выполнение всех целевых требований. Положительное среднее на одну
сделку нельзя приравнивать к годовой доходности капитала или к доказанной стратегии.

## Данные и воспроизведение

Использованы исходные минутные IF, IC, IH. Исторические ноутбуки 01–04 сохранены.
Ноутбук 05 содержит обучение на IF и окончательный общий расчёт. Единый расчёт:
`python -m src.project_report`. Проверки: `python -m unittest discover -s tests -v`.
Все таблицы, сделки, минутная переоценка и SHA256 исходных файлов находятся рядом
с этим отчётом. Отчёт строится автоматически из фактических результатов.

'''+markdown_table(results['coverage'])+'''

Train — 2015–2018, validation — 2019, test — 2020 до фактического конца данных
2021-10-18. Периоды уже просматривались в ходе исследования: test здесь название
хронологического среза, а не нетронутая контрольная выборка. Каждый период начинает
с отдельного капитала 3 млн CNY. Три кривые нельзя склеить как единую историю счёта.

## Правила стратегии открытия

Закрытие первого 15-минутного бара сравнивается с фактическим закрытием предыдущего
торгового дня того же контракта. Если движение меньше отрицательного 80-го
процентиля абсолютных движений train, следующая свеча используется для входа short
по open. Выход — последнее закрытие дневной сессии. Порог фиксирован для каждого
инструмента. Сессии принимаются в наблюдаемых раскладках 18/16 полных баров.
Лаг предыдущего дня вычисляется до удаления неполных и смешанных дней.

'''+markdown_table(results['thresholds'])+'''

Размер позиции — целое число контрактов по 0,5% от меньшего из начального и
текущего капитала, делённым на оценку риска одного контракта. Риск — стандартное
отклонение дневных доходностей за 20 прошлых дней (минимум 15 наблюдений), известное
до открытия. Оценка имеет нижнюю границу 0,1%. Лимиты: 20% капитала на маржу одной
позиции, 60% суммарно, 1,5% суммарного оценочного риска и максимум 20 контрактов,
с дополнительным консервативным лимитом 10 контрактов в ограниченном режиме
2015-09-07–2017-02-16. Если один контракт не помещается в бюджет, сделки нет.
0,5% — правило размера, не стоп-лосс и не гарантия максимального убытка.
На одном инструменте одновременно допускается одна позиция, без пирамидинга.
Одновременные входы обрабатываются по имени инструмента; это фиксированный порядок,
не оптимизация по результатам. Маржа переоценивается на каждом минутном close.

## Издержки и расхождения с прежними результатами

1 bp = 0,01%; 1‰ = 0,1% = 10 bp. Excel содержит 0,23‰ и 3,45‰, что буквально
соответствует 2,3 и 34,5 bp. Старый код использовал 0,23 и 3,45 bp.
Числа из таблицы Excel поэтому не являются подтверждённой историей тарифа.

Основной сценарий historical_proxy восстанавливает тариф открытия 0,25 bp до
2015-08-03, затем 0,23 bp; закрытие в тот же день: обычный тариф до 2015-08-26,
затем 1,15 bp, с 2015-09-07 — 23 bp, с 2017-02-17 — 9,2 bp,
с 2017-09-18 — 6,9 bp, с 2018-12-03 — 4,6 bp и с 2019-04-22 — 3,45 bp.
Также учтён сценарный сбор 1 CNY за каждую из двух поданных заявок с 2015-08-03.
Маржа: консервативные 40% до 2017-02-20; далее IF/IH 20%, затем 15%, с
2018-12-04 — 10%; IC 30%, с 2018-12-04 — 15%, с 2019-04-23 — 12%.
Это реконструкция основных режимов и консервативные допущения ранней маржи;
брокерские надбавки, все промежуточные изменения, особые параметры экспирации и
клиентские ограничения полностью не восстановлены. Нельзя назвать её точным
историческим моделированием исполнения и клиринга.

Источники для проверки: [раскрытие CFFEX 2024](https://www.cffex.com.cn/cn/zjssf/20240701/39212.html),
[исследование изменений правил Huatai](https://www.htfc.com/wz_upload/png_upload/20200827/15985190946611d21b3.pdf),
[уведомление о режиме сентября 2015](https://cfc108.com/zxjtqh/2015-09/05/article_414101.shtml),
[изменения декабря 2018](https://www.bhfcc.com/customer-center-ques-details.html?id=2095),
[публикация уведомления апреля 2019](https://www.qhfco.net/customer-service/trading-guide/trading-notice/494460d8-518b-11f0-89c7-b8e92477e4ec).
Тариф CFFEX 2024 приведён для проверки единиц и не применяется ретроспективно.

Проскальзывание основного расчёта — один тик на каждую сторону, то есть два тика
за полный круг. Комиссия считается от цены исполнения каждой стороны, PnL от
мультипликатора IF/IH 300, IC 200. Сравнение сценариев ниже на один контракт,
до портфельного отбора и размера позиций:

'''+markdown_table(results['cost_sensitivity'])+'''

## Итоги портфеля открытия

'''+markdown_table(results['equity_summary'][columns])+'''

Годовая доходность — CAGR по фактическим календарным датам покрытия; Sharpe —
среднее/стандартное отклонение дневных доходностей × sqrt(252), ставка без риска 0.
В дневном календаре только наблюдаемые торговые даты, включая дни без сделок.
Просадка отсчитывается также от начальных 3 млн, чтобы первый убыток не исчезал.
Дневная и минутная просадки показаны раздельно. Минутная использует close и может
пропускать худшие движения внутри минуты. PnL сделок сверяется с итогом переоценки.

Средний payoff всех сигналов по одному контракту и доходность итогового портфеля
могут иметь разный знак. Правило размера пропускает часть сигналов, если один
контракт слишком велик для бюджета, и назначает разное число контрактов остальным.
В 2019 эта разница существенна: положительное среднее всех сигналов не сохранилось
в фактическом портфеле. Таблицы allocated_trades.csv и cost_sensitivity.csv позволяют
проверить это напрямую. Порог и размер не меняются ради исправления результата.

![Капитал и просадка](equity_drawdown.png)

'''+markdown_table(results['yearly_summary'])+'''

## Сравнение и комбинация

Часовой сигнал пересчитан на одинаковой вселенной IF/IC/IH: отрицательное движение
за четыре бара выше train-порога по модулю, вход на следующем open и выход после
четырёх баров внутри той же непрерывной сессии. Это исправленная сопоставимая
версия, отличающаяся от старого ноутбука 03, где также был RU и вход по close.
RU не включён в это сравнение, чтобы не смешивать календари и схемы комиссий.
Комбинация принимает оба сигнала с общими лимитами, запрещает пересечение позиций
одного инструмента и не использует будущую прибыль для выбора сигнала.

'''+markdown_table(comparison[['strategy','period']+columns])+'''

![Сравнение стратегий](strategy_comparison.png)

## Статистика и устойчивость

Средние, медианы и win rate в таблицах сделок относятся к фактически принятым
сделкам после издержек. Payoff — доля номинала контракта, не доходность маржи или
счёта. Наивный trade t-statistic только описательный: IF/IC/IH зависимы.
Для среднего дневного PnL дополнительно рассчитан воспроизводимый 95%-й интервал
блочным bootstrap (блок 5 торговых дней, 1000 повторов, seed 42), с одновременным
объединением инструментов по дню и включением нулевых дней. Интервал не исправляет
множественный перебор идей и просмотр контрольных данных.

'''+markdown_table(results['confidence_intervals'])+'''

Таблица robustness.csv проверяет train-квантили 70/80/90% и 1/2/3 тика на сторону.
Это диагностика payoff всех сигналов по одному контракту, а не повторный подбор
оптимальных параметров капитала на test. Средние взвешены числом сделок через
агрегацию самих сделок, а не равными весами инструментов.

## Проверка задания и вывод

'''+markdown_table(results['goal_check'][columns+[
            'return_above_20pct','sharpe_above_1_5','drawdown_below_15pct','required_goals_pass']])+'''

Цели: годовая доходность >20%, Sharpe >1,5, максимальная просадка <15%; win rate
>50% — дополнительная метрика. На показанных хронологических срезах стратегия
не удовлетворяет всем обязательным условиям. Комбинация также должна оцениваться
по таблице выше, а не по прежним иллюстративным числам. Законченный исследовательский
результат — неподтверждённый кандидат, а не обещание прибыльной торговли.

Дни смены контракта и неполные дни отбрасываются по всей дневной записи. Это
ex-post фильтр качества данных: вживую нельзя заранее узнать все смены символа
и пропуски. Потребуется известное заранее правило выбора контракта и полный
контрактный архив. Ограничения OHLC: цены открытия/закрытия не гарантируют исполнение,
нет bid/ask и очереди, модель не проверяет отсутствие встречной ликвидности на
лимитах. Размер позиции не проверен по доступному объёму каждого исполнения;
нет stop-loss, брокерского маржинального механизма и затрат на инфраструктуру.
Результат исследования не является разрешением запускать стратегию в торговлю.

Практическое продолжение после завершения проекта: получить свежий контрольный
период и контрактные данные, проверить календарь роллов, реальные брокерские
издержки и ликвидность. Новые модели, включая ML, потребуют новой контрольной
выборки. На этих данных увеличение плеча или подбор параметров ради требований
задания не доказывает наличие устойчивого преимущества.
'''
    (out/'OPENING_INDICES_REPORT.md').write_text(report,encoding='utf-8')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--output-dir',default='reports')
    args=parser.parse_args()
    results=run_opening_move_research(data_dir=args.data_dir)
    out=export_results(results,args.output_dir,args.data_dir)
    print(results['equity_summary'].to_string())
    print(f'Artifacts saved to {out.resolve()}')


if __name__=='__main__':
    main()
