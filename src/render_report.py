"""Render a compact submission PDF from the calculated CSVs; no manual metrics."""
from pathlib import Path
import html
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether


def render(directory='reports/final_portfolio'):
    out=Path(directory);s=pd.read_csv(out/'performance.csv')
    a=pd.read_csv(out/'selection_audit.csv');stress=pd.read_csv(out/'sensitivity.csv')
    conf=pd.read_csv(out/'confidence.csv');audit=pd.read_csv(out/'execution_volume_audit.csv')
    regular=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold=regular.with_name('DejaVuSans-Bold.ttf')
    if not regular.exists():raise FileNotFoundError('DejaVu Sans font required for Russian PDF')
    pdfmetrics.registerFont(TTFont('DejaVu',str(regular)))
    pdfmetrics.registerFont(TTFont('DejaVuBold',str(bold)))
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText','Heading1','Heading2','Title']:
        styles[name].fontName='DejaVuBold' if name in ['Title','Heading1','Heading2'] else 'DejaVu'
    styles['Normal'].fontSize=9;styles['Normal'].leading=12
    styles['Title'].fontSize=18;styles['Title'].leading=24
    styles.add(ParagraphStyle(name='Cell',fontName='DejaVu',fontSize=7,leading=10,alignment=TA_LEFT))
    story=[]
    def paragraph(text,style='Normal'):
        story.append(Paragraph(text,styles[style]));story.append(Spacer(1,.2*cm))
    def table(frame,widths=None):
        def fmt(v):
            if isinstance(v,(float,np.floating)):return f'{v:.2f}' if np.isfinite(v) else '—'
            return str(v)
        rows=[[Paragraph(html.escape(str(c)),styles['Cell']) for c in frame.columns]]
        rows += [[Paragraph(html.escape(fmt(v)),styles['Cell']) for v in r]
                 for r in frame.itertuples(index=False,name=None)]
        t=Table(rows,colWidths=widths or [17*cm/len(frame.columns)]*len(frame.columns),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dae5ef')),
                               ('GRID',(0,0),(-1,-1),.3,colors.lightgrey),
                               ('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
        story.append(KeepTogether([t]));story.append(Spacer(1,.3*cm))
    paragraph('MFT: многоинструментная стратегия', 'Title')
    paragraph('Автор: Егор Зобков. Воспроизводимый исследовательский проект на минутных фьючерсных данных.')
    paragraph('1. Задание и честный итог','Heading2')
    paragraph('Начальный капитал 3 000 000 CNY. Цели: годовая доходность &gt;20%, Sharpe &gt;1,5, '
              'максимальная просадка &lt;15%; win rate &gt;50% необязателен. '
              'Основной год проверки — полный 2020 год. 2019 использован для отбора, а не независимого подтверждения.')
    row=s[s.period.eq('2020')].iloc[0]
    paragraph('Цели в базовом сценарии выполнены.' if row.all_required_pass else
              'Цели на 2020 год НЕ выполнены. Положительный validation не подтвердился на последующих данных. '
              'Результат нельзя представлять как доказанную прибыльную стратегию.')
    display=s[['period','annualized_return_pct','sharpe','max_drawdown_pct','trades','win_rate_pct']].copy()
    display.columns=['Период','Годовая, %','Sharpe','Минутная DD, %','Сделки','Win rate, %'];table(display)
    paragraph('Train 2015–2018, validation 2019, test 2020–2021-10-18. Счёт сбрасывается до 3 млн в начале '
              'каждого периода. 2021 неполный: годовая цифра — экстраполяция CAGR, не фактическая годовая доходность. '
              'Исторические годы уже просматривались: весь проект ретроспективен.')
    paragraph('2. Логика и отбор','Heading2')
    paragraph('Скрининг: 65 инструментов с известной метаданной из 77 CSV; 17 290 конфигураций. '
              'Гипотезы: intraday momentum/reversal, движение открытия, дневной тренд/возврат, часы сессий. '
              'Пороги — только по train. По одному правилу на рынок: положительные средние train и validation, '
              'минимум 100/30 сигналов, меньшая t-статистика ≥0,5; затем положительный PnL после sizing '
              'в обоих периодах и ≥10 исполненных validation сделок. Рейтинг не является тестом значимости '
              'после тысяч сравнений. Правила заморожены до первого портфельного test; после него не перевыбирались.')
    picks=a[a.selected][['symbol','family','timeframe','lookback','holding','quantile','side']].copy()
    picks.columns=['Рынок','Правило','TF, мин','Lookback','Holding','Квантиль','Сторона']
    table(picks,[v*cm for v in [1.2,4.7,1.5,2,2,2.4,2.2]])
    paragraph('Дневной trend/reversal: сумма лог-доходностей прошлых N дней, направление/противоположное направление. '
              'Дневное opening: первый 5-минутный close против предыдущего дневного close того же symbol. '
              'Вход второго бара, выход конца дневной сессии. Intraday: close против open N-го с конца бара '
              'текущего блока, вход следующего бара; holding=0 — до конца наблюдаемого блока, иначе заданное число баров. '
              'opening без daily означает первый бар блока против предыдущего полного бара, не дневной close.')
    paragraph('3. Учёт капитала и исполнения','Heading2')
    paragraph('Целые лоты; риск-бюджет 1% меньшего начального/текущего капитала, делённый на notional и '
              '20-дневную лагированную волатильность (минимум 10 дней, floor 0,5%). Это не стоп и не гарантия убытка. '
              'Предел лотов: 1% уже известного объёма сигнального бара и 1% OI. OI&lt;100, volume=0, '
              'отсутствие истории риска или размер меньше одного лота дают пропуск. Одна позиция на рынок. '
              'При одновременном входе порядок по имени рынка фиксирован.')
    paragraph('Маржа 20% notional — сценарий, не точная историческая ставка. Бюджет 30% капитала на рынок '
              'и 60% суммарно. После minute-close переоценки превышение вызывает закрытие крупнейшей позиции '
              'с актуальной ценой, с комиссией и tick. Контроль добавлен после первого test как исправление '
              'риска; исходные цифры сохранены отдельно. Пороги и состав не менялись.')
    paragraph('Комиссии — буквально из пользовательского Excel 2026: ‰ делится на 1000, CNY берётся за лот, '
              'для внутридневного выхода применяется close-today. Исторические тарифы менялись: это сценарий '
              'затрат, не документированная история 2015–2021. Проскальзывание: 1 tick НА КАЖДОЙ стороне. '
              'PnL ledger сверяется с конечным денежным остатком, open fees списываются при входе, close fees при выходе.')
    risk=s[['period','max_margin_pct','forced_margin_exits','margin_breaches']].copy()
    risk.columns=['Период','Макс. маржа, %','Принудит. выходы','Остаточные превышения'];table(risk)
    paragraph('Доходность: (equity_end / initial)^(365.25 / elapsed_days) − 1. Sharpe — дневные доходности '
              'счёта с нулевыми днями, безрисковая ставка 0, annualization по наблюдениям/год. '
              'Календарь приближённый: все будни и наблюдаемые ночные выходные cash-flow дни. '
              'DD включает начальный капитал и все minute-close marks, но не внутриминутные high/low экстремумы.')
    paragraph('4. Стресс-проверки','Heading2')
    x=stress[stress.period.eq('2020')][['scenario','annualized_return_pct','sharpe','max_drawdown_pct']].copy()
    x.columns=['Сценарий, 2020','Годовая, %','Sharpe','DD, %'];table(x)
    paragraph('95% блочный bootstrap: 5-дневные блоки, 1000 повторов, seed 42; интервалы условны на '
              'выбранные правила и не корректируют selection bias всего поиска.')
    c=conf[conf.period.isin(['validation','2020','test'])].copy()
    c.columns=['Период','Годовая low, %','Годовая high, %','Sharpe low','Sharpe high'];table(c)
    paragraph('5. Кривая капитала и минутная просадка','Heading2')
    story.append(Image(str(out/'equity_drawdown.png'),width=14.0*cm,height=10.77*cm))
    paragraph('6. Ограничения, источники и вывод','Heading2')
    entry=(audit.entry_participation>.05).sum();exit=(audit.exit_participation>.05).sum()
    paragraph(f'Объёмный audit: среди {len(audit)} исполненных сделок entry participation &gt;5% '
              f'в {entry} случаях, exit &gt;5% в {exit}. Будущий объём применяется только для диагностики, '
              'не для подбора удачных сделок. Лимит прошлого объёма не гарантирует заполнение следующей минуты.')
    paragraph('Continuous symbol, роллы, неполные блоки и mixed-symbol дни: ex-post data-quality screens '
              'не эквивалентны живой политике роллов/выхода. Выход по наблюдаемому концу блока может быть ранним '
              'из-за пропусков данных. Ночные блоки обрываются на полуночи. Нет bid/ask, очередей, исторических '
              'лимитов/маржи, settlement и отдельных истекающих контрактов: точное исполнение не доказано. '
              'Заявлять успех можно лишь условно на допущения, и только если конкретный год проходит цели.')
    paragraph('Из отчёта Alexandr Khozhanov (пользовательский MFT Report_Alex K (1).pdf) взяты идеи '
              'экстремального 5-минутного возврата, следующего-бара входа и диверсификации. Чужие метрики '
              'не использованы как наши. В оригинале отличаются JD/RU в составе, slippage не описан. '
              'Здесь издержки обоих fill учитываются явно; результаты другие.')
    paragraph('Следующий честный контроль: отдельные контрактные ряды, историческая спецификация затрат '
              'и новый непросмотренный holdout либо paper trading. Перебирать правила на уже известном test '
              'до достижения 20%/1,5 — не независимая проверка.')
    paragraph('Воспроизведение и файлы','Heading2')
    paragraph('README.md; configs/final_portfolio.json; src/; tests/; ноутбуки 01–06; '
              'reports/final_portfolio/REPORT.md и CSV с ledger, minute/daily equity, sensitivity, '
              'confidence, search/selection audit. manifest.json содержит SHA256 входов и кода. '
              'Запуск: python -m src.final_project; тесты: python -m unittest discover -s tests -v. '
              'Полные команды поиска — в README.')
    def footer(canvas,doc):
        canvas.setFont('DejaVu',8)
        canvas.drawString(2*cm,1.1*cm,'MFT • исследование, не рекомендация к реальной торговле')
        canvas.drawRightString(19*cm,1.1*cm,str(doc.page))
    path=out/'MFT_Project_Report.pdf'
    doc=SimpleDocTemplate(str(path),leftMargin=2*cm,rightMargin=2*cm,
                          topMargin=1.7*cm,bottomMargin=1.8*cm,
                          title='MFT Project — Egor Zobkov',author='Egor Zobkov')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    # Cosmetic-only rerender updates this renderer hash; calculation code/data
    # hashes and evaluation timestamp are left unchanged.
    import json, hashlib
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        renderer=Path('src/render_report.py')
        manifest['code_and_config_sha256'][str(renderer)]=hashlib.sha256(renderer.read_bytes()).hexdigest()
        manifest['pdf_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PDF:',path)
    return path


if __name__=='__main__':render()
