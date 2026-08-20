# MFT: многоинструментное исследование китайских фьючерсов

Python-проект на предоставленных минутных OHLCV: 77 файлов, скрининг 65 инструментов
с известной метаданной, 17 290 комбинаций правил. Капитал 3 млн CNY.
Цели задания: годовая доходность >20%, Sharpe >1,5, просадка <15%; win rate >50%
необязателен. Проверяется полный 2020 год, а не только удобная часть истории.
Положительный отбор на 2019 **не означает успеха на test**. Все результаты —
историческое исследование с явно указанными ограничениями исполнения и перебора.

Начать с [итогового отчёта](reports/final_portfolio/REPORT.md) и
[ноутбука 06](06_final_portfolio.ipynb). Ноутбуки 01–05 сохранены как история;
ранние цифры и чужой пример отчёта не заменяют пересчитанные итоги портфеля.

## Запуск

Python 3.11+; исходные CSV должны находиться в data/ (в репозиторий не включаются).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.final_project
.venv/bin/python -m src.execute_notebook 06_final_portfolio.ipynb
```

Для ноутбука выбрать созданное окружение и выполнить Run All. Полный расчёт
также доступен без Jupyter через команду выше. В reports/final_portfolio/ сохраняются
CSV со сделками и пропущенными сигналами, графики, отчёт, параметры и SHA256 данных.
Для сдачи сформирован PDF: `reports/final_portfolio/MFT_Project_Report.pdf`;
его можно пересобрать отдельно: `.venv/bin/python -m src.render_report`.
Дата конца test берётся из наблюдаемых данных, а не искусственно продлевается
до конца 2021 года. Итоговые артефакты будут обновляться при повторном запуске.

## Организация

- configs/final_portfolio.json: выбор правил и пороги, замороженные до расчёта портфеля test.
- src/universe_research.py: 5/15/30-минутные momentum/reversal, реестр 9 360 конфигураций.
- src/daily_universe_research.py: дневные тренды/возврат/открытие и часы сессий, 5 590 конфигураций.
- reports/opening_universe/: ещё 2 340 внутрисессионных конфигураций открытия.
- src/portfolio_engine.py: long/short, целые лоты, комиссии, прошлый объём/OI, маржа, minute MTM.
- src/final_project.py: frozen evaluation, годовая проверка целей, bootstrap, стресс-сценарии, отчёт.
- src/data_loader.py, src/bar_builder.py: загрузка и 15-минутные бары.
- src/opening_move_final.py, src/project_report.py: прежний трёхиндексный эксперимент.
- tests/: проверки лагов, порога, комиссий, ограничений и учёта капитала.
- src/opening_move_strategy.py: ранняя версия исследовательских функций;
  окончательный отчёт использует opening_move_final.py.

## Полное воспроизведение поиска

```bash
.venv/bin/python -m src.universe_research
.venv/bin/python -c "from src.universe_research import scan; scan(output_dir='reports/opening_universe', timeframes=(5,15,30), lookbacks=(1,), holdings=(0,), quantiles=(.5,.8,.95), families=('opening_reversal','opening_momentum'))"
.venv/bin/python -m src.daily_universe_research
```

Реестры содержат только train/validation статистику. Выбор зафиксирован в configs/;
обычный запуск final_project **не перевыбирает** правила. Для независимого повторения
процедуры отбора есть `python -m src.final_project --freeze`: команда перезапишет
сгенерированную конфигурацию и selection_audit, поэтому не использовать её для
подгонки под уже просмотренный test. Pickle-кеш локальных баров — в artifacts/;
внешние недоверенные pickle не загружать. Данные и кеш не включаются в Git.

Train 2015–2018, validation 2019, test 2020–2021-10-18. Периоды уже исследовались:
test не является полностью нетронутым holdout. Портфель использует literal Excel
комиссии 2026 как сценарий, маржу 20% notional как допущение и 1 tick **на сторону**.
Это не точная история комиссий/маржи. Смешанные дни и неполные блоки отсекаются
постфактум как data-quality screen, что ограничивает применимость к live trading.
Исторический индексный эксперимент содержит отдельный historical_proxy тарифов.
Не смешивать его метрики с итоговым многоинструментным экспериментом
