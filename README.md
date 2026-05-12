# Linear Algebra Lab 2

Автономная реализация лабораторной по однослойному перцептрону.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py
```

Команда создаёт:

- `artifacts/figures/` - графики;
- `artifacts/tables/` - таблицы экспериментов;
- `artifacts/results.json` - машинно-читаемая сводка;
- `output/report.pdf` - финальный отчёт на русском языке.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Модель реализована с нуля. Готовые ML-модели не используются; `scikit-learn`
применяется только для генерации рекомендованного набора данных и разбиения.
