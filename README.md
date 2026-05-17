# Лабораторная работа №2: однослойный перцептрон

Проект содержит автономную реализацию однослойного перцептрона для бинарной
классификации. Модель, обучение, метрики и дополнительные эксперименты написаны
на Python в учебных целях: готовые ML-модели не используются.

Готовый отчёт находится в [`output/report.pdf`](output/report.pdf).

## Что реализовано

- Подготовка данных для бинарной классификации: стратифицированное разбиение,
  стандартизация признаков по обучающей выборке, перенос тех же параметров на тест.
- Перцептрон с sigmoid-активацией и бинарной кросс-энтропией.
- Mini-batch gradient descent с сохранением train/validation loss по эпохам.
- Эксперименты со скоростью обучения, размером batch и инициализацией весов.
- Дополнительные задания: собственные генераторы данных, hinge loss, L2-регуляризация,
  precision/recall/F1/ROC-AUC, ROC-кривая, анализ ошибок, momentum и 5-fold
  cross-validation.
- Генерация графиков, CSV-таблиц и PDF-отчёта.

## Структура проекта

```text
src/
  data.py         # генерация и подготовка данных
  perceptron.py   # модель и обучение
  metrics.py      # метрики качества
  plotting.py     # графики экспериментов
  experiments.py  # полный сценарий экспериментов
  report.py       # сборка PDF-отчёта
tests/
  test_core.py    # базовые проверки ядра
artifacts/
  figures/        # графики
  tables/         # таблицы результатов
output/
  report.pdf      # финальный отчёт
```

## Установка

Рекомендуется использовать локальное виртуальное окружение, чтобы не засорять
глобальный Python.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

На Linux/macOS команды аналогичны:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Запуск

Полный запуск пересобирает таблицы, графики и PDF-отчёт:

```powershell
.\.venv\Scripts\python.exe main.py
```

Для Linux/macOS:

```bash
./.venv/bin/python main.py
```

После выполнения появятся:

- `artifacts/results.json` - сводка результатов;
- `artifacts/figures/` - графики loss, ROC, границы решений и сравнения экспериментов;
- `artifacts/tables/` - CSV-таблицы с метриками;
- `output/report.pdf` - финальный отчёт.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m compileall main.py src tests
```

Ожидаемый результат: все unit-тесты проходят, компиляция Python-файлов завершается
без ошибок.

## Важные ограничения

- `scikit-learn` используется только для `make_classification` и
  `train_test_split`, что соответствует заданию.
- `PyTorch`, `TensorFlow`, `Keras`, `sklearn.linear_model` и другие готовые
  классификаторы не используются.
- PDF-отчёт собирается через ReportLab с Unicode-шрифтом. Если на системе нет
  подходящего шрифта, можно явно задать путь:

```bash
REPORT_FONT_PATH=/path/to/DejaVuSans.ttf ./.venv/bin/python main.py
```

## Воспроизводимость

В экспериментах используется фиксированный `random_state = 42`. Базовая модель
после полного запуска показывает test accuracy около `0.89` и ROC-AUC около
`0.94`; небольшие отличия возможны при изменении версий зависимостей.
