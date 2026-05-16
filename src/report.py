from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_mpl_config_dir = Path("tmp/matplotlib").resolve()
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import pandas as pd
from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def _font_name() -> str:
    candidates = []
    if os.environ.get("REPORT_FONT_PATH"):
        candidates.append(Path(os.environ["REPORT_FONT_PATH"]))
    candidates.extend(
        [
            Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
        ]
    )

    for font_path in candidates:
        if font_path.exists():
            font_name = "LabUnicode"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name

    raise RuntimeError(
        "No Unicode TTF font found for PDF report. Set REPORT_FONT_PATH to a font "
        "with Cyrillic support, for example DejaVuSans.ttf."
    )

def _styles() -> dict[str, ParagraphStyle]:
    font = _font_name()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=font,
            fontSize=22,
            leading=28,
            alignment=1,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=font,
            fontSize=16,
            leading=20,
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=font,
            fontSize=13,
            leading=16,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=10,
            leading=14,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8,
            leading=10,
            spaceAfter=4,
        ),
    }

def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def _add_table(
    story: list[Any],
    csv_path: str,
    styles: dict[str, ParagraphStyle],
    max_rows: int = 12,
    columns: list[str] | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    if columns is not None:
        df = df[[col for col in columns if col in df.columns]]
    if len(df) > max_rows:
        df = df.head(max_rows)
    data = [[Paragraph(str(col), styles["small"]) for col in df.columns]]
    for _, row in df.iterrows():
        data.append([Paragraph(_fmt(value), styles["small"]) for value in row.tolist()])
    total_width = 18.0 * cm
    table = Table(data, repeatRows=1, colWidths=[total_width / len(data[0])] * len(data[0]))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))

def _add_image(story: list[Any], path: str, width_cm: float = 15.5) -> None:
    image_path = Path(path)
    if not image_path.exists():
        return
    img = Image(str(image_path))
    max_width = width_cm * cm
    scale = max_width / img.drawWidth
    img.drawWidth *= scale
    img.drawHeight *= scale
    story.append(img)
    story.append(Spacer(1, 0.25 * cm))

def _p(story: list[Any], text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> None:
    story.append(Paragraph(text, styles[style]))

def build_report(
    results_path: str | Path = "artifacts/results.json",
    output_path: str | Path = "output/report.pdf",
) -> str:
    results_path = Path(results_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    styles = _styles()
    story: list[Any] = []

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    _p(story, "Лабораторная работа №2", styles, "title")
    _p(story, "Однослойный перцептрон: реализация, обучение и анализ", styles, "h1")
    _p(
        story,
        "Работа выполнена автономно на Python. Модель перцептрона реализована с нуля: "
        "готовые ML-модели не использовались. sklearn применён только для генерации "
        "рекомендованного набора данных и стратифицированного разбиения.",
        styles,
    )
    _p(
        story,
        "Базовые параметры: eta = 0.1, epochs = 100, batch_size = 32, random_state = 42. "
        "Тестовая выборка одновременно использована как validation-набор для построения "
        "кривых loss, потому что отдельная validation-выборка в задании не задана.",
        styles,
    )

    _p(story, "1. Теоретическая часть и реализация", styles, "h1")
    _p(
        story,
        "Перцептрон вычисляет z = w^T x + b и для BCE использует sigmoid(z) = "
        "1 / (1 + exp(-z)). Оптимизация выполняется mini-batch gradient descent. "
        "Для BCE градиент по весам имеет вид X^T(y_hat - y) / m; для hinge loss "
        "обновляются только объекты с margin меньше 1.",
        styles,
    )
    _p(
        story,
        "В реализации есть варианты инициализации весов: нулевая, малая случайная "
        "N(0, 0.01) и большая случайная N(0, 10). Также поддержаны L2-регуляризация "
        "и momentum с beta из набора 0.5, 0.9, 0.99.",
        styles,
    )

    _p(story, "2. Базовое обучение", styles, "h1")
    _add_table(story, results["tables"]["base_metrics"], styles)
    _add_image(story, results["figures"]["base_loss"])
    _add_image(story, results["figures"]["base_boundary"])
    _add_image(story, results["figures"]["roc"])
    _add_image(story, results["figures"]["base_errors"])
    base = results["sections"]["base"]
    _p(
        story,
        f"Итоговая test accuracy базовой модели: {base['test_accuracy']:.4f}; "
        f"ROC-AUC: {base['test_roc_auc']:.4f}; F1-score: {base['test_f1']:.4f}.",
        styles,
    )

    story.append(PageBreak())
    _p(story, "3. Обязательные эксперименты", styles, "h1")
    _p(story, "Влияние скорости обучения", styles, "h2")
    _add_table(
        story,
        results["tables"]["learning_rate"],
        styles,
        columns=["lr", "train_accuracy", "test_accuracy", "final_val_loss", "weight_norm"],
    )
    _add_image(story, results["figures"]["learning_rate_loss"])
    lr_best = results["sections"]["learning_rate_best"]
    _p(
        story,
        f"Лучший результат среди проверенных eta по test accuracy: eta = {lr_best['lr']} "
        f"с accuracy = {lr_best['test_accuracy']:.4f}. Малые eta дают медленную "
        "сходимость, слишком крупные могут увеличивать колебания loss.",
        styles,
    )

    _p(story, "Влияние размера batch", styles, "h2")
    _add_table(
        story,
        results["tables"]["batch_size"],
        styles,
        columns=["batch_size", "train_accuracy", "test_accuracy", "final_val_loss", "weight_norm"],
    )
    _add_image(story, results["figures"]["batch_size_loss"])
    batch_best = results["sections"]["batch_size_best"]
    _p(
        story,
        f"Лучший batch по test accuracy: {batch_best['batch_size']}. Batch = 1 "
        "даёт более шумную траекторию, крупные batch делают обновления стабильнее, "
        "но иногда медленнее реагируют на структуру данных.",
        styles,
    )

    _p(story, "Влияние инициализации весов", styles, "h2")
    _add_table(
        story,
        results["tables"]["initialization"],
        styles,
        columns=["init", "train_accuracy", "test_accuracy", "final_val_loss", "weight_norm"],
    )
    _add_image(story, results["figures"]["initialization_loss"])
    _p(
        story,
        "Большая случайная инициализация может насыщать sigmoid и ухудшать начальные "
        "градиенты. Малая случайная и нулевая инициализации для однослойной модели "
        "работают приемлемо, потому что проблемы симметрии скрытых нейронов здесь нет.",
        styles,
    )

    story.append(PageBreak())
    _p(story, "4. Дополнительные задания", styles, "h1")
    _p(story, "Собственный генератор данных", styles, "h2")
    _add_table(
        story,
        results["tables"]["custom_data"],
        styles,
        columns=["dataset", "train_accuracy", "test_accuracy", "test_f1", "final_val_loss"],
    )
    _add_image(story, results["figures"]["custom_linear_boundary"])
    _add_image(story, results["figures"]["custom_xor_boundary"])
    _add_image(story, results["figures"]["custom_circle_boundary"])
    _p(
        story,
        "На линейно разделимых данных перцептрон строит корректную прямую границу. "
        "На XOR и окружности одна прямая не может описать истинное правило, поэтому "
        "качество ограничено геометрией модели.",
        styles,
    )

    _p(story, "Hinge loss и L2-регуляризация", styles, "h2")
    _add_table(
        story,
        results["tables"]["loss_comparison"],
        styles,
        columns=["loss", "train_accuracy", "test_accuracy", "test_f1", "final_val_loss"],
    )
    _add_image(story, results["figures"]["loss_comparison"])
    _add_table(
        story,
        results["tables"]["l2_regularization"],
        styles,
        columns=["l2", "test_accuracy", "final_val_loss", "weight_norm"],
    )
    _add_image(story, results["figures"]["l2_loss"])
    _add_image(story, results["figures"]["l2_weight_norm"])
    _p(
        story,
        "L2-регуляризация уменьшает норму весов и может улучшать обобщение, если "
        "коэффициент умеренный. Слишком большой lambda переусиливает штраф и ухудшает fit.",
        styles,
    )

    _p(story, "Метрики качества и анализ ошибок", styles, "h2")
    _p(
        story,
        "Для тестовой выборки рассчитаны accuracy, precision, recall, F1-score и ROC-AUC. "
        "Ошибки визуализированы поверх разделяющей границы; они концентрируются около "
        "переходной области между классами.",
        styles,
    )

    _p(story, "Momentum", styles, "h2")
    _add_table(
        story,
        results["tables"]["momentum"],
        styles,
        columns=["momentum", "test_accuracy", "final_val_loss", "weight_norm"],
    )
    _add_image(story, results["figures"]["momentum_loss"])
    momentum_best = results["sections"]["momentum_best"]
    _p(
        story,
        f"Наименьший validation loss среди momentum-настроек получен при beta = "
        f"{momentum_best['momentum']}. Слишком высокий beta может вызывать инерционные "
        "колебания около минимума.",
        styles,
    )

    _p(story, "5-fold cross-validation", styles, "h2")
    _add_table(
        story,
        results["tables"]["cross_validation"],
        styles,
        columns=["lr", "batch_size", "mean_accuracy", "std_accuracy"],
    )
    _add_image(story, results["figures"]["cross_validation"])
    cv_best = results["sections"]["cross_validation_best"]
    final_model = results["sections"]["final_model"]
    _p(
        story,
        f"Лучшие параметры по 5-fold CV: eta = {cv_best['lr']}, batch_size = "
        f"{cv_best['batch_size']}, mean accuracy = {cv_best['mean_accuracy']:.4f} "
        f"+/- {cv_best['std_accuracy']:.4f}. Финальная модель с этими параметрами "
        f"дала test accuracy = {final_model['test_accuracy']:.4f}.",
        styles,
    )
    _add_image(story, results["figures"]["final_boundary"])

    _p(story, "Выводы", styles, "h1")
    _p(
        story,
        "Однослойный перцептрон с sigmoid и BCE успешно решает двумерную бинарную "
        "классификацию, когда классы близки к линейно разделимым. Скорость обучения, "
        "размер batch и масштаб начальных весов заметно влияют на сходимость. "
        "Ограничение модели принципиально: для XOR и круговой структуры требуется "
        "нелинейная модель или расширение признакового пространства.",
        styles,
    )

    doc.build(story)
    return str(output_path)
