# Work Log

## 2026-05-04

- Read `алгебра (3).pdf` and `лаб2 (2).pdf`; confirmed the lab requires a from-scratch single-layer perceptron for binary classification, mandatory experiments, and five optional tasks.
- User selected maximum scope: mandatory part plus all five optional tasks, final code plus Russian PDF report.
- User requested a persistent lightweight work log so context compression preserves implementation decisions and status.
- Corrected dependency policy after user interruption: global Python must stay clean; project dependencies must be installed only into local `.venv`.
- Removed packages that were accidentally installed into global user Python during the interrupted install: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `scipy`, `reportlab`, and fresh dependencies. Kept pre-existing PDF tooling (`pdfplumber`, `pypdf`, `pillow`) intact.
- Created project-local `.venv` and installed dependencies from `requirements.txt` into it only.
- Added implementation files: perceptron core, data preparation/generators, metrics, plotting, experiment runner, PDF report builder, `main.py`, README, and unit tests.
- Implementation decisions: use the assignment's train/test split as train/validation for loss curves; keep `random_state = 42`; implement metrics and perceptron training manually; use sklearn only for `make_classification` and `train_test_split`.
- Full experiment run completed through `.venv`: generated figures, tables, `artifacts/results.json`, and `output/report.pdf`.
- Fixed Matplotlib cache location by setting `MPLCONFIGDIR` to project-local `tmp/matplotlib`; no writes to `C:\Users\Main\.matplotlib` after the fix.
- PDF report rendered successfully with Poppler into `tmp/pdfs/report-*.png`; visually checked title/base page, experiment tables, custom data page, CV/final model page, and conclusion page.
- Verification passed: `.venv\Scripts\python.exe -m unittest discover -s tests` ran 6 tests OK; `.venv\Scripts\python.exe -m compileall main.py src tests` passed.
- Key observed results: base test accuracy 0.8867, base ROC-AUC 0.9413; best 5-fold CV setting eta = 0.1, batch_size = 16, mean accuracy 0.8658 +/- 0.0209; final CV-selected model test accuracy 0.8867.
- `.gitignore` intentionally ignores `.venv`, caches, and `tmp/`, but keeps `artifacts/` and `output/` visible because they are deliverables.
