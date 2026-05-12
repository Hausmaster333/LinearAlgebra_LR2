from __future__ import annotations

from src.experiments import run_all
from src.report import build_report


def main() -> None:
    run_all("artifacts")
    report_path = build_report("artifacts/results.json", "output/report.pdf")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()

