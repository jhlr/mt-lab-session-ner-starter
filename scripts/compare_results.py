"""Carrega as metricas das 3 tecnicas e monta uma tabela comparativa."""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "results"

TAGS = ["TIPO", "MARCA", "MODELO", "MEMORIA", "RAM", "COR", "TELA"]


def load_report(name: str) -> dict:
    data = json.loads((RESULTS_DIR / f"{name}_metrics.json").read_text(encoding="utf-8"))
    if "report" in data:
        return data["report"]
    return data


def build_overall_table() -> pd.DataFrame:
    rows = []
    for name in ["spacy", "crf", "bertimbau"]:
        report = load_report(name)
        overall = report["micro avg"]
        rows.append(
            {
                "tecnica": name,
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1-score"],
                "support": overall["support"],
            }
        )
    return pd.DataFrame(rows).set_index("tecnica").sort_values("f1", ascending=False)


def build_per_tag_table() -> pd.DataFrame:
    rows = {}
    for name in ["spacy", "crf", "bertimbau"]:
        report = load_report(name)
        rows[name] = {tag: report[tag]["f1-score"] for tag in TAGS if tag in report}
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Overall:")
    print(build_overall_table())
    print("\nPer-tag F1:")
    print(build_per_tag_table())
