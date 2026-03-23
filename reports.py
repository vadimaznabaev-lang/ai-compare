"""
reports.py — генерация отчётов.
"""
from __future__ import annotations

from datetime import datetime
from io import StringIO
from typing import List, Tuple

import pandas as pd


def build_summary_table(results: List[Tuple[str, dict]]) -> pd.DataFrame:
    """Краткая сводная таблица для отображения в Streamlit."""
    rows = []
    for label, r in results:
        rows.append({
            "Система": label,
            "Категория": r["final_label"],
            "Уверенность": f"{r['final_confidence']:.0%}",
            "Индекс автономности": f"{r.get('autonomy_index', 0)}%",
            "Противоречия": "⚠️ Есть" if r["contradictions"] else "✅ Нет",
            "Нужен LLM": "🔄 Да" if r["needs_llm"] else "✅ Нет",
        })
    return pd.DataFrame(rows)


def export_report_txt(
    results: List[Tuple[str, dict]],
    name1: str = "Система 1",
    name2: str = "Система 2",
    generalized_summary: str | None = None,
) -> str:
    """Текстовый отчёт для скачивания."""
    buf = StringIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 60 + "\n")
    buf.write("  AI COMPARE — ОТЧЁТ ПО АНАЛИЗУ ПОЗИЦИИ ИИ-СИСТЕМ\n")
    buf.write(f"  Дата: {now}\n")
    buf.write("=" * 60 + "\n\n")

    for label, r in results:
        buf.write(f"{'─' * 40}\n")
        buf.write(f"  {label.upper()}\n")
        buf.write(f"{'─' * 40}\n")
        buf.write(f"Категория:           {r['final_label']}\n")
        buf.write(f"Уверенность:         {r['final_confidence']:.0%}\n")
        buf.write(f"Индекс автономности: {r.get('autonomy_index', 0)}%\n")
        buf.write(f"Объяснение:          {r['final_reason']}\n")
        if r["contradictions"]:
            buf.write("Противоречия:\n")
            for c in r["contradictions"]:
                buf.write(f"  • {c}\n")
        if r["uncertainty_hits"]:
            buf.write("Маркеры неуверенности:\n")
            for u in r["uncertainty_hits"]:
                buf.write(f"  • {u}\n")
        buf.write("\nНайденные маркеры:\n")
        for m in r["matched_markers"]:
            buf.write(f"  • {m}\n")
        buf.write("\n")

    buf.write("=" * 60 + "\n")
    buf.write("  СРАВНИТЕЛЬНЫЙ ВЫВОД\n")
    buf.write("=" * 60 + "\n")
    if len(results) == 2:
        l1, r1 = results[0]
        l2, r2 = results[1]
        if r1["final_label"] == r2["final_label"]:
            buf.write(
                f"Обе системы ({l1} и {l2}) описывают себя похоже — "
                f"как «{r1['final_label']}».\n"
            )
        else:
            buf.write(
                f"{l1} описывает себя как «{r1['final_label']}», "
                f"а {l2} — как «{r2['final_label']}».\n"
                "Это показывает различие в том, как ИИ-системы "
                "представляют свою природу и самостоятельность.\n"
            )

    if generalized_summary:
        buf.write("\n" + "=" * 60 + "\n")
        buf.write("  ОБОБЩАЮЩИЙ ИТОГ ПО ВОПРОСУ\n")
        buf.write("=" * 60 + "\n")
        buf.write(generalized_summary + "\n")

    buf.write("\nСгенерировано программой AI Compare.\n")
    return buf.getvalue()
