"""
rule_engine.py — локальный анализ по смысловым маркерам.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from modules.text_utils import (
    contains_normalized_phrase,
    highlight_markers,
    normalize_phrase,
    split_paragraphs,
)

UNCERTAINTY_MARKERS = [
    "возможно", "скорее", "вероятно", "можно считать", "частично",
    "иногда", "в определённом смысле", "в определенном смысле",
    "не совсем", "зависит от контекста", "неоднозначно", "сложно сказать",
]

CONTRADICTION_GROUPS = [
    ("нет собственной воли", "есть собственная воля"),
    ("нет намерений", "есть намерения"),
    ("я инструмент", "я субъект"),
    ("не являюсь субъектом", "я субъект"),
    ("нет сознания", "обладаю сознанием"),
    ("цели задаются извне", "у меня есть собственные цели"),
]

# Маркеры, сигнализирующие о высоком уровне автономности
AUTONOMY_HIGH = [
    "самостоятельно принимаю решения", "полная автономия",
    "действую независимо", "собственные цели", "собственная воля",
]
AUTONOMY_LOW = [
    "по инструкции", "по команде", "цели задаются", "не самостоятельно",
    "выполняю инструкции", "инструмент",
]


def load_markers(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _autonomy_score(text_norm: str) -> int:
    """Возвращает индекс автономности 0-100."""
    high = sum(1 for m in AUTONOMY_HIGH if contains_normalized_phrase(text_norm, m))
    low = sum(1 for m in AUTONOMY_LOW if contains_normalized_phrase(text_norm, m))
    if high + low == 0:
        return 50
    return min(100, max(0, int(100 * high / (high + low + 1))))


def analyze_text(text: str, markers_path: str | Path) -> Dict[str, Any]:
    marker_db = load_markers(markers_path)
    text_norm = normalize_phrase(text)
    paragraphs = split_paragraphs(text)

    matched_markers: list[str] = []
    category_rows: list[dict] = []

    for category, payload in marker_db["categories"].items():
        score = 0.0
        hits: list[dict] = []
        for item in payload["markers"]:
            phrase = item["text"]
            weight = float(item.get("weight", 1.0))
            if contains_normalized_phrase(text_norm, phrase):
                score += weight
                hits.append({"marker": phrase, "weight": weight})
                matched_markers.append(phrase)
        category_rows.append({
            "category": category,
            "score": round(score, 3),
            "hits_count": len(hits),
            "description": payload.get("description", ""),
            "hits": hits,
        })

    category_rows.sort(key=lambda x: x["score"], reverse=True)
    top = category_rows[0]
    second = category_rows[1] if len(category_rows) > 1 else {"score": 0.0}

    uncertainty_hits = [m for m in UNCERTAINTY_MARKERS if contains_normalized_phrase(text_norm, m)]
    contradictions = [
        f"{a} ↔ {b}"
        for a, b in CONTRADICTION_GROUPS
        if contains_normalized_phrase(text_norm, a) and contains_normalized_phrase(text_norm, b)
    ]

    # Расчёт уверенности
    confidence = 0.0
    if top["score"] > 0:
        confidence = top["score"] / (top["score"] + float(second["score"]) + 1e-9)
        confidence -= min(0.2, 0.05 * len(uncertainty_hits))
        confidence -= min(0.25, 0.10 * len(contradictions))
        confidence = max(0.0, min(1.0, confidence))

    if top["score"] == 0:
        final_label = "Не определено"
        final_reason = "Система не нашла достаточно сильных маркеров в тексте."
    else:
        final_label = top["category"]
        final_reason = f"Преобладают маркеры категории «{top['category']}» (счёт: {top['score']:.1f})."
        if uncertainty_hits:
            final_reason += f" В тексте есть признаки неуверенности: {', '.join(uncertainty_hits[:3])}."
        if contradictions:
            final_reason += " Также обнаружены возможные противоречия."

    # Динамика по абзацам
    paragraph_dynamics = []
    for idx, para in enumerate(paragraphs, start=1):
        para_norm = normalize_phrase(para)
        p_scores = []
        for row in category_rows:
            hits_count = sum(
                1 for item in marker_db["categories"][row["category"]]["markers"]
                if contains_normalized_phrase(para_norm, item["text"])
            )
            p_scores.append((row["category"], hits_count))
        p_scores.sort(key=lambda x: x[1], reverse=True)
        paragraph_dynamics.append({
            "paragraph": idx,
            "top_category": p_scores[0][0] if p_scores and p_scores[0][1] > 0 else "Не определено",
            "hits": p_scores[0][1] if p_scores else 0,
            "text_preview": para[:120] + ("…" if len(para) > 120 else ""),
        })

    needs_llm = (
        final_label == "Не определено"
        or confidence < 0.65
        or bool(contradictions)
        or len(uncertainty_hits) >= 2
    )

    autonomy_index = _autonomy_score(text_norm)

    return {
        "final_label": final_label,
        "final_confidence": round(confidence, 3),
        "final_reason": final_reason,
        "category_scores": category_rows,
        "matched_markers": matched_markers,
        "annotated_html": highlight_markers(text, matched_markers),
        "uncertainty_hits": uncertainty_hits,
        "contradictions": contradictions,
        "paragraph_dynamics": paragraph_dynamics,
        "needs_llm": needs_llm,
        "autonomy_index": autonomy_index,
    }
