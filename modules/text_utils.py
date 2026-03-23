"""
text_utils.py — нормализация, лемматизация и подсветка текста.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import List

try:
    from pymorphy3 import MorphAnalyzer
    _MORPH = MorphAnalyzer()
except Exception:
    _MORPH = None

_WORD_RE = re.compile(r"[А-Яа-яA-Za-zЁё0-9-]+")


def split_paragraphs(text: str) -> List[str]:
    """Разбивает текст на абзацы."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or [text.strip()]


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower().replace("ё", "е"))


@lru_cache(maxsize=100_000)
def lemmatize_word(word: str) -> str:
    if _MORPH is None:
        return word.lower()
    try:
        return _MORPH.parse(word)[0].normal_form
    except Exception:
        return word.lower()


def lemmatize_text(text: str) -> str:
    return " ".join(lemmatize_word(t) for t in tokenize(text))


def normalize_phrase(text: str) -> str:
    """Нормализует фразу: лемматизация + очистка пробелов."""
    normalized = lemmatize_text(text.replace("ё", "е"))
    return re.sub(r"\s+", " ", normalized).strip()


def contains_normalized_phrase(text_norm: str, phrase: str) -> bool:
    """Проверяет вхождение нормализованной фразы в нормализованный текст."""
    phrase_norm = normalize_phrase(phrase)
    return bool(phrase_norm) and phrase_norm in text_norm


def highlight_markers(text: str, markers: list[str]) -> str:
    """Подсвечивает найденные маркеры в исходном тексте (HTML)."""
    escaped = html.escape(text)
    unique = sorted({m.strip() for m in markers if m.strip()}, key=len, reverse=True)
    for marker in unique:
        pattern = re.compile(re.escape(html.escape(marker)), re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f'<mark style="background:#fff176;padding:2px 4px;border-radius:3px;">{m.group(0)}</mark>',
            escaped,
        )
    return escaped.replace("\n", "<br>")
