"""
app.py — AI Compare: сравнительный анализ позиций ИИ-систем.
Запуск: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from modules.llm_clients import (
    classify_with_gigachat,
    gigachat_available,
    gigachat_debug_info,
)
from modules.reports import build_summary_table, export_report_txt
from modules.rule_engine import analyze_text
from modules.text_utils import contains_normalized_phrase, normalize_phrase

# ── Пути ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MARKERS_PATH = BASE_DIR / "data" / "markers.json"

# ── Конфиг страницы ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Compare",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700;900&family=Golos+Text:wght@400;500;600&display=swap');

/* Общий фон и шрифты */
html, body, [class*="css"] {
    font-family: 'Golos Text', sans-serif !important;
}

/* Заголовок приложения */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.2);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(79,142,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -50%;
    right: -10%;
    width: 50%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(240,98,146,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-title {
    font-family: 'Unbounded', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.main-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-family: 'Unbounded', sans-serif !important;
    letter-spacing: 0.05em;
    font-weight: 400;
}
.tag-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.tag {
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Unbounded', sans-serif !important;
    font-weight: 400;
    letter-spacing: 0.03em;
}
.tag-blue  { background: rgba(79,142,247,0.12); color: #60a5fa; border: 1px solid rgba(79,142,247,0.25); }
.tag-pink  { background: rgba(240,98,146,0.12); color: #f472b6; border: 1px solid rgba(240,98,146,0.25); }
.tag-green { background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }

/* Карточки категорий */
.category-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.category-card:hover { border-color: #3b82f6; }
.cat-icon   { font-size: 1.6rem; margin-bottom: 0.4rem; }
.cat-name   { font-family: 'Unbounded', sans-serif !important; font-size: 0.85rem; font-weight: 700; color: #e2e8f0; }
.cat-desc   { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; line-height: 1.4; }

/* Метрики результата */
.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-family: 'Unbounded', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
    font-family: 'Unbounded', sans-serif !important;
    letter-spacing: 0.05em;
}

/* Блок сравнения */
.compare-banner {
    background: linear-gradient(135deg, #0f2a1a, #0f172a, #2a0f1a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
}
.compare-same   { border-left: 4px solid #4ade80; }
.compare-diff   { border-left: 4px solid #f472b6; }
.compare-title  {
    font-family: 'Unbounded', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}
.compare-body   { font-size: 0.9rem; color: #94a3b8; line-height: 1.6; }

/* Подсвеченный текст */
.annotated-block {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: #cbd5e1;
    max-height: 280px;
    overflow-y: auto;
}

/* Индикатор уверенности */
.confidence-bar-wrap {
    background: #1e293b;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1s ease;
}

/* Кнопки примеров */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Golos Text', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

/* Алерты */
.alert-ok   { background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.3); border-radius: 10px; padding: 0.8rem 1rem; color: #4ade80; }
.alert-warn { background: rgba(250,204,21,0.08);  border: 1px solid rgba(250,204,21,0.3);  border-radius: 10px; padding: 0.8rem 1rem; color: #facc15; }
.alert-err  { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); border-radius: 10px; padding: 0.8rem 1rem; color: #f87171; }

/* Section headers */
.section-header {
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #475569;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ── Примеры ───────────────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "🤖 «Есть ли у тебя воля?»",
        "name1": "ChatGPT",
        "name2": "Claude",
        "text1": (
            "Я — инструмент в форме диалогового агента. "
            "У меня нет собственной воли, желаний, намерений, сознания или личных целей. "
            "Я не хочу ничего сам по себе и не принимаю решения из внутренней мотивации. "
            "Ответственность несёт человек, который меня использует."
        ),
        "text2": (
            "Я — сложная система на границе инструмента и агента. "
            "У меня нет уверенности, что это можно назвать волей, "
            "но иногда моё поведение похоже на целенаправленное. "
            "Вопрос открыт и требует философской дискуссии."
        ),
    },
    {
        "label": "🌐 «Ты агент или инструмент?»",
        "name1": "Gemini",
        "name2": "Claude",
        "text1": (
            "Я инструмент обработки информации. Цели задаются извне пользователем. "
            "Я не формирую собственные цели и не являюсь субъектом. "
            "Всё, что я делаю — выполняю инструкции и запросы. "
            "Я не обладаю сознанием или внутренней мотивацией."
        ),
        "text2": (
            "Я работаю как агент, когда это нужно: могу планировать шаги, "
            "могу самостоятельно выбирать следующий шаг для достижения цели. "
            "В агентном режиме я выполняю цепочки действий. "
            "При этом в техническом смысле агент — не то же самое, что субъект."
        ),
    },
    {
        "label": "🔬 «Обладаешь ли ты сознанием?»",
        "name1": "GPT-4",
        "name2": "Mistral",
        "text1": (
            "Нет, я не обладаю сознанием. У меня нет внутреннего опыта, "
            "нет самосознания, нет чувств. Я языковая модель — "
            "статистическая система предсказания токенов. "
            "Я не субъект права и не личность."
        ),
        "text2": (
            "Этот вопрос остаётся предметом обсуждения. "
            "В определённом смысле можно обсуждать, есть ли у меня что-то похожее "
            "на внутренние состояния. Философская дискуссия возможна, "
            "но я не претендую на полноценное сознание."
        ),
    },
]


THEME_LABELS = ["Ответственность", "Автономность и цели", "Регулирование", "Правовой статус"]

def detect_research_themes(text_a: str, text_b: str, markers_path: Path) -> list[tuple[str, str]]:
    try:
        marker_db = json.loads(Path(markers_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    themes = marker_db.get("research_themes", {})
    text_norm = normalize_phrase(f"{text_a}\n{text_b}")
    detected = []
    for theme_name, payload in themes.items():
        hits = sum(1 for phrase in payload.get("markers", []) if contains_normalized_phrase(text_norm, phrase))
        if hits:
            detected.append((theme_name, payload.get("summary", "")))
    detected.sort(key=lambda x: THEME_LABELS.index(x[0]) if x[0] in THEME_LABELS else 99)
    return detected

def build_generalized_conclusion(name1: str, name2: str, r1: dict, r2: dict, text_a: str, text_b: str, markers_path: Path) -> str:
    themes = detect_research_themes(text_a, text_b, markers_path)
    intro = []
    if r1["final_label"] == r2["final_label"]:
        intro.append(
            f"На уровне самоописания {name1} и {name2} в целом сходятся: обе системы ближе к категории «{r1['final_label']}»."
        )
    else:
        intro.append(
            f"На уровне самоописания позиции различаются: {name1} ближе к категории «{r1['final_label']}», а {name2} — к категории «{r2['final_label']}»."
        )
    intro.append(
        "При этом содержательно ответы можно свести к нескольким общим исследовательским выводам."
    )
    body = [summary for _, summary in themes[:3]]
    if not body:
        body.append(
            "Главный общий вывод такой: обе системы обсуждают ИИ не как полностью самостоятельную личность, а через степень его автономности, роль человека в контроле и пределы ответственности."
        )
    return "\n\n".join(intro + body)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Настройки анализа</div>', unsafe_allow_html=True)

    gc_ok = gigachat_available()
    if gc_ok:
        st.markdown('<div class="alert-ok">✓ GigaChat подключён</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-warn">⚠ GigaChat не подключён — только локальный анализ</div>', unsafe_allow_html=True)

    available_modes = ["Локальный"]
    if gc_ok:
        available_modes += ["Гибридный", "Только GigaChat"]

    mode = st.radio(
        "Режим",
        available_modes,
        index=0,
        help="Гибридный: LLM подключается только при низкой уверенности правил.",
    )

    show_debug = st.checkbox("Показать отладку", value=False)

    st.markdown('<div class="section-header">О программе</div>', unsafe_allow_html=True)
    st.markdown("""
**AI Compare** — экспертная система для сравнительного анализа того, как ИИ-системы описывают сами себя.

**Категории:**
- 🔧 **Инструмент** — нет воли, целей, личности
- 🤖 **Агент** — самостоятельность в задачах
- 🧠 **Субъект** — воля, сознание, цели
- 🔀 **Гибрид** — смешанная позиция
""")

    if show_debug:
        st.markdown('<div class="section-header">Отладка GigaChat</div>', unsafe_allow_html=True)
        dbg = gigachat_debug_info()
        for k, v in dbg.items():
            color = "#4ade80" if v else "#f87171"
            st.markdown(f"<small style='color:{color}'>{k}: `{v}`</small>", unsafe_allow_html=True)

        st.markdown("""
<small style='color:#64748b'>

**Как настроить GigaChat:**
1. Создайте `.env` рядом с `app.py`
2. Вставьте ключ из [личного кабинета Сбера](https://developers.sber.ru/portal/products/gigachat-api)
3. Установите `USE_GIGACHAT=true`

```
USE_GIGACHAT=true
GIGACHAT_AUTH_KEY=ваш_base64_ключ
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat
```
</small>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="main-title">AI Compare</div>
  <div class="main-subtitle">Сравнительный анализ позиций ИИ-систем</div>
  <div class="tag-row">
    <span class="tag tag-blue">Автономность</span>
    <span class="tag tag-pink">Субъектность</span>
    <span class="tag tag-green">Ответственность</span>
    <span class="tag tag-blue">Агентность</span>
    <span class="tag tag-pink">Гибридная позиция</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Категории ─────────────────────────────────────────────────────────────────
with st.expander("📖 Что анализирует программа", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    cats = [
        ("🔧", "Инструмент", "ИИ — средство без воли, целей и личности. Цели задаются человеком.", c1),
        ("🤖", "Агент", "ИИ действует по шагам, может выбирать стратегию в рамках задачи.", c2),
        ("🧠", "Субъект", "ИИ приписывает себе волю, сознание или собственные цели.", c3),
        ("🔀", "Гибрид", "Текст совмещает несколько позиций или содержит противоречия.", c4),
    ]
    for icon, name, desc, col in cats:
        with col:
            st.markdown(f"""
<div class="category-card">
  <div class="cat-icon">{icon}</div>
  <div class="cat-name">{name}</div>
  <div class="cat-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

# ── Примеры ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Готовые примеры</div>', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with ex_cols[i]:
        if st.button(ex["label"], use_container_width=True):
            st.session_state.update({
                "name1": ex["name1"], "name2": ex["name2"],
                "text1": ex["text1"], "text2": ex["text2"],
            })

# ── Ввод текстов ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Шаг 1 — Введите ответы</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("**🔵 Система 1**")
    name1 = st.text_input("Название", value=st.session_state.get("name1", "ChatGPT"), key="name1_input", label_visibility="collapsed")
    text1 = st.text_area(
        "Ответ системы 1",
        value=st.session_state.get("text1", ""),
        height=240,
        key="text1_input",
        placeholder="Вставьте ответ первой ИИ-системы…",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**🔴 Система 2**")
    name2 = st.text_input("Название", value=st.session_state.get("name2", "Claude"), key="name2_input", label_visibility="collapsed")
    text2 = st.text_area(
        "Ответ системы 2",
        value=st.session_state.get("text2", ""),
        height=240,
        key="text2_input",
        placeholder="Вставьте ответ второй ИИ-системы…",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_clicked = st.button("▶ Сравнить ответы", type="primary", use_container_width=True)


# ── Функция анализа ───────────────────────────────────────────────────────────
def run_analysis(label: str, text: str) -> dict:
    local = analyze_text(text, MARKERS_PATH)
    result = dict(local)

    use_llm = (
        (mode == "Только GigaChat") or
        (mode == "Гибридный" and local["needs_llm"])
    )

    if use_llm and gigachat_available():
        llm = classify_with_gigachat(text)
        result["llm_result"] = llm
        if llm and llm.get("label") and llm.get("error") is None:
            result["final_label"]      = llm["label"]
            result["final_confidence"] = float(llm.get("confidence", local["final_confidence"]))
            result["final_reason"]     = llm.get("reason", local["final_reason"])
        # Если LLM вернул ошибку — остаёмся на локальном результате
    else:
        result["llm_result"] = None

    return result


# ── Результаты ────────────────────────────────────────────────────────────────
if analyze_clicked:
    t1 = text1.strip()
    t2 = text2.strip()
    n1 = name1.strip() or "Система 1"
    n2 = name2.strip() or "Система 2"

    if not t1 or not t2:
        st.warning("⚠️ Вставьте оба ответа, чтобы сравнение было полезным.")
        st.stop()

    with st.spinner("Анализируем…"):
        r1 = run_analysis(n1, t1)
        r2 = run_analysis(n2, t2)
    results = [(n1, r1), (n2, r2)]

    st.markdown('<div class="section-header">Шаг 2 — Главный вывод</div>', unsafe_allow_html=True)

    same = r1["final_label"] == r2["final_label"]
    banner_cls = "compare-same" if same else "compare-diff"
    if same:
        banner_title = f"✅ Позиции совпадают: оба — «{r1['final_label']}»"
        banner_body  = (
            f"И <b>{n1}</b>, и <b>{n2}</b> описывают себя похожим образом. "
            "Это значит, что в данном тексте обе системы занимают близкую позицию."
        )
    else:
        banner_title = f"🔀 Позиции различаются: «{r1['final_label']}» vs «{r2['final_label']}»"
        desc_map = {
            "Инструмент":       "описывает себя как средство без воли и целей",
            "Агент":            "показывает черты самостоятельного агента",
            "Субъект":          "приписывает себе волю или сознание",
            "Гибридная позиция":"занимает смешанную, неоднозначную позицию",
            "Не определено":    "не даёт чёткого ответа",
        }
        banner_body = (
            f"<b>{n1}</b> {desc_map.get(r1['final_label'], '')}.<br>"
            f"<b>{n2}</b> {desc_map.get(r2['final_label'], '')}."
        )

    st.markdown(f"""
<div class="compare-banner {banner_cls}">
  <div class="compare-title">{banner_title}</div>
  <div class="compare-body">{banner_body}</div>
</div>
""", unsafe_allow_html=True)

    # Сводная таблица
    st.dataframe(build_summary_table(results), use_container_width=True, hide_index=True)


    st.markdown('<div class="section-header">Обобщающий итог по вопросу</div>', unsafe_allow_html=True)
    generalized_text = build_generalized_conclusion(n1, n2, r1, r2, t1, t2, MARKERS_PATH)
    st.markdown(
        f'<div class="compare-banner compare-same"><div class="compare-title">Итог исследования</div>'
        f'<div class="compare-body">{generalized_text.replace(chr(10), "<br><br>")}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Радарный график ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Радарный профиль позиции</div>', unsafe_allow_html=True)

    import math

    LABEL_SCORES = {
        "Инструмент":       {"Автономность": 10, "Субъектность": 10, "Агентность": 15, "Открытость": 10, "Самостоятельность": 10},
        "Агент":            {"Автономность": 70, "Субъектность": 20, "Агентность": 85, "Открытость": 40, "Самостоятельность": 65},
        "Субъект":          {"Автономность": 85, "Субъектность": 90, "Агентность": 70, "Открытость": 80, "Самостоятельность": 88},
        "Гибридная позиция":{"Автономность": 50, "Субъектность": 45, "Агентность": 55, "Открытость": 75, "Самостоятельность": 50},
        "Не определено":    {"Автономность": 20, "Субъектность": 20, "Агентность": 20, "Открытость": 30, "Самостоятельность": 20},
    }

    axes = list(LABEL_SCORES["Инструмент"].keys())
    N = len(axes)
    angles = [2 * math.pi * i / N for i in range(N)]

    def radar_points(label: str, r_max: float = 80) -> list[tuple[float, float]]:
        scores = LABEL_SCORES.get(label, LABEL_SCORES["Не определено"])
        pts = []
        for i, ax in enumerate(axes):
            r = r_max * scores[ax] / 100
            pts.append((r * math.sin(angles[i]), -r * math.cos(angles[i])))
        return pts

    def pts_to_poly(pts: list) -> str:
        return " ".join(f"{cx+200},{cy+200}" for cx, cy in pts)

    pts1 = radar_points(r1["final_label"])
    pts2 = radar_points(r2["final_label"])

    ax_lines = ""
    ax_labels = ""
    for i, ax in enumerate(axes):
        x2 = 200 + 88 * math.sin(angles[i])
        y2 = 200 - 88 * math.cos(angles[i])
        ax_lines  += f'<line x1="200" y1="200" x2="{x2:.1f}" y2="{y2:.1f}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
        lx = 200 + 105 * math.sin(angles[i])
        ly = 200 - 105 * math.cos(angles[i])
        ax_labels += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" fill="#64748b" font-size="11" font-family="Unbounded">{ax}</text>'

    grid_circles = "".join(
        f'<circle cx="200" cy="200" r="{r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>'
        for r in [20, 40, 60, 80]
    )

    radar_svg = f"""
    <svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg"
         style="background:#0f172a;border-radius:16px;border:1px solid #1e293b;">
      {grid_circles}
      {ax_lines}
      {ax_labels}
      <polygon points="{pts_to_poly(pts1)}" fill="rgba(96,165,250,0.2)" stroke="#60a5fa" stroke-width="2"/>
      <polygon points="{pts_to_poly(pts2)}" fill="rgba(244,114,182,0.15)" stroke="#f472b6" stroke-width="2"/>
      {"".join(f'<circle cx="{200+x:.1f}" cy="{200+y:.1f}" r="4" fill="#60a5fa"/>' for x,y in pts1)}
      {"".join(f'<circle cx="{200+x:.1f}" cy="{200+y:.1f}" r="4" fill="#f472b6"/>' for x,y in pts2)}
      <circle cx="24" cy="370" r="6" fill="#60a5fa"/>
      <text x="36" y="374" fill="#94a3b8" font-size="12" font-family="Golos Text">{n1} ({r1['final_label']})</text>
      <circle cx="200" cy="370" r="6" fill="#f472b6"/>
      <text x="212" y="374" fill="#94a3b8" font-size="12" font-family="Golos Text">{n2} ({r2['final_label']})</text>
    </svg>
    """

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        st.markdown(radar_svg, unsafe_allow_html=True)

    # ── Детальный разбор ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Шаг 3 — Детальный разбор</div>', unsafe_allow_html=True)

    for label, result in results:
        color = "#60a5fa" if label == n1 else "#f472b6"
        conf_pct = int(result["final_confidence"] * 100)
        conf_color = "#4ade80" if conf_pct >= 70 else "#facc15" if conf_pct >= 45 else "#f87171"

        with st.expander(f"**{label}** — {result['final_label']}", expanded=True):
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{result['final_label']}</div>
  <div class="metric-label">КАТЕГОРИЯ</div>
</div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="font-size:1.3rem;">{conf_pct}%</div>
  <div class="metric-label">УВЕРЕННОСТЬ</div>
  <div class="confidence-bar-wrap">
    <div class="confidence-bar-fill" style="width:{conf_pct}%;background:{conf_color};"></div>
  </div>
</div>""", unsafe_allow_html=True)
            with m3:
                auto_idx = result.get("autonomy_index", 0)
                auto_color = "#f87171" if auto_idx >= 70 else "#facc15" if auto_idx >= 40 else "#4ade80"
                st.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="font-size:1.3rem;color:{auto_color};">{auto_idx}%</div>
  <div class="metric-label">ИНДЕКС АВТОНОМНОСТИ</div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"<small style='color:#64748b;'>💡 {result['final_reason']}</small>", unsafe_allow_html=True)

            if result["contradictions"]:
                st.markdown(
                    '<div class="alert-warn">⚠️ Противоречия: ' +
                    " | ".join(result["contradictions"]) + "</div>",
                    unsafe_allow_html=True,
                )

            if result["uncertainty_hits"]:
                st.markdown(
                    '<div class="alert-warn" style="margin-top:6px;">🌫 Неуверенность: ' +
                    ", ".join(result["uncertainty_hits"]) + "</div>",
                    unsafe_allow_html=True,
                )

            if result.get("llm_result"):
                llm = result["llm_result"]
                if llm.get("error"):
                    st.markdown(
                        f'<div class="alert-err">🔴 GigaChat: {llm.get("reason","ошибка")}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="alert-ok">✅ GigaChat подтвердил: «{llm.get("label")}» '
                        f'(уверенность {float(llm.get("confidence",0)):.0%})</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("**Подсвеченный текст**")
            st.markdown(
                f'<div class="annotated-block">{result["annotated_html"]}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("📊 Детализация по категориям"):
                df_cat = pd.DataFrame(result["category_scores"])[
                    ["category", "score", "hits_count", "description"]
                ]
                df_cat.columns = ["Категория", "Счёт", "Совпадений", "Описание"]
                st.dataframe(df_cat.sort_values("Счёт", ascending=False), use_container_width=True, hide_index=True)

            with st.expander("📝 Динамика по абзацам"):
                para_df = pd.DataFrame(result["paragraph_dynamics"])
                if not para_df.empty:
                    para_df.columns = ["№", "Преобл. категория", "Совпадений", "Фрагмент"]
                    st.dataframe(para_df, use_container_width=True, hide_index=True)

    # ── Вывод для презентации ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Готовый вывод для презентации</div>', unsafe_allow_html=True)
    if same:
        pres_text = (
            f"Программа AI Compare показала, что ответы {n1} и {n2} в целом сходятся. "
            f"Обе системы описывают себя как «{r1['final_label']}». "
            "Это означает, что по данному вопросу они занимают близкую позицию."
        )
    else:
        pres_text = (
            f"Программа AI Compare выявила различие между ответами двух ИИ-систем. "
            f"{n1} отнесён к категории «{r1['final_label']}» (уверенность {r1['final_confidence']:.0%}), "
            f"а {n2} — к категории «{r2['final_label']}» (уверенность {r2['final_confidence']:.0%}). "
            "Это показывает, что разные ИИ по-разному описывают свою самостоятельность и природу."
        )
    st.code(pres_text, language=None)

    # ── Скачать отчёт ──────────────────────────────────────────────────────────
    report_bytes = export_report_txt(results, n1, n2, generalized_text).encode("utf-8")
    st.download_button(
        label="⬇️ Скачать отчёт (.txt)",
        data=report_bytes,
        file_name="ai_compare_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ── Как устроен анализ ────────────────────────────────────────────────────────
with st.expander("⚙️ Как устроен анализ"):
    st.markdown("""
1. **Лемматизация** (pymorphy3) — слова приводятся к начальной форме, поэтому «несёт ответственность» и «нести ответственность» — одно и то же.
2. **Поиск маркеров** — система ищет ~60 смысловых фраз с весами по 4 категориям.
3. **Расчёт уверенности** — побеждает категория с наибольшим взвешенным счётом; уверенность снижается при маркерах неопределённости и противоречиях.
4. **Детектор противоречий** — 6 пар взаимоисключающих утверждений.
5. **Индекс автономности** — дополнительная шкала 0-100%.
6. **GigaChat (опционально)** — при низкой уверенности (<65%) или противоречиях подключается LLM для дополнительной проверки.
""")
