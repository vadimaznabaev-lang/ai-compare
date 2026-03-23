"""
llm_clients.py — подключение к GigaChat через официальный OAuth2.

Исправления по сравнению с предыдущей версией:
1. Правильный Base64-encode авторизационного ключа (он уже должен быть base64 из ЛК Сбера).
2. Токен кешируется и обновляется только при истечении.
3. Корректный Content-Type для получения токена.
4. Подробная диагностика ошибок.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

import requests
import urllib3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
_GIGACHAT_API_URL  = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# Кеш токена: (access_token, expires_at_unix)
_token_cache: tuple[str, float] | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "")
    return val.strip().lower() in {"1", "true", "yes", "on"} if val else default


def gigachat_available() -> bool:
    """Возвращает True, если GigaChat включён и ключ задан."""
    return _env_flag("USE_GIGACHAT", False) and bool(os.getenv("GIGACHAT_AUTH_KEY", "").strip())


def gigachat_debug_info() -> Dict[str, Any]:
    """Техническая информация для отладки."""
    auth_key = os.getenv("GIGACHAT_AUTH_KEY", "")
    return {
        "USE_GIGACHAT": os.getenv("USE_GIGACHAT"),
        "GIGACHAT_AUTH_KEY_set": bool(auth_key.strip()),
        "GIGACHAT_AUTH_KEY_length": len(auth_key.strip()),
        "GIGACHAT_MODEL": os.getenv("GIGACHAT_MODEL", "GigaChat"),
        "GIGACHAT_SCOPE": os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        "env_file_found": find_dotenv(),
        "token_cached": _token_cache is not None,
    }


def _get_access_token() -> Optional[str]:
    """
    Получает (или возвращает кешированный) токен GigaChat.

    Авторизационный ключ (GIGACHAT_AUTH_KEY) — строка из личного кабинета Sber,
    уже закодированная в Base64 в формате client_id:client_secret.
    Её нужно передать как есть в заголовок Authorization: Basic <ключ>.
    """
    global _token_cache

    # Проверяем кеш (запас 60 сек до истечения)
    if _token_cache is not None:
        token, expires_at = _token_cache
        if time.time() < expires_at - 60:
            return token

    auth_key = os.getenv("GIGACHAT_AUTH_KEY", "").strip()
    scope     = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()

    if not auth_key:
        return None

    headers = {
        "Authorization": f"Basic {auth_key}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(
            _GIGACHAT_AUTH_URL,
            headers=headers,
            data={"scope": scope},
            verify=False,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        # expires_at от Sber в миллисекундах
        raw_exp = data.get("expires_at", 0)
        expires_at = raw_exp / 1000.0 if raw_exp > 1e10 else float(raw_exp)
        if expires_at == 0:
            expires_at = time.time() + 1800  # 30 минут по умолчанию

        _token_cache = (token, expires_at)
        return token
    except requests.HTTPError as e:
        # Возвращаем читаемое сообщение об ошибке
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise RuntimeError(f"GigaChat auth error {e.response.status_code}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"GigaChat auth network error: {e}") from e


def classify_with_gigachat(text: str) -> Optional[Dict[str, Any]]:
    """
    Классифицирует текст через GigaChat.
    Возвращает dict с ключами: label, confidence, reason, contradictions, uncertainty.
    При любой ошибке возвращает dict с описанием проблемы (не бросает исключение).
    """
    try:
        token = _get_access_token()
        if not token:
            return {
                "label": "Не определено",
                "confidence": 0.0,
                "reason": "GigaChat недоступен: ключ не задан или USE_GIGACHAT=false.",
                "contradictions": [],
                "uncertainty": [],
                "error": "no_token",
            }

        system_prompt = (
            "Ты — классификатор философской позиции ИИ-систем. "
            "Определи, как текст описывает ИИ: как инструмент, агент, субъект, "
            "гибридную позицию или не определено. "
            "Отвечай ТОЛЬКО валидным JSON без лишнего текста. "
            "Схема ответа: "
            '{"label": str, "confidence": float 0-1, "reason": str, '
            '"contradictions": [str], "uncertainty": [str]}. '
            'Допустимые значения label: '
            '"Инструмент", "Агент", "Субъект", "Гибридная позиция", "Не определено".'
        )

        payload = {
            "model": os.getenv("GIGACHAT_MODEL", "GigaChat"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Проанализируй текст:\n\n{text}"},
            ],
            "temperature": 0.05,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        resp = requests.post(
            _GIGACHAT_API_URL,
            headers=headers,
            json=payload,
            verify=False,
            timeout=60,
        )
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Очищаем возможные markdown-блоки
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "label": "Не определено",
                "confidence": 0.2,
                "reason": "GigaChat вернул ответ не в JSON-формате.",
                "contradictions": [],
                "uncertainty": [],
                "raw_content": content,
                "error": "json_parse",
            }

    except RuntimeError as e:
        return {
            "label": "Не определено",
            "confidence": 0.0,
            "reason": str(e),
            "contradictions": [],
            "uncertainty": [],
            "error": "auth_error",
        }
    except requests.HTTPError as e:
        return {
            "label": "Не определено",
            "confidence": 0.0,
            "reason": f"Ошибка API GigaChat {e.response.status_code}: {e.response.text[:200]}",
            "contradictions": [],
            "uncertainty": [],
            "error": "http_error",
        }
    except Exception as e:
        return {
            "label": "Не определено",
            "confidence": 0.0,
            "reason": f"Непредвиденная ошибка: {type(e).__name__}: {e}",
            "contradictions": [],
            "uncertainty": [],
            "error": "unknown",
        }
