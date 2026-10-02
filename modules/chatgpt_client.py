"""ChatGPT (OpenAI) request helper for the UI panel."""

from __future__ import annotations

import os
from typing import Optional

import requests
from PySide6.QtCore import QThread, Signal


DEFAULT_MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"


def resolve_api_key(explicit: Optional[str] = None) -> str:
    """Prefer an explicit key, then OPENAI_API_KEY from the environment."""
    key = (explicit or "").strip()
    if key:
        return key
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def ask_chatgpt(
    prompt: str,
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = 90.0,
) -> str:
    """Send a single user prompt and return the assistant reply text."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Input is empty")
    if not api_key:
        raise ValueError(
            "OpenAI API key missing. Set it in Settings or OPENAI_API_KEY."
        )

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        try:
            detail = response.json()
            message = (
                detail.get("error", {}).get("message")
                or detail.get("message")
                or response.text
            )
        except Exception:
            message = response.text
        raise RuntimeError(f"OpenAI error ({response.status_code}): {message}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenAI returned an empty response")
    return content


class ChatGPTWorker(QThread):
    """Run a ChatGPT request off the UI thread."""

    finished_ok = Signal(str)
    finished_error = Signal(str)

    def __init__(self, prompt: str, api_key: str, model: str = DEFAULT_MODEL, parent=None):
        super().__init__(parent)
        self._prompt = prompt
        self._api_key = api_key
        self._model = model

    def run(self) -> None:
        try:
            text = ask_chatgpt(self._prompt, self._api_key, model=self._model)
            self.finished_ok.emit(text)
        except Exception as exc:
            self.finished_error.emit(str(exc))
