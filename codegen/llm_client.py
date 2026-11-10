"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib import error, request


class OpenAICompatibleClient:
    """Minimal client for /chat/completions style APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 12000,
        timeout: int = 1200,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = None if temperature is None else float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)

    def generate(self, prompt: str) -> str:
        """Call the chat completions API and return the assistant message text."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = self._post_json(f"{self.base_url}/chat/completions", payload)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected chat completions response shape: {response}") from exc

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {error_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM API returned non-JSON response: {response_body[:1000]}") from exc
