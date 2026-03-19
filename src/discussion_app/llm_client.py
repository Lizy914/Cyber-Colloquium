from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import requests

from .models import AttachmentPayload, ProviderConfig


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        attachments: list[AttachmentPayload] | None = None,
        max_tokens: int = 800,
        max_continuations: int = 1,
    ) -> str:
        request_url = self._resolve_request_url()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            self._build_user_message(user_prompt, attachments or []),
        ]

        chunks: list[str] = []
        remaining_continuations = max(0, max_continuations)
        current_tokens = max_tokens

        while True:
            data = self._post_chat(request_url, messages, current_tokens)
            content = self._extract_content(data).strip()
            if content:
                chunks.append(content)

            finish_reason = self._extract_finish_reason(data)
            if finish_reason != "length" or remaining_continuations <= 0:
                break

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous answer was cut off by the output limit. Continue exactly from where you stopped. Do not repeat earlier content and do not add extra explanation.",
                }
            )
            remaining_continuations -= 1
            current_tokens = max(320, min(current_tokens, 1200))

        return "\n\n".join(chunk for chunk in chunks if chunk).strip()

    def _post_chat(self, request_url: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        payload = {
            "model": self.provider.model,
            "messages": messages,
            "temperature": self.provider.temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.provider.api_key}",
        }
        try:
            response = requests.post(
                request_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
        except requests.Timeout as exc:
            raise LLMError(f"{self.provider.name} request timed out at {request_url}: {exc}") from exc
        except requests.RequestException as exc:
            raise LLMError(f"{self.provider.name} request failed at {request_url}: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(self._format_http_error(response.status_code, response.text, request_url))
        return response.json()

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.provider.name} returned an unrecognized response payload: {data}") from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            return "\n".join(texts)
        return str(content)

    def _extract_finish_reason(self, data: dict[str, Any]) -> str | None:
        try:
            return data["choices"][0].get("finish_reason")
        except Exception:  # noqa: BLE001
            return None

    def _resolve_request_url(self) -> str:
        raw = self.provider.base_url.strip().rstrip("/")
        if not raw:
            return raw
        if raw.endswith("/chat/completions") or raw.endswith("/responses"):
            return raw

        parsed = urlparse(raw)
        path = parsed.path.rstrip("/")
        host = parsed.netloc.lower()

        if host == "api.openai.com" and path == "/v1":
            return f"{raw}/chat/completions"
        if host == "api.moonshot.cn" and path == "/v1":
            return f"{raw}/chat/completions"
        if host == "api.deepseek.com" and path in {"", "/beta"}:
            return f"{raw}/chat/completions"
        if host == "ark.cn-beijing.volces.com" and path == "/api/v3":
            return f"{raw}/chat/completions"
        if "dashscope" in host and path == "/compatible-mode/v1":
            return f"{raw}/chat/completions"
        if host in {"api.minimax.io", "api.minimax.com"} and path == "/v1":
            return f"{raw}/chat/completions"
        if host == "open.bigmodel.cn" and path == "/api/paas/v4":
            return f"{raw}/chat/completions"

        if path.endswith("/v1") or path.endswith("/v3") or path.endswith("/v4"):
            return f"{raw}/chat/completions"
        return raw

    def _format_http_error(self, status_code: int, body: str, request_url: str) -> str:
        text = (body or "").strip()
        text_excerpt = text[:400]
        text_lower = text.lower()

        if status_code == 404:
            hint = self._build_404_hint(request_url)
            if text_excerpt:
                return f"{self.provider.name} API error 404 at {request_url}: {text_excerpt} {hint}".strip()
            return f"{self.provider.name} API error 404 at {request_url}. {hint}".strip()

        if status_code == 400 and "range of input length should be [1, 3072]" in text_lower:
            return (
                f"{self.provider.name} API error 400 at {request_url}: input is too long."
                " Reduce the context, shorten the question, or trim attachment content."
            )

        if status_code == 401 and "invalid api key" in text_lower:
            return (
                f"{self.provider.name} API error 401 at {request_url}: API key is invalid."
                " Check that the provider, network entrypoint, and key belong together."
            )

        if status_code == 402 and "insufficient balance" in text_lower:
            return (
                f"{self.provider.name} API error 402 at {request_url}: account balance is insufficient."
                " Verify the corresponding provider balance or quota."
            )

        if status_code == 429 and "insufficient_quota" in text_lower:
            return (
                f"{self.provider.name} API error 429 at {request_url}: quota is exhausted or billing is not enabled."
                " Check billing or quota on the corresponding platform."
            )

        return f"{self.provider.name} API error {status_code} at {request_url}: {text_excerpt}"

    def _build_404_hint(self, request_url: str) -> str:
        host = urlparse(request_url).netloc.lower()
        provider_name = self.provider.name.lower()

        if "minimax" in provider_name or "minimax" in host:
            return "MiniMax should use the correct .com or .io domain for your current network path, plus /v1 or /chat/completions."
        if "dashscope" in host or "qwen" in provider_name:
            return "Qwen compatible mode should use https://dashscope.aliyuncs.com/compatible-mode/v1 or its /chat/completions path."
        if "deepseek" in host or "deepseek" in provider_name:
            return "DeepSeek should use https://api.deepseek.com or its /chat/completions path."
        if "moonshot" in host or "kimi" in provider_name:
            return "Kimi should use https://api.moonshot.cn/v1 or its /chat/completions path."
        if "volces.com" in host or "doubao" in provider_name or "ark" in provider_name:
            return "Doubao/Ark compatible mode usually uses https://ark.cn-beijing.volces.com/api/v3 or its /chat/completions path, and often requires an endpoint ID such as ep-xxxx as the model value."
        if "bigmodel" in host or "glm" in provider_name:
            return "GLM should use https://open.bigmodel.cn/api/paas/v4 or its /chat/completions path."
        return "Check whether Base URL points to the provider's official chat endpoint."

    def _build_user_message(self, prompt: str, attachments: list[AttachmentPayload]) -> dict[str, Any]:
        if not attachments or not any(item.kind == "image" for item in attachments):
            return {"role": "user", "content": prompt}

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for attachment in attachments:
            if attachment.kind == "image" and self.provider.supports_vision:
                content.append({"type": "image_url", "image_url": {"url": attachment.content}})
        return {"role": "user", "content": content}
