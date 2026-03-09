from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import DUTY_OPTIONS, EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, LITERATURE_DUTY, REPORT_DUTY, ProviderConfig


CONFIG_PATH = Path("app_config.json")

LEGACY_DUTY_MAP = {
    "\u603b\u8d1f\u8d23": LEAD_DUTY,
    "\u4e3b\u6301\u4eba": HOST_DUTY,
    "\u4e13\u5bb6\u7ec4": EXPERT_DUTY,
    "\u7efc\u8ff0\u4e13\u5bb6": LITERATURE_DUTY,
    "\u7edf\u7a3f\u4eba": REPORT_DUTY,
    "\u4e13\u5bb6": EXPERT_DUTY,
    "\u6587\u732e\u7efc\u8ff0": LITERATURE_DUTY,
    "???": LEAD_DUTY,
    "????": LITERATURE_DUTY,
}

DEFAULT_SPECIALTIES = {
    LEAD_DUTY: "Task decomposition, delegation, and quality control",
    HOST_DUTY: "Coordination, workflow pacing, and meeting control",
    LITERATURE_DUTY: "Literature review, related-work mapping, and research context",
    REPORT_DUTY: "Live logging, structured synthesis, and final report writing",
    "MiniMax": "Critical review, gap finding, and concise revision",
    "Qwen3-Max": "Literature synthesis, high-level reasoning, and framework design",
    "Qwen-Math": "Mathematical derivation, quantitative checking, and rigor review",
    "DeepSeek": "Counterexamples, failure analysis, and argument stress-testing",
    "GLM": "Live logging, report integration, and editorial polishing",
}

MOJIBAKE_HINTS = "銆鍏鍒缁浣鐮娴鏃妭绔鏌绛閫楠诲琛粨鍐欎綔鍙崐"


def default_providers() -> list[ProviderConfig]:
    return [
        ProviderConfig(
            name="Kimi Lead",
            model="moonshot-v1-8k",
            base_url="https://api.moonshot.cn/v1",
            enabled=False,
            supports_vision=False,
            duty=LEAD_DUTY,
            specialty=DEFAULT_SPECIALTIES[LEAD_DUTY],
        ),
        ProviderConfig(
            name="Kimi Host",
            model="moonshot-v1-8k",
            base_url="https://api.moonshot.cn/v1",
            enabled=False,
            supports_vision=False,
            duty=HOST_DUTY,
            specialty=DEFAULT_SPECIALTIES[HOST_DUTY],
        ),
        ProviderConfig(
            name="MiniMax",
            model="MiniMax-M2.5",
            base_url="https://api.minimax.io/v1",
            supports_vision=False,
            duty=EXPERT_DUTY,
            specialty=DEFAULT_SPECIALTIES["MiniMax"],
        ),
        ProviderConfig(
            name="Qwen3-Max",
            model="qwen3-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            supports_vision=True,
            duty=EXPERT_DUTY,
            specialty=DEFAULT_SPECIALTIES["Qwen3-Max"],
        ),
        ProviderConfig(
            name="Qwen-Math",
            model="qwen-math-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            supports_vision=False,
            duty=EXPERT_DUTY,
            specialty=DEFAULT_SPECIALTIES["Qwen-Math"],
        ),
        ProviderConfig(
            name="Doubao Review",
            model="",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            enabled=False,
            supports_vision=False,
            duty=LITERATURE_DUTY,
            specialty=DEFAULT_SPECIALTIES[LITERATURE_DUTY],
        ),
        ProviderConfig(
            name="GLM Reporter",
            model="glm-4.5",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            supports_vision=True,
            duty=REPORT_DUTY,
            specialty=DEFAULT_SPECIALTIES[REPORT_DUTY],
        ),
        ProviderConfig(
            name="DeepSeek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            supports_vision=False,
            duty=EXPERT_DUTY,
            specialty=DEFAULT_SPECIALTIES["DeepSeek"],
        ),
    ]


def load_providers() -> list[ProviderConfig]:
    if not CONFIG_PATH.exists():
        providers = default_providers()
        save_providers(providers)
        return providers

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    providers = [ProviderConfig(**item) for item in raw.get("providers", []) if isinstance(item, dict)]
    providers = _migrate_legacy_providers(providers)
    save_providers(providers)
    return providers


def save_providers(providers: list[ProviderConfig]) -> None:
    payload = {"providers": [asdict(provider) for provider in providers]}
    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _migrate_legacy_providers(providers: list[ProviderConfig]) -> list[ProviderConfig]:
    migrated: list[ProviderConfig] = []
    for provider in providers:
        provider.name = (provider.name or "").strip() or _fallback_name(provider)
        provider.model = (provider.model or "").strip()
        provider.base_url = _normalize_base_url((provider.base_url or "").strip())
        provider.api_key = (provider.api_key or "").strip()
        provider.duty = _infer_duty(provider)
        provider.specialty = _normalize_specialty(provider)
        migrated.append(provider)
    return migrated or default_providers()


def _fallback_name(provider: ProviderConfig) -> str:
    lowered = f"{provider.model} {provider.base_url}".lower()
    if "moonshot" in lowered or "kimi" in lowered:
        return "Kimi Lead"
    if "minimax" in lowered:
        return "MiniMax"
    if "dashscope" in lowered and "math" in lowered:
        return "Qwen-Math"
    if "dashscope" in lowered:
        return "Qwen3-Max"
    if "deepseek" in lowered:
        return "DeepSeek"
    if "volces" in lowered or "doubao" in lowered or "ark" in lowered:
        return "Doubao Review"
    if "glm" in lowered or "bigmodel" in lowered:
        return "GLM Reporter"
    return "New Role"


def _normalize_base_url(base_url: str) -> str:
    lowered = base_url.lower()
    if not base_url:
        return base_url
    if "operator.las.cn-beijing.volces.com" in lowered:
        return "https://ark.cn-beijing.volces.com/api/v3"
    if "ark.cn-beijing.volces.com" in lowered and "/api/v3" not in lowered:
        return "https://ark.cn-beijing.volces.com/api/v3"
    return base_url.rstrip("/")


def _infer_duty(provider: ProviderConfig) -> str:
    duty = LEGACY_DUTY_MAP.get(provider.duty, provider.duty)
    lowered = f"{provider.name} {provider.model} {provider.base_url}".lower()

    if any(token in lowered for token in ["lead", "\u603b\u8d1f\u8d23"]):
        return LEAD_DUTY
    if any(token in lowered for token in ["host", "\u4e3b\u6301\u4eba"]):
        return HOST_DUTY
    if any(
        token in lowered
        for token in [
            "literature",
            "review",
            "\u7efc\u8ff0",
            "\u6587\u732e",
            "doubao",
            "ark.cn-beijing.volces.com",
        ]
    ):
        return LITERATURE_DUTY
    if any(token in lowered for token in ["report", "reporter", "\u7edf\u7a3f", "logger"]):
        return REPORT_DUTY
    if duty in DUTY_OPTIONS:
        return duty
    return EXPERT_DUTY


def _normalize_specialty(provider: ProviderConfig) -> str:
    specialty = (provider.specialty or "").strip()
    fallback = _fallback_specialty(provider)
    if specialty and not _looks_like_mojibake(specialty) and not _looks_like_non_english_default(specialty):
        return specialty
    if fallback:
        return fallback
    return "Analysis, evidence checking, and conclusion review"


def _fallback_specialty(provider: ProviderConfig) -> str:
    if provider.duty in DEFAULT_SPECIALTIES:
        return DEFAULT_SPECIALTIES[provider.duty]
    if provider.name in DEFAULT_SPECIALTIES:
        return DEFAULT_SPECIALTIES[provider.name]
    lowered_name = provider.name.lower()
    for key, value in DEFAULT_SPECIALTIES.items():
        if key.lower() in lowered_name:
            return value
    lowered_model = provider.model.lower()
    for key, value in DEFAULT_SPECIALTIES.items():
        if key.lower() in lowered_model:
            return value
    return "Analysis, evidence checking, and conclusion review"


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    if "�" in text:
        return True
    hint_count = sum(1 for char in text if char in MOJIBAKE_HINTS)
    return hint_count >= max(3, len(text) // 5)


def _looks_like_non_english_default(text: str) -> bool:
    if not text:
        return False
    has_ascii_word = any(char.isascii() and char.isalpha() for char in text)
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    return has_cjk and not has_ascii_word
