from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


UI_SETTINGS_PATH = Path("ui_settings.json")
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_OPTIONS = (THEME_DARK, THEME_LIGHT)


@dataclass(frozen=True)
class UiSettings:
    theme: str = THEME_DARK


def load_ui_settings(path: Path = UI_SETTINGS_PATH) -> UiSettings:
    if not path.exists():
        settings = UiSettings()
        save_ui_settings(settings, path)
        return settings

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        settings = UiSettings()
        save_ui_settings(settings, path)
        return settings

    theme = str(raw.get("theme") or THEME_DARK).strip().lower()
    if theme not in THEME_OPTIONS:
        theme = THEME_DARK
    settings = UiSettings(theme=theme)
    save_ui_settings(settings, path)
    return settings


def save_ui_settings(settings: UiSettings, path: Path = UI_SETTINGS_PATH) -> None:
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
