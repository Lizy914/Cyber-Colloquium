from __future__ import annotations

import re


def detect_primary_language(text: str) -> str:
    if not text:
        return "en"

    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_letter_count = sum(1 for char in text if char.isascii() and char.isalpha())
    ascii_word_count = len(re.findall(r"[A-Za-z]+", text))

    if cjk_count >= 4 and cjk_count >= max(2, ascii_word_count):
        return "zh"
    if cjk_count >= 8 and cjk_count * 2 >= max(1, ascii_letter_count):
        return "zh"
    return "en"


def choose_language(language: str, zh_text: str, en_text: str) -> str:
    return zh_text if language == "zh" else en_text


def language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"
