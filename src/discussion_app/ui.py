from __future__ import annotations

from dataclasses import replace
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QTextDocument
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleOptionSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .attachments import load_attachment
from .config import load_providers, save_providers
from .meeting_minutes import save_discussion_outputs, save_failure_snapshot
from .models import DUTY_OPTIONS, EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, LITERATURE_DUTY, REPORT_DUTY, AttachmentPayload, DiscussionMessage, DiscussionResult, ProviderConfig
from .orchestrator import DiscussionOrchestrator
from .pdf_reader import PdfReaderBuildResult, PdfReaderBuilder, pdf_reader_badge
from .workflow_config import WorkflowConfig, load_workflow_config, save_workflow_config
from .workflow_graph import build_workflow_graph, render_workflow_graph_mermaid, workflow_policy_snapshot
from .workflow_settings import (
    SUMMARY_SLOT_LABELS,
    WorkflowSettingsState,
    apply_workflow_settings,
    render_workflow_settings_summary,
    validate_workflow_settings,
    workflow_settings_from_config,
)
from .ui_settings import THEME_DARK, THEME_LIGHT, UiSettings, load_ui_settings, save_ui_settings


APP_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PHOTO_DIR = APP_ROOT / "Profile Photo"
APP_LOGO_PATH = APP_ROOT / "Overall Picture.png"
WORKFLOW_EXPORT_DIR = APP_ROOT / "generated_artifacts" / "workflow_graph_exports"
PROFILE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

MODEL_COLORS = {
    "Kimi Host": "#6C4CF1",
    "Kimi Lead": "#6C4CF1",
    "MiniMax": "#316CFF",
    "Qwen3-Max": "#FF7A18",
    "Qwen-Math": "#C96C10",
    "GLM": "#16A085",
    "GLM Reporter": "#16A085",
    "DeepSeek": "#E64C66",
    "Doubao Review": "#0EA5A4",
    "USER": "#1F2937",
    "HOST": "#8E44AD",
}

FALLBACK_COLORS = [
    "#2563EB",
    "#DC2626",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#0891B2",
    "#BE185D",
    "#4F46E5",
]

THEME_LABELS = {
    THEME_DARK: "深色",
    THEME_LIGHT: "浅色",
}

CHEVRON_COLORS = {
    THEME_DARK: QColor("#7CE7FF"),
    THEME_LIGHT: QColor("#4687C4"),
}


def _dark_stylesheet() -> str:
    return """
    QWidget {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #040915, stop:0.48 #071424, stop:1 #0B1830);
        color: #E8F6FF;
        font-size: 13px;
        font-family: 'Microsoft YaHei UI';
    }
    QWidget#leftPanel {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #060E1D, stop:1 #0B1830);
    }
    QWidget#rightPanel {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #040915, stop:1 #0B1D3B);
    }
    QGroupBox {
        background: rgba(8, 18, 38, 0.84);
        border: 1px solid rgba(83, 216, 255, 0.16);
        border-radius: 22px;
        margin-top: 10px;
        padding-top: 16px;
        font-size: 13px;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 6px;
        color: #D7F7FF;
    }
    QLabel#panelTitle { color: #F2FBFF; font-size: 22px; font-weight: 800; }
    QLabel#panelSubtitle, QLabel#panelHint, QLabel#workflowSummary, QLabel#sectionCaption { color: #9DB7D3; }
    QLabel#workflowSummary { color: #B7D7F3; }
    QLabel#flowValue {
        background: rgba(11, 46, 87, 0.86);
        color: #7FE2FF;
        border: 1px solid rgba(83, 216, 255, 0.28);
        border-radius: 14px;
        padding: 10px 12px;
        font-size: 12px;
        font-weight: 700;
    }
    QFrame#providerCard {
        background: rgba(10, 24, 48, 0.92);
        border: 1px solid rgba(83, 216, 255, 0.16);
        border-radius: 18px;
    }
    QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QComboBox, QSpinBox {
        background: rgba(4, 12, 26, 0.96);
        color: #EFFBFF;
        border: 1px solid rgba(110, 176, 255, 0.26);
        border-radius: 14px;
        padding: 9px 10px;
        selection-background-color: rgba(65, 175, 255, 0.82);
        selection-color: #04101F;
    }
    QSpinBox {
        padding-right: 48px;
    }
    QLineEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QListWidget:focus, QComboBox:focus, QSpinBox:focus {
        border: 1px solid rgba(111, 235, 255, 0.62);
    }
    QSpinBox::up-button, QSpinBox::down-button {
        subcontrol-origin: border;
        width: 24px;
        background: rgba(10, 27, 54, 0.88);
        border-left: 1px solid rgba(83, 216, 255, 0.24);
        color: #D8F6FF;
        font-weight: 800;
    }
    QSpinBox::up-button {
        subcontrol-position: top right;
        border-top-right-radius: 14px;
        border-bottom: 1px solid rgba(83, 216, 255, 0.18);
    }
    QSpinBox::down-button {
        subcontrol-position: bottom right;
        border-bottom-right-radius: 14px;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background: rgba(17, 44, 86, 0.92);
    }
    QLineEdit#roleNameEdit { font-size: 13px; font-weight: 700; }
    QPushButton#providerSummaryName {
        background: transparent;
        color: #F2FBFF;
        border: none;
        padding: 0;
        text-align: left;
        font-size: 14px;
        font-weight: 800;
    }
    QPushButton#providerSummaryName:hover {
        background: transparent;
        color: #8FEAFF;
        border: none;
    }
    QLabel#providerMeta {
        color: #8EA9C5;
        font-size: 11px;
        line-height: 1.35;
    }
    QScrollArea { border: none; background: transparent; }
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0F2A55, stop:1 #13407F);
        color: #F1FCFF;
        border: 1px solid rgba(91, 198, 255, 0.22);
        border-radius: 14px;
        padding: 10px 18px;
        font-size: 13px;
        font-weight: 700;
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #14366D, stop:1 #1953A7);
    }
    QPushButton:disabled {
        background: rgba(79, 104, 141, 0.7);
        color: #C7D3E2;
        border: 1px solid rgba(140, 160, 188, 0.18);
    }
    QPushButton#ghostAction {
        background: rgba(10, 27, 54, 0.88);
        color: #D8F6FF;
        border: 1px solid rgba(83, 216, 255, 0.18);
        border-radius: 12px;
        padding: 8px 14px;
        font-size: 12px;
    }
    QPushButton#ghostAction:hover {
        background: rgba(17, 44, 86, 0.92);
        border: 1px solid rgba(105, 233, 255, 0.32);
    }
    QPushButton#providerTab {
        background: rgba(10, 27, 54, 0.72);
        color: #BDEBFF;
        border: 1px solid rgba(83, 216, 255, 0.16);
        border-radius: 12px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
    }
    QPushButton#providerTab:hover {
        background: rgba(17, 44, 86, 0.92);
        border: 1px solid rgba(105, 233, 255, 0.32);
    }
    QPushButton#providerTab:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0F2A55, stop:1 #13407F);
        color: #F4FCFF;
        border: 1px solid rgba(111, 235, 255, 0.42);
    }
    QToolButton#collapseToggle {
        background: transparent;
        color: #91D8FF;
        border: none;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: 700;
    }
    QToolButton#collapseToggle:hover { color: #D7F8FF; }
    QCheckBox { spacing: 6px; }
    QFrame#topBar { background: transparent; border: none; }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 5px;
        border: 1px solid rgba(93, 170, 255, 0.36);
        background: rgba(5, 18, 36, 0.95);
    }
    QCheckBox::indicator:checked {
        background: #5FDFFF;
        border: 1px solid #B1F7FF;
    }
    QFrame#headerStrip, QFrame#chatSurface { background: transparent; border: none; }
    QLabel#brandLogo {
        background: rgba(5, 18, 36, 0.92);
        border: 1px solid rgba(103, 232, 249, 0.32);
        border-radius: 20px;
        padding: 2px;
    }
    QLabel#heroTitle { color: #F1FCFF; font-size: 36px; font-weight: 800; }
    QLabel#heroSubtitle { color: #64E2FF; font-size: 15px; font-weight: 700; }
    QFrame#composerCard {
        background: rgba(7, 18, 37, 0.88);
        border: 1px solid rgba(83, 216, 255, 0.14);
        border-radius: 26px;
    }
    QWidget#chatCanvas { background: transparent; }
    QPlainTextEdit#promptEdit {
        background: rgba(4, 14, 28, 0.96);
        border: 1px solid rgba(102, 181, 255, 0.26);
        border-radius: 20px;
        padding: 14px 16px;
        color: #EFFBFF;
    }
    QPushButton#primaryAction {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0C6FFF, stop:1 #46DAFF);
        color: #03111E;
        border: 1px solid rgba(177, 247, 255, 0.68);
        border-radius: 18px;
        min-height: 46px;
        padding: 10px 22px;
    }
    QPushButton#primaryAction:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2F86FF, stop:1 #72ECFF);
    }
    QPushButton#primaryAction:disabled {
        background: rgba(68, 151, 219, 0.6);
        color: #D9F9FF;
        border: 1px solid rgba(112, 215, 255, 0.28);
    }
    QPushButton#primaryAction[state="running"],
    QPushButton#primaryAction[state="running"]:disabled {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #11487C, stop:1 #1F78B4);
        color: #E6FBFF;
    }
    QPushButton#secondaryAction {
        background: rgba(9, 25, 51, 0.88);
        color: #98B4D3;
        border: 1px solid rgba(83, 216, 255, 0.14);
        border-radius: 18px;
        min-height: 46px;
        padding: 10px 22px;
    }
    QPushButton#secondaryAction:hover,
    QPushButton#secondaryAction:enabled {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #C83A54, stop:1 #F8647B);
        color: white;
        border: 1px solid rgba(255, 170, 196, 0.46);
    }
    QPushButton#secondaryAction[state="stopping"] {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B67A08, stop:1 #FFD05C);
        color: #1F1720;
        border: 1px solid rgba(255, 230, 170, 0.56);
    }
    QPushButton#secondaryAction[state="stopping"]:disabled {
        background: rgba(214, 170, 63, 0.78);
        color: #221A18;
        border: 1px solid rgba(255, 226, 148, 0.48);
    }
    QScrollBar:vertical {
        background: rgba(4, 13, 28, 0.72);
        width: 12px;
        border-radius: 6px;
        margin: 6px 4px 6px 4px;
    }
    QScrollBar::handle:vertical {
        background: rgba(73, 193, 255, 0.48);
        border-radius: 6px;
        min-height: 36px;
    }
    QScrollBar::handle:vertical:hover { background: rgba(109, 228, 255, 0.72); }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: transparent;
        height: 0px;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        border: none;
        width: 28px;
        margin: 2px 4px 2px 0;
        border-radius: 10px;
        background: rgba(10, 27, 54, 0.72);
    }
    QComboBox::down-arrow {
        image: none;
        width: 0px;
        height: 0px;
        border: none;
    }
    QListView, QAbstractItemView {
        background: rgba(6, 17, 33, 0.98);
        color: #EFFBFF;
        border: 1px solid rgba(92, 181, 255, 0.24);
        selection-background-color: rgba(52, 157, 255, 0.55);
        selection-color: #F6FDFF;
    }
    QSplitter::handle { background: transparent; }
    """


def _light_stylesheet() -> str:
    return """
    QWidget {
        background: #FFFFFF;
        color: #15324D;
        font-size: 13px;
        font-family: 'Microsoft YaHei UI';
    }
    QWidget#leftPanel, QWidget#rightPanel { background: #FFFFFF; }
    QGroupBox {
        background: #FFFFFF;
        border: 1px solid #D9E5F1;
        border-radius: 22px;
        margin-top: 10px;
        padding-top: 16px;
        font-size: 13px;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 6px;
        color: #21486E;
    }
    QLabel#panelTitle { color: #163350; font-size: 22px; font-weight: 800; }
    QLabel#panelSubtitle, QLabel#panelHint, QLabel#workflowSummary, QLabel#sectionCaption { color: #627F99; }
    QLabel#workflowSummary { color: #44637D; }
    QLabel#flowValue {
        background: #F3F8FC;
        color: #2B6C96;
        border: 1px solid #D4E4F1;
        border-radius: 14px;
        padding: 10px 12px;
        font-size: 12px;
        font-weight: 700;
    }
    QFrame#providerCard {
        background: #FFFFFF;
        border: 1px solid #D7E4EF;
        border-radius: 18px;
    }
    QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QComboBox, QSpinBox {
        background: #FFFFFF;
        color: #16314B;
        border: 1px solid #CCDCE8;
        border-radius: 14px;
        padding: 9px 10px;
        selection-background-color: #D7EEFF;
        selection-color: #0F2740;
    }
    QSpinBox {
        padding-right: 48px;
    }
    QLineEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QListWidget:focus, QComboBox:focus, QSpinBox:focus {
        border: 1px solid #6EB7E7;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        subcontrol-origin: border;
        width: 24px;
        background: #F7FAFD;
        border-left: 1px solid #D8E4EE;
        color: #305678;
        font-weight: 800;
    }
    QSpinBox::up-button {
        subcontrol-position: top right;
        border-top-right-radius: 14px;
        border-bottom: 1px solid #D8E4EE;
    }
    QSpinBox::down-button {
        subcontrol-position: bottom right;
        border-bottom-right-radius: 14px;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background: #EDF4FA;
    }
    QLineEdit#roleNameEdit { font-size: 13px; font-weight: 700; }
    QPushButton#providerSummaryName {
        background: transparent;
        color: #173552;
        border: none;
        padding: 0;
        text-align: left;
        font-size: 14px;
        font-weight: 800;
    }
    QPushButton#providerSummaryName:hover {
        background: transparent;
        color: #2A77A7;
        border: none;
    }
    QLabel#providerMeta {
        color: #647F97;
        font-size: 11px;
        line-height: 1.35;
    }
    QScrollArea { border: none; background: transparent; }
    QPushButton {
        background: #F4F8FC;
        color: #16314B;
        border: 1px solid #D4E2EE;
        border-radius: 14px;
        padding: 10px 18px;
        font-size: 13px;
        font-weight: 700;
    }
    QPushButton:hover {
        background: #EAF2F9;
        border: 1px solid #BDD2E6;
    }
    QPushButton:disabled {
        background: #F6F8FA;
        color: #99AAB8;
        border: 1px solid #E3EBF1;
    }
    QPushButton#ghostAction {
        background: #F7FAFD;
        color: #305678;
        border: 1px solid #D8E4EE;
        border-radius: 12px;
        padding: 8px 14px;
        font-size: 12px;
    }
    QPushButton#ghostAction:hover {
        background: #EDF4FA;
        border: 1px solid #BDD1E3;
    }
    QPushButton#providerTab {
        background: #F7FAFD;
        color: #315678;
        border: 1px solid #D8E4EE;
        border-radius: 12px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
    }
    QPushButton#providerTab:hover {
        background: #EDF4FA;
        border: 1px solid #BDD1E3;
    }
    QPushButton#providerTab:checked {
        background: #DFF1FF;
        color: #18476C;
        border: 1px solid #8FC6E9;
    }
    QToolButton#collapseToggle {
        background: transparent;
        color: #4C6B87;
        border: none;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: 700;
    }
    QToolButton#collapseToggle:hover { color: #15324D; }
    QCheckBox { spacing: 6px; }
    QFrame#topBar { background: transparent; border: none; }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 5px;
        border: 1px solid #BFD2E3;
        background: #FFFFFF;
    }
    QCheckBox::indicator:checked {
        background: #46B9F7;
        border: 1px solid #46B9F7;
    }
    QFrame#headerStrip, QFrame#chatSurface { background: transparent; border: none; }
    QLabel#brandLogo {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(184, 210, 232, 0.88);
        border-radius: 20px;
        padding: 2px;
    }
    QLabel#heroTitle { color: #16314B; font-size: 36px; font-weight: 800; }
    QLabel#heroSubtitle { color: #1E7BC9; font-size: 15px; font-weight: 700; }
    QFrame#composerCard {
        background: #FFFFFF;
        border: 1px solid #D7E4EF;
        border-radius: 26px;
    }
    QWidget#chatCanvas { background: transparent; }
    QPlainTextEdit#promptEdit {
        background: #FFFFFF;
        border: 1px solid #CCDDE8;
        border-radius: 20px;
        padding: 14px 16px;
        color: #15324D;
    }
    QPushButton#primaryAction {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0C6FFF, stop:1 #46DAFF);
        color: white;
        border: 1px solid #7ECFFF;
        border-radius: 18px;
        min-height: 46px;
        padding: 10px 22px;
    }
    QPushButton#primaryAction:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2F86FF, stop:1 #72ECFF);
    }
    QPushButton#primaryAction:disabled {
        background: #DDEAF5;
        color: #90A6BA;
        border: 1px solid #D7E3EE;
    }
    QPushButton#primaryAction[state="running"],
    QPushButton#primaryAction[state="running"]:disabled {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #11487C, stop:1 #1F78B4);
        color: white;
    }
    QPushButton#secondaryAction {
        background: #F7FAFD;
        color: #6F879C;
        border: 1px solid #D4E2EE;
        border-radius: 18px;
        min-height: 46px;
        padding: 10px 22px;
    }
    QPushButton#secondaryAction:hover,
    QPushButton#secondaryAction:enabled {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #C83A54, stop:1 #F8647B);
        color: white;
        border: 1px solid rgba(255, 170, 196, 0.46);
    }
    QPushButton#secondaryAction[state="stopping"] {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #B67A08, stop:1 #FFD05C);
        color: #1F1720;
        border: 1px solid rgba(255, 230, 170, 0.56);
    }
    QPushButton#secondaryAction[state="stopping"]:disabled {
        background: rgba(214, 170, 63, 0.42);
        color: #5F5139;
        border: 1px solid rgba(210, 196, 154, 0.48);
    }
    QScrollBar:vertical {
        background: #F2F6FA;
        width: 12px;
        border-radius: 6px;
        margin: 6px 4px 6px 4px;
    }
    QScrollBar::handle:vertical {
        background: #B8D3EA;
        border-radius: 6px;
        min-height: 36px;
    }
    QScrollBar::handle:vertical:hover { background: #9CC4E6; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: transparent;
        height: 0px;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        border: none;
        width: 28px;
        margin: 2px 4px 2px 0;
        border-radius: 10px;
        background: #F3F8FC;
    }
    QComboBox::down-arrow {
        image: none;
        width: 0px;
        height: 0px;
        border: none;
    }
    QListView, QAbstractItemView {
        background: #FFFFFF;
        color: #16314B;
        border: 1px solid #D5E3EF;
        selection-background-color: #D7EEFF;
        selection-color: #0F2740;
    }
    QSplitter::handle { background: transparent; }
    """


def _normalize_asset_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _load_profile_photo_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not PROFILE_PHOTO_DIR.exists():
        return mapping
    for path in PROFILE_PHOTO_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in PROFILE_IMAGE_EXTENSIONS:
            mapping[_normalize_asset_key(path.stem)] = path
    return mapping


PROFILE_PHOTO_MAP = _load_profile_photo_map()


def color_for_speaker(speaker: str) -> str:
    if speaker in MODEL_COLORS:
        return MODEL_COLORS[speaker]
    score = sum((index + 1) * ord(char) for index, char in enumerate(speaker))
    return FALLBACK_COLORS[score % len(FALLBACK_COLORS)]


def avatar_photo_path(speaker: str, duty: str = "") -> Path | None:
    for candidate in (speaker, duty):
        key = _normalize_asset_key(candidate)
        if key and key in PROFILE_PHOTO_MAP:
            return PROFILE_PHOTO_MAP[key]
    if duty == EXPERT_DUTY:
        variants = [item for item in (PROFILE_PHOTO_MAP.get("expert"), PROFILE_PHOTO_MAP.get("expert1")) if item]
        if variants:
            score = sum((index + 1) * ord(char) for index, char in enumerate(speaker))
            return variants[score % len(variants)]
    return None


def circular_pixmap(path: Path | None, size: int) -> QPixmap | None:
    if path is None or not path.exists():
        return None
    source = QPixmap(str(path))
    if source.isNull():
        return None
    scaled = source.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    target = QPixmap(size, size)
    target.fill(Qt.transparent)
    painter = QPainter(target)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    clip_path = QPainterPath()
    clip_path.addEllipse(0, 0, size, size)
    painter.setClipPath(clip_path)
    painter.drawPixmap(0, 0, scaled)
    painter.end()
    return target


def build_logo_pixmap(width: int, height: int) -> QPixmap | None:
    if not APP_LOGO_PATH.exists():
        return None
    pixmap = QPixmap(str(APP_LOGO_PATH))
    if pixmap.isNull():
        return None
    return pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ChevronComboBox(QComboBox):
    def __init__(self, theme: str = THEME_DARK, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = QColor(CHEVRON_COLORS.get(self._theme, CHEVRON_COLORS[THEME_DARK]))
        if not self.isEnabled():
            color.setAlpha(120)
        pen = QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        center_x = self.width() - 18
        center_y = self.height() // 2 + 1
        painter.drawLine(center_x - 5, center_y - 3, center_x, center_y + 2)
        painter.drawLine(center_x, center_y + 2, center_x + 5, center_y - 3)
        painter.end()


class DiscussionWorker(QObject):
    message_ready = Signal(object)
    status_ready = Signal(str)
    finished = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        providers: list[ProviderConfig],
        prompt: str,
        attachments: list[AttachmentPayload],
        generate_literature_review: bool,
        local_execution_authorized: bool,
        workflow_config: WorkflowConfig,
    ) -> None:
        super().__init__()
        self.providers = providers
        self.prompt = prompt
        self.attachments = attachments
        self.generate_literature_review = generate_literature_review
        self.local_execution_authorized = local_execution_authorized
        self.workflow_config = workflow_config
        self.cancel_event = Event()

    def run(self) -> None:
        orchestrator = DiscussionOrchestrator(self.providers, workflow_config=self.workflow_config)
        try:
            result = orchestrator.run_discussion(
                user_request=self.prompt,
                attachments=self.attachments,
                generate_literature_review=self.generate_literature_review,
                local_execution_authorized=self.local_execution_authorized,
                on_message=self.message_ready.emit,
                on_status=self.status_ready.emit,
                should_cancel=self.cancel_event.is_set,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(
                {
                    "error_text": str(exc),
                    "traceback": traceback.format_exc(),
                    "partial_result": orchestrator.latest_result,
                }
            )
            return
        self.finished.emit(result)

    def stop(self) -> None:
        self.cancel_event.set()


class PdfReaderWorker(QObject):
    status_ready = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, provider: ProviderConfig | None, attachments: list[AttachmentPayload]) -> None:
        super().__init__()
        self.provider = provider
        self.attachments = attachments

    def run(self) -> None:
        try:
            builder = PdfReaderBuilder(self.provider)
            results = builder.build_many(self.attachments, on_status=self.status_ready.emit)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
            return
        self.finished.emit(results)


class BubbleText(QLabel):
    def __init__(self, body: str, theme: str = THEME_DARK, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setOpenExternalLinks(True)
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.set_theme(theme)
        doc = QTextDocument(self)
        doc.setDefaultFont(self.font())
        doc.setMarkdown(body)
        self.setText(doc.toHtml())

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        color = "#E6F6FF" if theme == THEME_DARK else "#15324D"
        self.setStyleSheet(
            f"background: transparent; border: none; color: {color}; font-size: 13px; font-family: 'Microsoft YaHei UI';"
        )


class TimelineDivider(QFrame):
    def __init__(self, text: str, accent: bool = False, theme: str = THEME_DARK) -> None:
        super().__init__()
        self.theme = theme
        self.accent = accent
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(10)

        self.left_line = QFrame()
        self.left_line.setFrameShape(QFrame.HLine)
        self.right_line = QFrame()
        self.right_line.setFrameShape(QFrame.HLine)
        self.chip = QLabel(text)
        self.chip.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.left_line, 1)
        layout.addWidget(self.chip)
        layout.addWidget(self.right_line, 1)
        self.set_theme(theme)

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        if theme == THEME_DARK:
            line_color = "rgba(83, 216, 255, 0.34)"
            chip_bg = "rgba(17, 77, 122, 0.88)" if self.accent else "rgba(8, 29, 60, 0.88)"
            chip_fg = "#89E8FF" if self.accent else "#A7C5E5"
        else:
            line_color = "rgba(85, 127, 166, 0.32)"
            chip_bg = "#DFF3FF" if self.accent else "#F3F8FC"
            chip_fg = "#106291" if self.accent else "#47637B"
        self.left_line.setStyleSheet(f"color: {line_color};")
        self.right_line.setStyleSheet(f"color: {line_color};")
        self.chip.setStyleSheet(
            f"background: {chip_bg}; color: {chip_fg}; border-radius: 999px; padding: 6px 14px; font-size: 11px; font-weight: 700; font-family: 'Microsoft YaHei UI';"
        )


class BrandHeaderFrame(QFrame):
    def __init__(self, theme: str = THEME_DARK, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 30, 30)

        base = QLinearGradient(rect.topLeft(), rect.bottomRight())
        if self.theme == THEME_DARK:
            base.setColorAt(0.0, QColor(8, 24, 49, 248))
            base.setColorAt(0.52, QColor(15, 43, 86, 242))
            base.setColorAt(1.0, QColor(22, 31, 72, 244))
            glow_specs = [
                (rect.left() + 180, rect.top() + 40, 180, 120, QColor(93, 222, 255, 32)),
                (rect.center().x() - 90, rect.top() + 18, 240, 110, QColor(123, 184, 255, 28)),
                (rect.right() - 260, rect.bottom() - 72, 220, 120, QColor(118, 98, 255, 24)),
            ]
            line_pen = QPen(QColor(119, 233, 255, 110), 1.4)
            node_color = QColor(137, 232, 255, 140)
            border_pen = QPen(QColor(89, 214, 255, 56), 1.2)
        else:
            base.setColorAt(0.0, QColor(255, 255, 255, 248))
            base.setColorAt(0.55, QColor(242, 248, 255, 244))
            base.setColorAt(1.0, QColor(233, 242, 252, 242))
            glow_specs = [
                (rect.left() + 180, rect.top() + 40, 180, 120, QColor(93, 162, 255, 24)),
                (rect.center().x() - 90, rect.top() + 18, 240, 110, QColor(123, 184, 255, 20)),
                (rect.right() - 260, rect.bottom() - 72, 220, 120, QColor(118, 98, 255, 16)),
            ]
            line_pen = QPen(QColor(74, 145, 206, 74), 1.2)
            node_color = QColor(87, 168, 219, 120)
            border_pen = QPen(QColor(118, 168, 214, 52), 1.0)
        painter.fillPath(path, base)

        painter.setPen(Qt.NoPen)
        painter.setClipPath(path)
        for x, y, w, h, color in glow_specs:
            painter.setBrush(color)
            painter.drawEllipse(x, y, w, h)

        painter.setPen(line_pen)

        left_x = rect.left() + 20
        top_y = rect.top() + 24
        spans = (0, 16, 32)
        lengths = (78, 112, 92)
        for index, offset in enumerate(spans):
            y = top_y + offset
            x1 = left_x + (index * 8)
            x2 = x1 + lengths[index]
            painter.drawLine(x1, y, x2, y)
            painter.drawLine(x2, y, x2 + 22, y + 14)
            painter.drawLine(x2 + 22, y + 14, x2 + 42, y + 14)
        for node_x, node_y in ((left_x + 140, top_y + 14), (left_x + 168, top_y + 30), (left_x + 152, top_y + 46)):
            painter.setBrush(node_color)
            painter.drawEllipse(node_x, node_y, 5, 5)

        right_x = rect.right() - 20
        for index, offset in enumerate(spans):
            y = top_y + offset
            x2 = right_x - (index * 8)
            x1 = x2 - lengths[index]
            painter.drawLine(x1, y, x2, y)
            painter.drawLine(x1 - 22, y + 14, x1, y)
            painter.drawLine(x1 - 42, y + 14, x1 - 22, y + 14)
        for node_x, node_y in ((right_x - 173, top_y + 14), (right_x - 201, top_y + 30), (right_x - 185, top_y + 46)):
            painter.setBrush(node_color)
            painter.drawEllipse(node_x, node_y, 5, 5)

        painter.setClipping(False)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 30, 30)
        painter.end()


class ChatSurfaceFrame(QFrame):
    def __init__(self, theme: str = THEME_DARK, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 28, 28)

        fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        if self.theme == THEME_DARK:
            fill.setColorAt(0.0, QColor(7, 17, 34, 232))
            fill.setColorAt(1.0, QColor(3, 10, 22, 240))
            grid_pen = QPen(QColor(88, 204, 255, 18), 1)
            dot_color = QColor(153, 235, 255, 52)
            small_dot_color = QColor(153, 235, 255, 26)
            circuit_pen = QPen(QColor(84, 221, 255, 36), 1.2)
            border_pen = QPen(QColor(89, 214, 255, 42), 1.0)
        else:
            fill.setColorAt(0.0, QColor(255, 255, 255, 242))
            fill.setColorAt(1.0, QColor(247, 250, 255, 246))
            grid_pen = QPen(QColor(120, 164, 201, 20), 1)
            dot_color = QColor(118, 186, 235, 28)
            small_dot_color = QColor(118, 186, 235, 18)
            circuit_pen = QPen(QColor(117, 170, 217, 24), 1.0)
            border_pen = QPen(QColor(129, 175, 215, 36), 1.0)
        painter.fillPath(path, fill)
        painter.setClipPath(path)

        painter.setPen(grid_pen)
        grid_step_x = 74
        grid_step_y = 58
        for x in range(rect.left() + 22, rect.right(), grid_step_x):
            painter.drawLine(x, rect.top() + 18, x, rect.bottom() - 18)
        for y in range(rect.top() + 18, rect.bottom(), grid_step_y):
            painter.drawLine(rect.left() + 20, y, rect.right() - 20, y)

        painter.setPen(Qt.NoPen)
        width = max(180, rect.width() - 48)
        height = max(140, rect.height() - 40)
        for index in range(22):
            x = rect.left() + 24 + ((index * 97) % width)
            y = rect.top() + 20 + ((index * 61) % height)
            radius = 2 if index % 6 == 0 else 1
            painter.setBrush(dot_color if radius == 2 else small_dot_color)
            painter.drawEllipse(x, y, radius + 1, radius + 1)

        painter.setPen(circuit_pen)
        base_y = rect.top() + 34
        for index in range(4):
            x1 = rect.left() + 36 + index * 54
            x2 = x1 + 30
            y = base_y + (index % 2) * 18
            painter.drawLine(x1, y, x2, y)
            painter.drawLine(x2, y, x2 + 16, y + 12)
            painter.drawLine(x2 + 16, y + 12, x2 + 42, y + 12)
        corner_y = rect.bottom() - 52
        for index in range(3):
            x2 = rect.right() - 34 - index * 58
            x1 = x2 - 34
            y = corner_y - (index % 2) * 16
            painter.drawLine(x1, y, x2, y)
            painter.drawLine(x1 - 16, y - 12, x1, y)
            painter.drawLine(x1 - 42, y - 12, x1 - 16, y - 12)

        painter.setClipping(False)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 28, 28)
        painter.end()


def apply_outer_glow(widget: QWidget, color: QColor, blur_radius: int, alpha: int) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setOffset(0, 0)
    effect.setBlurRadius(blur_radius)
    glow = QColor(color)
    glow.setAlpha(alpha)
    effect.setColor(glow)
    widget.setGraphicsEffect(effect)


class GlowButton(QPushButton):
    def __init__(self, text: str, glow_color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._glow_color = QColor(glow_color)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setOffset(0, 0)
        self.setGraphicsEffect(self._glow)
        self._set_glow(False)

    def _set_glow(self, hover: bool) -> None:
        color = QColor(self._glow_color)
        if not self.isEnabled():
            color.setAlpha(0)
            blur = 0
        elif hover:
            color.setAlpha(168)
            blur = 28
        else:
            color.setAlpha(56)
            blur = 16
        self._glow.setColor(color)
        self._glow.setBlurRadius(blur)

    def enterEvent(self, event) -> None:  # noqa: ANN001
        self._set_glow(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self._set_glow(False)
        super().leaveEvent(event)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._set_glow(False)


class PlusMinusSpinBox(QSpinBox):
    def __init__(self) -> None:
        super().__init__()
        self.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.setAccelerated(True)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)

        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        up_rect = self.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, self)
        down_rect = self.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, self)
        if not up_rect.isValid() or not down_rect.isValid():
            return

        color = self.palette().buttonText().color()
        if not self.isEnabled():
            color.setAlpha(140)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap))
        self._draw_minus(painter, down_rect)
        self._draw_plus(painter, up_rect)

    @staticmethod
    def _draw_minus(painter: QPainter, rect) -> None:  # noqa: ANN001
        half = max(4, min(rect.width(), rect.height()) // 5)
        center = rect.center()
        painter.drawLine(center.x() - half, center.y(), center.x() + half, center.y())

    @staticmethod
    def _draw_plus(painter: QPainter, rect) -> None:  # noqa: ANN001
        half = max(4, min(rect.width(), rect.height()) // 5)
        center = rect.center()
        painter.drawLine(center.x() - half, center.y(), center.x() + half, center.y())
        painter.drawLine(center.x(), center.y() - half, center.x(), center.y() + half)


class ProviderConfigEditor(QFrame):
    def __init__(self, provider: ProviderConfig | None = None) -> None:
        super().__init__()
        self.provider = provider
        self.theme = THEME_DARK
        self._loading_provider = False
        self.setObjectName("providerCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.enabled_check = QCheckBox("启用该角色")
        layout.addWidget(self.enabled_check)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("roleNameEdit")
        self.name_edit.setPlaceholderText("Role name")

        self.duty_combo = ChevronComboBox(self.theme)
        self.duty_combo.addItems(DUTY_OPTIONS)

        self.specialty_edit = QLineEdit()
        self.specialty_edit.setPlaceholderText("例如：文献复核、实验设计、数学推导")

        self.model_edit = QLineEdit()
        self.base_url_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.vision_check = QCheckBox("Enable vision")

        form.addRow("角色名称", self.name_edit)
        form.addRow("Duty", self.duty_combo)
        form.addRow("Specialty", self.specialty_edit)
        form.addRow("Model", self.model_edit)
        form.addRow("Base URL", self.base_url_edit)
        form.addRow("API Key", self.api_key_edit)
        form.addRow("Capability", self.vision_check)

        layout.addLayout(form)
        layout.addStretch(1)
        self.set_provider(provider)

    def set_provider(self, provider: ProviderConfig | None) -> None:
        self.provider = provider
        controls = [
            self.enabled_check,
            self.name_edit,
            self.duty_combo,
            self.specialty_edit,
            self.model_edit,
            self.base_url_edit,
            self.api_key_edit,
            self.vision_check,
        ]
        self._loading_provider = True
        for control in controls:
            control.blockSignals(True)
        if provider is None:
            self.enabled_check.setChecked(False)
            self.name_edit.clear()
            self.duty_combo.setCurrentText(EXPERT_DUTY)
            self.specialty_edit.clear()
            self.model_edit.clear()
            self.base_url_edit.clear()
            self.api_key_edit.clear()
            self.vision_check.setChecked(False)
            self.setEnabled(False)
            for control in controls:
                control.blockSignals(False)
            self._loading_provider = False
            return

        self.setEnabled(True)
        self.enabled_check.setChecked(provider.enabled)
        self.name_edit.setText(provider.name)
        self.duty_combo.setCurrentText(provider.duty if provider.duty in DUTY_OPTIONS else EXPERT_DUTY)
        self.specialty_edit.setText(provider.specialty)
        self.model_edit.setText(provider.model)
        self.base_url_edit.setText(provider.base_url)
        self.api_key_edit.setText(provider.api_key)
        self.vision_check.setChecked(provider.supports_vision)
        for control in controls:
            control.blockSignals(False)
        self._loading_provider = False

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        if isinstance(self.duty_combo, ChevronComboBox):
            self.duty_combo.set_theme(theme)

    def apply_to_provider(self, target: ProviderConfig | None = None) -> ProviderConfig:
        provider = target or self.provider
        if provider is None:
            provider = ProviderConfig(name="未命名角色", model="", base_url="")
        provider.enabled = self.enabled_check.isChecked()
        provider.name = self.name_edit.text().strip() or provider.name or "未命名角色"
        provider.duty = self.duty_combo.currentText()
        provider.specialty = self.specialty_edit.text().strip()
        provider.model = self.model_edit.text().strip()
        provider.base_url = self.base_url_edit.text().strip()
        provider.api_key = self.api_key_edit.text().strip()
        provider.supports_vision = self.vision_check.isChecked()
        self.provider = provider
        return provider


class ProviderSummaryCard(QFrame):
    edit_requested = Signal(int)

    def __init__(self, provider: ProviderConfig, index: int) -> None:
        super().__init__()
        self.provider = provider
        self.index = index
        self.theme = THEME_DARK
        self.setObjectName("providerCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(provider.enabled)
        self.enabled_check.toggled.connect(self._on_enabled_toggled)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)

        self.name_button = QPushButton(provider.name)
        self.name_button.setObjectName("providerSummaryName")
        self.name_button.clicked.connect(self._emit_edit_requested)

        self.meta_label = QLabel()
        self.meta_label.setObjectName("providerMeta")
        self.meta_label.setWordWrap(True)

        content_layout.addWidget(self.name_button)
        content_layout.addWidget(self.meta_label)

        self.edit_button = QPushButton("配置")
        self.edit_button.setObjectName("ghostAction")
        self.edit_button.clicked.connect(self._emit_edit_requested)

        layout.addWidget(self.enabled_check, 0, Qt.AlignTop)
        layout.addLayout(content_layout, 1)
        layout.addWidget(self.edit_button, 0, Qt.AlignCenter)
        self.refresh_from_provider()

    def _on_enabled_toggled(self, checked: bool) -> None:
        self.provider.enabled = checked
        self.refresh_from_provider()

    def _emit_edit_requested(self) -> None:
        self.edit_requested.emit(self.index)

    def refresh_from_provider(self) -> None:
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(self.provider.enabled)
        self.enabled_check.blockSignals(False)
        self.name_button.setText(self.provider.name or "未命名角色")
        specialty = self.provider.specialty or "未填写专长"
        model = self.provider.model or "未设置模型"
        key_state = "已配置 API" if self.provider.api_key.strip() else "未配置 API"
        vision_state = "Vision" if self.provider.supports_vision else "Text"
        state_label = "已启用" if self.provider.enabled else "已停用"
        self.meta_label.setText(
            f"{self.provider.duty} · {model}\n"
            f"{state_label} · {key_state} · {vision_state} · {specialty}"
        )

    def set_theme(self, theme: str) -> None:
        self.theme = theme

    def apply_to_provider(self) -> ProviderConfig:
        self.provider.enabled = self.enabled_check.isChecked()
        return self.provider


class ProviderManagerDialog(QDialog):
    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        start_index: int = 0,
        theme: str = THEME_DARK,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.original_providers = providers
        self.working_providers = [replace(provider) for provider in providers]
        self.current_index = -1
        self.current_filter_duty = self._initial_filter_duty(start_index)
        self.theme = theme
        self.visible_provider_indices: list[int] = []
        self.filter_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle("角色配置")
        self.setModal(True)
        self.resize(1100, 720)
        self.setMinimumSize(960, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("角色配置")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        hint = QLabel("左侧切换角色，右侧编辑详细配置。主控制台只保留总览与开关，详细设置统一在这里完成。")
        hint.setObjectName("panelSubtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        tab_row = QHBoxLayout()
        tab_row.setSpacing(8)
        for duty in DUTY_OPTIONS:
            button = QPushButton(duty)
            button.setObjectName("providerTab")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, duty_name=duty: self._set_filter_duty(duty_name))
            self.filter_buttons[duty] = button
            tab_row.addWidget(button)
        tab_row.addStretch(1)
        self.add_expert_button = QPushButton("+ 专家")
        self.add_expert_button.setObjectName("ghostAction")
        self.add_expert_button.clicked.connect(self._add_expert_provider)
        tab_row.addWidget(self.add_expert_button)
        layout.addLayout(tab_row)

        body = QHBoxLayout()
        body.setSpacing(14)

        self.provider_list = QListWidget()
        self.provider_list.setMinimumWidth(260)
        self.provider_list.setMaximumWidth(340)
        self.provider_list.currentRowChanged.connect(self._on_provider_selected)
        body.addWidget(self.provider_list, 0)

        self.editor = ProviderConfigEditor()
        self.editor.set_theme(theme)
        self.editor.name_edit.textChanged.connect(self._refresh_current_list_item)
        self.editor.duty_combo.currentTextChanged.connect(self._refresh_current_list_item)
        self.editor.enabled_check.toggled.connect(lambda _checked: self._refresh_current_list_item())
        body.addWidget(self.editor, 1)
        layout.addLayout(body, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        save_btn = buttons.button(QDialogButtonBox.Save)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        if save_btn is not None:
            save_btn.setText("保存并关闭")
        if cancel_btn is not None:
            cancel_btn.setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._set_filter_duty(self.current_filter_duty, preferred_index=start_index)

    def _initial_filter_duty(self, start_index: int) -> str:
        if 0 <= start_index < len(self.working_providers):
            duty = self.working_providers[start_index].duty
            if duty in DUTY_OPTIONS:
                return duty
        return LEAD_DUTY

    def _filtered_provider_indices(self, duty: str) -> list[int]:
        indices = [index for index, provider in enumerate(self.working_providers) if provider.duty == duty]
        if indices:
            return indices
        return []

    def _populate_provider_list(self, *, preferred_index: int | None = None) -> None:
        selected_index = self.current_index if preferred_index is None else preferred_index
        self.visible_provider_indices = self._filtered_provider_indices(self.current_filter_duty)
        self.provider_list.blockSignals(True)
        self.provider_list.clear()
        for index in self.visible_provider_indices:
            self.provider_list.addItem(self._provider_list_label(self.working_providers[index]))
        if self.visible_provider_indices:
            if selected_index in self.visible_provider_indices:
                row = self.visible_provider_indices.index(selected_index)
            else:
                row = 0
            self.provider_list.setCurrentRow(row)
        else:
            self.provider_list.setCurrentRow(-1)
        self.provider_list.blockSignals(False)
        if self.visible_provider_indices:
            self._on_provider_selected(self.provider_list.currentRow())
        else:
            self.current_index = -1
            self.editor.set_provider(None)

    def _provider_list_label(self, provider: ProviderConfig) -> str:
        name = provider.name or "未命名角色"
        state = "已启用" if provider.enabled else "已停用"
        model = provider.model or "未设置模型"
        return f"{name}\n{state} · {model}"

    def _set_filter_duty(self, duty: str, *, preferred_index: int | None = None) -> None:
        if duty not in DUTY_OPTIONS:
            return
        if self.current_index != -1 and 0 <= self.current_index < len(self.working_providers):
            self.editor.apply_to_provider(self.working_providers[self.current_index])
        self.current_filter_duty = duty
        for duty_name, button in self.filter_buttons.items():
            button.blockSignals(True)
            button.setChecked(duty_name == duty)
            button.blockSignals(False)
        self.add_expert_button.setEnabled(duty == EXPERT_DUTY)
        self._populate_provider_list(preferred_index=preferred_index)

    def _sync_current_provider(self) -> None:
        if not (0 <= self.current_index < len(self.working_providers)):
            return
        self.editor.apply_to_provider(self.working_providers[self.current_index])
        self._populate_provider_list(preferred_index=self.current_index)

    def _refresh_list_item(self, index: int) -> None:
        if index not in self.visible_provider_indices:
            return
        row = self.visible_provider_indices.index(index)
        item = self.provider_list.item(row)
        if item is None or not (0 <= index < len(self.working_providers)):
            return
        item.setText(self._provider_list_label(self.working_providers[index]))

    def _refresh_current_list_item(self) -> None:
        if self.editor._loading_provider:
            return
        if not (0 <= self.current_index < len(self.working_providers)):
            return
        self.editor.apply_to_provider(self.working_providers[self.current_index])
        if self.working_providers[self.current_index].duty != self.current_filter_duty:
            self._set_filter_duty(self.working_providers[self.current_index].duty, preferred_index=self.current_index)
            return
        self._refresh_list_item(self.current_index)

    def _on_provider_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.visible_provider_indices):
            self.current_index = -1
            self.editor.set_provider(None)
            return
        provider_index = self.visible_provider_indices[row]
        if self.current_index == provider_index:
            self.editor.set_provider(self.working_providers[provider_index])
            return
        if self.current_index != -1:
            self.editor.apply_to_provider(self.working_providers[self.current_index])
        self.current_index = provider_index
        self.editor.set_provider(self.working_providers[provider_index])

    def _add_expert_provider(self) -> None:
        next_index = 1 + sum(1 for provider in self.working_providers if provider.duty == EXPERT_DUTY)
        provider = ProviderConfig(
            name=f"专家{next_index}",
            model="",
            base_url="",
            api_key="",
            enabled=False,
            supports_vision=False,
            duty=EXPERT_DUTY,
            specialty="",
        )
        self.working_providers.append(provider)
        new_index = len(self.working_providers) - 1
        self.current_filter_duty = EXPERT_DUTY
        self._set_filter_duty(EXPERT_DUTY, preferred_index=new_index)

    def accept(self) -> None:
        if self.current_index != -1:
            self.editor.apply_to_provider(self.working_providers[self.current_index])
        for index, working in enumerate(self.working_providers):
            if index < len(self.original_providers):
                original = self.original_providers[index]
                original.enabled = working.enabled
                original.name = working.name
                original.duty = working.duty
                original.specialty = working.specialty
                original.model = working.model
                original.base_url = working.base_url
                original.api_key = working.api_key
                original.supports_vision = working.supports_vision
            else:
                self.original_providers.append(replace(working))
        if len(self.original_providers) > len(self.working_providers):
            del self.original_providers[len(self.working_providers) :]
        super().accept()


class WorkflowSettingsDialog(QDialog):
    def __init__(self, workflow_config: WorkflowConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workflow_config = workflow_config
        self.role_checks: dict[str, QCheckBox] = {}
        self.summary_slot_checks: dict[str, QCheckBox] = {}
        self.setWindowTitle("工作流设置")
        self.setModal(True)
        self.resize(980, 760)
        self.setMinimumSize(860, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 0)
        content_layout.setSpacing(14)
        scroll.setWidget(content)

        intro = QLabel(
            "调整下一次运行的检索、复核、代码生成与本地执行行为。除角色名称外，其余说明均使用中文。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 12px; font-family: 'Microsoft YaHei UI'; color: #9DB7D3;")
        content_layout.addWidget(intro)

        state = workflow_settings_from_config(workflow_config)

        policy_group = QGroupBox("讨论策略")
        policy_form = QFormLayout(policy_group)
        policy_form.setContentsMargins(14, 14, 14, 14)
        policy_form.setSpacing(10)

        self.max_rounds_spin = PlusMinusSpinBox()
        self.max_rounds_spin.setRange(1, 24)
        self.max_rounds_spin.setValue(state.max_rounds)
        self.checkpoint_spin = PlusMinusSpinBox()
        self.checkpoint_spin.setRange(1, 24)
        self.checkpoint_spin.setValue(state.checkpoint_every_n_rounds)
        self.reviewer_check = QCheckBox("启用复核阶段")
        self.reviewer_check.setChecked(state.reviewer_enabled)

        policy_form.addRow("最大讨论轮数", self.max_rounds_spin)
        policy_form.addRow("每 N 轮创建检查点", self.checkpoint_spin)
        policy_form.addRow("复核阶段", self.reviewer_check)
        content_layout.addWidget(policy_group)

        main_row = QHBoxLayout()
        main_row.setSpacing(14)
        content_layout.addLayout(main_row)

        left_column = QVBoxLayout()
        left_column.setSpacing(14)
        right_column = QVBoxLayout()
        right_column.setSpacing(14)
        main_row.addLayout(left_column, 3)
        main_row.addLayout(right_column, 2)

        tooling_group = QGroupBox("产物工具")
        tooling_layout = QVBoxLayout(tooling_group)
        tooling_layout.setContentsMargins(14, 14, 14, 14)
        tooling_layout.setSpacing(8)
        self.arxiv_discovery_check = QCheckBox("在材料摄取前启用 arXiv 检索")
        self.arxiv_discovery_check.setChecked(state.arxiv_discovery_enabled)
        self.arxiv_download_check = QCheckBox("将选中的 arXiv PDF 下载到项目文献库")
        self.arxiv_download_check.setChecked(state.arxiv_download_enabled)
        self.arxiv_download_check.setEnabled(state.arxiv_discovery_enabled)
        self.arxiv_discovery_check.toggled.connect(self.arxiv_download_check.setEnabled)
        self.arxiv_max_results_spin = PlusMinusSpinBox()
        self.arxiv_max_results_spin.setRange(1, 20)
        self.arxiv_max_results_spin.setValue(state.arxiv_max_results)
        self.python_artifact_check = QCheckBox("在报告阶段后生成 Python 草稿产物")
        self.python_artifact_check.setChecked(state.python_artifact_enabled)
        self.python_execution_test_check = QCheckBox("允许在隔离运行目录中执行本地 Python smoke test（需本轮授权）")
        self.python_execution_test_check.setChecked(state.python_execution_test_enabled)
        self.python_execution_test_check.setEnabled(state.python_artifact_enabled)
        self.python_full_execution_check = QCheckBox("允许在当前解释器中执行完整 Python 运行（需本轮授权）")
        self.python_full_execution_check.setChecked(state.python_full_execution_enabled)
        self.python_full_execution_check.setEnabled(state.python_artifact_enabled)
        self.python_timeout_spin = PlusMinusSpinBox()
        self.python_timeout_spin.setRange(5, 600)
        self.python_timeout_spin.setValue(state.python_execution_timeout_seconds)
        self.python_timeout_spin.setEnabled(state.python_artifact_enabled)
        self.python_full_timeout_spin = PlusMinusSpinBox()
        self.python_full_timeout_spin.setRange(10, 3600)
        self.python_full_timeout_spin.setValue(state.python_full_execution_timeout_seconds)
        self.python_full_timeout_spin.setEnabled(state.python_artifact_enabled)
        self.python_input_limit_spin = PlusMinusSpinBox()
        self.python_input_limit_spin.setRange(1, 2048)
        self.python_input_limit_spin.setValue(state.python_workspace_input_limit_mb)
        self.python_input_limit_spin.setEnabled(state.python_artifact_enabled)
        self.python_artifact_check.toggled.connect(self.python_execution_test_check.setEnabled)
        self.python_artifact_check.toggled.connect(self.python_full_execution_check.setEnabled)
        self.python_artifact_check.toggled.connect(self.python_timeout_spin.setEnabled)
        self.python_artifact_check.toggled.connect(self.python_full_timeout_spin.setEnabled)
        self.python_artifact_check.toggled.connect(self.python_input_limit_spin.setEnabled)
        self.python_artifact_check.toggled.connect(
            lambda checked: self.python_execution_test_check.setChecked(
                self.python_execution_test_check.isChecked() if checked else False
            )
        )
        self.python_artifact_check.toggled.connect(
            lambda checked: self.python_full_execution_check.setChecked(
                self.python_full_execution_check.isChecked() if checked else False
            )
        )
        self.latex_artifact_check = QCheckBox("在报告阶段后生成 LaTeX 文档草稿")
        self.latex_artifact_check.setChecked(state.latex_artifact_enabled)
        self.bibtex_artifact_check = QCheckBox("根据已发现论文生成 BibTeX 文献库")
        self.bibtex_artifact_check.setChecked(state.bibtex_artifact_enabled)
        self.latex_compile_check = QCheckBox("在草稿生成后允许本地 Tectonic 编译（需本轮授权）")
        self.latex_compile_check.setChecked(state.latex_compile_enabled)
        self.latex_compile_check.setEnabled(state.latex_artifact_enabled)
        self.latex_artifact_check.toggled.connect(self.latex_compile_check.setEnabled)
        self.latex_artifact_check.toggled.connect(
            lambda checked: self.latex_compile_check.setChecked(
                self.latex_compile_check.isChecked() if checked else False
            )
        )
        tooling_layout.addWidget(self.arxiv_discovery_check)
        tooling_layout.addWidget(self.arxiv_download_check)
        tooling_layout.addWidget(self.python_artifact_check)
        tooling_layout.addWidget(self.python_execution_test_check)
        tooling_layout.addWidget(self.python_full_execution_check)
        tooling_layout.addWidget(self.latex_artifact_check)
        tooling_layout.addWidget(self.bibtex_artifact_check)
        tooling_layout.addWidget(self.latex_compile_check)

        limits_grid = QGridLayout()
        limits_grid.setHorizontalSpacing(12)
        limits_grid.setVerticalSpacing(6)
        limits_grid.setColumnStretch(0, 1)
        limits_grid.setColumnStretch(1, 1)
        limits_grid.addWidget(QLabel("arXiv 最大结果数"), 0, 0)
        limits_grid.addWidget(QLabel("Python 执行超时（秒）"), 0, 1)
        limits_grid.addWidget(self.arxiv_max_results_spin, 1, 0)
        limits_grid.addWidget(self.python_timeout_spin, 1, 1)
        limits_grid.addWidget(QLabel("Python 完整运行超时（秒）"), 2, 0)
        limits_grid.addWidget(QLabel("Python 映射输入上限（MB）"), 2, 1)
        limits_grid.addWidget(self.python_full_timeout_spin, 3, 0)
        limits_grid.addWidget(self.python_input_limit_spin, 3, 1)
        tooling_layout.addLayout(limits_grid)
        left_column.addWidget(tooling_group)

        role_group = QGroupBox("启用角色")
        role_layout = QGridLayout(role_group)
        role_layout.setContentsMargins(14, 14, 14, 14)
        role_layout.setHorizontalSpacing(12)
        role_layout.setVerticalSpacing(8)
        role_layout.setColumnStretch(0, 1)
        role_layout.setColumnStretch(1, 1)
        for index, role in enumerate(workflow_config.team_roles):
            checkbox = QCheckBox(role.duty)
            checkbox.setChecked(role.enabled)
            if role.required:
                checkbox.setToolTip("该角色在默认团队模板中被标记为必需。")
            self.role_checks[role.duty] = checkbox
            role_layout.addWidget(checkbox, index // 2, index % 2)
        right_column.addWidget(role_group)

        summary_group = QGroupBox("结构化摘要槽位")
        summary_layout = QGridLayout(summary_group)
        summary_layout.setContentsMargins(14, 14, 14, 14)
        summary_layout.setHorizontalSpacing(12)
        summary_layout.setVerticalSpacing(8)
        summary_layout.setColumnStretch(0, 1)
        summary_layout.setColumnStretch(1, 1)
        for index, (slot_key, label) in enumerate(SUMMARY_SLOT_LABELS.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(slot_key in state.summary_slots)
            self.summary_slot_checks[slot_key] = checkbox
            summary_layout.addWidget(checkbox, index // 2, index % 2)
        right_column.addWidget(summary_group)
        right_column.addStretch(1)

        tip = QLabel(
            "这些变更会在下一轮讨论中生效。如果关闭过多角色，工作流仍可能运行，但输出会不够完整。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("font-size: 11px; font-family: 'Microsoft YaHei UI'; color: #88A6C7;")
        content_layout.addWidget(tip)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        save_btn = buttons.button(QDialogButtonBox.Save)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        if save_btn is not None:
            save_btn.setText("保存")
        if cancel_btn is not None:
            cancel_btn.setText("取消")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings_state(self) -> WorkflowSettingsState:
        return WorkflowSettingsState(
            max_rounds=self.max_rounds_spin.value(),
            checkpoint_every_n_rounds=self.checkpoint_spin.value(),
            reviewer_enabled=self.reviewer_check.isChecked(),
            enabled_roles={duty: checkbox.isChecked() for duty, checkbox in self.role_checks.items()},
            summary_slots=[slot for slot, checkbox in self.summary_slot_checks.items() if checkbox.isChecked()],
            arxiv_discovery_enabled=self.arxiv_discovery_check.isChecked(),
            arxiv_download_enabled=self.arxiv_download_check.isChecked(),
            arxiv_max_results=self.arxiv_max_results_spin.value(),
            python_artifact_enabled=self.python_artifact_check.isChecked(),
            latex_artifact_enabled=self.latex_artifact_check.isChecked(),
            bibtex_artifact_enabled=self.bibtex_artifact_check.isChecked(),
            python_execution_test_enabled=self.python_execution_test_check.isChecked(),
            python_full_execution_enabled=self.python_full_execution_check.isChecked(),
            python_execution_timeout_seconds=self.python_timeout_spin.value(),
            python_full_execution_timeout_seconds=self.python_full_timeout_spin.value(),
            python_workspace_input_limit_mb=self.python_input_limit_spin.value(),
            latex_compile_enabled=self.latex_compile_check.isChecked(),
        )

    def _accept_if_valid(self) -> None:
        settings = self.settings_state()
        errors = validate_workflow_settings(settings, list(self.role_checks))
        if errors:
            QMessageBox.warning(self, "工作流设置无效", "\n".join(f"- {error}" for error in errors))
            return
        self.accept()


class ChatBubble(QFrame):
    def __init__(
        self,
        speaker: str,
        duty: str,
        body: str,
        meta: str,
        color: str,
        align_right: bool = False,
        emphasis: bool = False,
        theme: str = THEME_DARK,
    ) -> None:
        super().__init__()
        self.align_right = align_right
        self.theme = theme
        self.color = color
        self.emphasis = emphasis
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        outer.setAlignment(Qt.AlignRight if align_right else Qt.AlignLeft)

        self.avatar = QLabel(speaker[:1].upper())
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setFixedSize(42, 42)
        avatar_pixmap = circular_pixmap(avatar_photo_path(speaker, duty), 42)
        if avatar_pixmap is not None:
            self.avatar.setText("")
            self.avatar.setPixmap(avatar_pixmap)
        else:
            self.avatar.setPixmap(QPixmap())

        self.bubble = QFrame()
        self.bubble.setObjectName("messageCard")
        self.bubble.setMinimumWidth(460)
        self.bubble.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)

        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(18, 14, 18, 16)
        bubble_layout.setSpacing(8)

        self.title = QLabel(speaker)
        self.meta_label = QLabel(meta)
        self.content = BubbleText(body, theme=theme)

        bubble_layout.addWidget(self.title)
        bubble_layout.addWidget(self.meta_label)
        bubble_layout.addWidget(self.content)

        if align_right:
            outer.addStretch(1)
            outer.addWidget(self.bubble)
            outer.addWidget(self.avatar)
        else:
            outer.addWidget(self.avatar)
            outer.addWidget(self.bubble)
            outer.addStretch(1)

        self.set_theme(theme)
        self._apply_dynamic_width()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._apply_dynamic_width()

    def _apply_dynamic_width(self) -> None:
        available = max(620, self.width() - 120)
        if available <= 900:
            ratio = 0.78
        elif available >= 1500:
            ratio = 0.70
        else:
            ratio = 0.78 - ((available - 900) / 600) * 0.08
        target_width = int(available * ratio)
        target_width = max(460, min(target_width, available))
        self.bubble.setFixedWidth(target_width)

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        if self.avatar.pixmap() is not None and not self.avatar.pixmap().isNull():
            border_color = "rgba(126, 242, 255, 0.88)" if theme == THEME_DARK else "rgba(86, 163, 214, 0.72)"
            self.avatar.setStyleSheet(f"background: transparent; border: 2px solid {border_color}; border-radius: 21px;")
        else:
            self.avatar.setStyleSheet(
                f"background:{self.color}; color:white; border-radius:21px; font-size: 16px; font-weight: 700; font-family: 'Microsoft YaHei UI';"
            )
        if theme == THEME_DARK:
            background = "rgba(34, 28, 56, 0.95)" if self.emphasis else "rgba(8, 20, 42, 0.94)"
            border = "rgba(154, 110, 255, 0.55)" if self.emphasis else "rgba(83, 216, 255, 0.28)"
            title_color = "#F2FBFF"
            meta_color = "#86A8CC"
        else:
            background = "#FFF2F7" if self.emphasis else "#FFFFFF"
            border = "rgba(214, 99, 142, 0.30)" if self.emphasis else "rgba(156, 184, 209, 0.52)"
            title_color = "#132E4D"
            meta_color = "#68839F"
        self.bubble.setStyleSheet(
            "#messageCard {"
            f"background: {background};"
            f"border: 1px solid {border};"
            "border-radius: 24px;"
            "}"
        )
        self.title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; font-family: 'Microsoft YaHei UI'; color: {title_color};"
        )
        self.meta_label.setStyleSheet(
            f"font-size: 11px; font-family: 'Microsoft YaHei UI'; color: {meta_color};"
        )
        self.content.set_theme(theme)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("赛博课题组")
        if APP_LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_LOGO_PATH)))
        self.resize(1660, 1020)
        self.providers = load_providers()
        self.workflow_config = load_workflow_config()
        self.ui_settings: UiSettings = load_ui_settings()
        self.current_theme = self.ui_settings.theme
        self.attachments: list[AttachmentPayload] = []
        self.provider_cards: list[ProviderSummaryCard] = []
        self.provider_summary_layout: QVBoxLayout | None = None
        self.workflow_summary_label: QLabel | None = None
        self.environment_status_label: QLabel | None = None
        self.theme_combo: QComboBox | None = None
        self.header_strip: BrandHeaderFrame | None = None
        self.chat_surface: ChatSurfaceFrame | None = None
        self.left_title_label: QLabel | None = None
        self.left_subtitle_label: QLabel | None = None
        self.worker_thread: QThread | None = None
        self.worker: DiscussionWorker | None = None
        self.reader_thread: QThread | None = None
        self.reader_worker: PdfReaderWorker | None = None
        self.pdf_reader_button: QPushButton | None = None
        self.local_execution_check: QCheckBox | None = None
        self.stop_requested = False
        self.current_prompt = ""
        self.last_displayed_round: int | None = None
        self.session_messages: list[DiscussionMessage] = []
        self.session_status_lines: list[str] = []
        self.entry_animations: list[QPropertyAnimation] = []
        self._build_ui()
        self._apply_styles()
        self._set_discussion_state("idle")
        QTimer.singleShot(0, self._startup_environment_check)

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        root.setHandleWidth(10)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel())
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([310, 1350])
        self.setCentralWidget(root)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("leftPanel")
        panel.setMinimumWidth(272)
        panel.setMaximumWidth(336)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.left_title_label = QLabel("讨论控制台")
        self.left_title_label.setObjectName("panelTitle")
        layout.addWidget(self.left_title_label)
        layout.addWidget(self._build_provider_group())
        layout.addWidget(self._build_attachment_group())
        layout.addWidget(self._build_workflow_group())
        layout.addStretch(1)
        return panel

    def _build_provider_group(self) -> QWidget:
        group = QGroupBox("角色")
        wrapper_layout = QVBoxLayout(group)
        wrapper_layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        manage_button = QPushButton("打开配置")
        manage_button.setObjectName("ghostAction")
        manage_button.clicked.connect(lambda: self._open_provider_manager())
        save_button = QPushButton("保存")
        save_button.setObjectName("ghostAction")
        save_button.clicked.connect(lambda: self._save_provider_config(show_feedback=True))
        controls.addWidget(manage_button)
        controls.addStretch(1)
        controls.addWidget(save_button)
        wrapper_layout.addLayout(controls)

        hint = QLabel("总览里只保留启用状态与概要信息。点击“配置”可在子界面里编辑详细参数。")
        hint.setObjectName("sectionCaption")
        hint.setWordWrap(True)
        wrapper_layout.addWidget(hint)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        self.provider_summary_layout = content_layout

        self._populate_provider_summary_cards()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(220)
        scroll.setMaximumHeight(392)
        scroll.setWidget(content)
        wrapper_layout.addWidget(scroll, 1)
        return group

    def _build_attachment_group(self) -> QWidget:
        group = QGroupBox("附件")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.attachment_list = QListWidget()
        self.attachment_list.setMinimumHeight(88)
        self.attachment_list.setMaximumHeight(132)
        layout.addWidget(self.attachment_list)

        self.literature_review_check = QCheckBox("启用文献综述")
        self.literature_review_check.setChecked(False)
        self.literature_review_check.setToolTip("启用后，如果已配置文献综述角色并提供参考材料，系统会在主讨论前先生成一份文献综述。")
        layout.addWidget(self.literature_review_check)

        button_row = QHBoxLayout()
        add_button = QPushButton("添加文件")
        remove_button = QPushButton("移除所选")
        add_button.clicked.connect(self._add_attachments)
        remove_button.clicked.connect(self._remove_selected_attachment)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        layout.addLayout(button_row)

        self.pdf_reader_button = QPushButton("构建 PDF Reader")
        self.pdf_reader_button.setObjectName("ghostAction")
        self.pdf_reader_button.setToolTip("构建或刷新 PDF Reader 缓存。后续讨论会从这里检索章节、图示和公式。")
        self.pdf_reader_button.clicked.connect(self._build_pdf_reader_cache)
        layout.addWidget(self.pdf_reader_button)
        return group

    def _build_workflow_group(self) -> QWidget:
        group = QGroupBox("工作流策略")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        appearance_form = QFormLayout()
        appearance_form.setContentsMargins(0, 0, 0, 0)
        appearance_form.setSpacing(8)
        self.theme_combo = ChevronComboBox(self.current_theme)
        self.theme_combo.addItem(THEME_LABELS[THEME_DARK], THEME_DARK)
        self.theme_combo.addItem(THEME_LABELS[THEME_LIGHT], THEME_LIGHT)
        current_index = self.theme_combo.findData(self.current_theme)
        self.theme_combo.setCurrentIndex(current_index if current_index >= 0 else 0)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        appearance_form.addRow("Theme", self.theme_combo)
        layout.addLayout(appearance_form)

        self.workflow_summary_label = QLabel()
        self.workflow_summary_label.setWordWrap(True)
        self.workflow_summary_label.setObjectName("workflowSummary")
        layout.addWidget(self.workflow_summary_label)

        self.environment_status_label = QLabel()
        self.environment_status_label.setWordWrap(True)
        self.environment_status_label.setObjectName("panelHint")
        layout.addWidget(self.environment_status_label)

        self.local_execution_check = QCheckBox("为本轮授权本地执行")
        self.local_execution_check.setChecked(False)
        self.local_execution_check.setToolTip(
            f"只有勾选后，应用才能运行生成的 Python 代码或启动本地 Tectonic 编译。Python 将使用当前解释器：{sys.executable}"
        )
        layout.addWidget(self.local_execution_check)

        button_column = QVBoxLayout()
        button_column.setContentsMargins(0, 0, 0, 0)
        button_column.setSpacing(6)

        edit_button = QPushButton("工作流设置")
        edit_button.setObjectName("ghostAction")
        edit_button.setToolTip("打开工作流策略对话框")
        edit_button.clicked.connect(self._edit_workflow_settings)
        export_button = QPushButton("导出流程图")
        export_button.setObjectName("ghostAction")
        export_button.setToolTip("导出当前工作流图和策略快照")
        export_button.clicked.connect(self._export_workflow_graph_snapshot)
        reload_button = QPushButton("重新加载配置")
        reload_button.setObjectName("ghostAction")
        reload_button.setToolTip("从磁盘重新加载 workflow_config.json")
        reload_button.clicked.connect(lambda: self._reload_workflow_config(show_feedback=True))
        for button in (edit_button, export_button, reload_button):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button_column.addWidget(button)
        layout.addLayout(button_column)

        self._refresh_workflow_summary()
        self._refresh_environment_status(show_warning=False)
        return group

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QVBoxLayout(top_bar)
        top_layout.setContentsMargins(22, 20, 22, 20)
        top_layout.setSpacing(14)

        self.header_strip = BrandHeaderFrame(theme=self.current_theme)
        self.header_strip.setObjectName("headerStrip")
        header_strip_layout = QVBoxLayout(self.header_strip)
        header_strip_layout.setContentsMargins(22, 18, 22, 18)
        header_strip_layout.setSpacing(10)
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(18)
        logo_label = QLabel()
        logo_pixmap = build_logo_pixmap(96, 96)
        if logo_pixmap is not None:
            logo_label.setPixmap(logo_pixmap)
            logo_label.setFixedSize(108, 108)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setObjectName("brandLogo")
            brand_row.addWidget(logo_label, 0, Qt.AlignTop)
        text_stack = QVBoxLayout()
        text_stack.setContentsMargins(0, 0, 0, 0)
        text_stack.setSpacing(4)
        title = QLabel("赛博课题组")
        title.setObjectName("heroTitle")
        apply_outer_glow(title, QColor(116, 238, 255), blur_radius=24, alpha=118)
        subtitle = QLabel("你的个人 AI 科研团队")
        subtitle.setObjectName("heroSubtitle")
        apply_outer_glow(subtitle, QColor(82, 204, 255), blur_radius=14, alpha=72)
        text_stack.addWidget(title)
        text_stack.addWidget(subtitle)
        brand_row.addLayout(text_stack, 1)
        header_strip_layout.addLayout(brand_row)
        top_layout.addWidget(self.header_strip)

        composer_card = QFrame()
        composer_card.setObjectName("composerCard")
        composer_layout = QHBoxLayout(composer_card)
        composer_layout.setContentsMargins(18, 18, 18, 18)
        composer_layout.setSpacing(14)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setObjectName("promptEdit")
        self.prompt_edit.setPlaceholderText("例如：围绕某个科研问题展开讨论。可以先阅读参考材料，再拆解子任务、分工协作、交叉复核并输出报告。")
        self.prompt_edit.setFixedHeight(100)
        composer_layout.addWidget(self.prompt_edit, 1)

        action_column = QVBoxLayout()
        action_column.setSpacing(10)
        flow_label = QLabel("执行模式")
        flow_label.setObjectName("sectionCaption")
        flow_value = QLabel("自动任务流")
        flow_value.setObjectName("flowValue")
        self.start_button = GlowButton("开始讨论", QColor(102, 231, 255))
        self.start_button.setObjectName("primaryAction")
        self.start_button.clicked.connect(self._start_discussion)
        self.stop_button = QPushButton("待机")
        self.stop_button.setObjectName("secondaryAction")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_discussion)
        action_column.addWidget(flow_label)
        action_column.addWidget(flow_value)
        action_column.addStretch(1)
        action_column.addWidget(self.start_button)
        action_column.addWidget(self.stop_button)
        composer_layout.addLayout(action_column)
        top_layout.addWidget(composer_card)
        layout.addWidget(top_bar)

        self.chat_surface = ChatSurfaceFrame(theme=self.current_theme)
        self.chat_surface.setObjectName("chatSurface")
        thread_layout = QVBoxLayout(self.chat_surface)
        thread_layout.setContentsMargins(0, 0, 0, 0)
        thread_layout.setSpacing(12)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_container = QWidget()
        self.chat_container.setObjectName("chatCanvas")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(12, 18, 12, 18)
        self.chat_layout.setSpacing(14)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_container)
        thread_layout.addWidget(self.chat_scroll, 1)
        layout.addWidget(self.chat_surface, 1)

        self._append_status_card("System", "先导入材料，再开始讨论。你也可以先启用文献综述，让系统整理参考资料后再由 Lead 拆题、专家分工并交叉复核。")
        return panel

    def _apply_styles(self) -> None:
        if self.current_theme == THEME_DARK:
            stylesheet = _dark_stylesheet()
        else:
            stylesheet = _light_stylesheet()
        self.setStyleSheet(stylesheet)
        self._apply_theme_to_custom_widgets()

    def _apply_theme_to_custom_widgets(self) -> None:
        if self.header_strip is not None:
            self.header_strip.set_theme(self.current_theme)
        if self.chat_surface is not None:
            self.chat_surface.set_theme(self.current_theme)
        if self.theme_combo is not None and isinstance(self.theme_combo, ChevronComboBox):
            self.theme_combo.set_theme(self.current_theme)
        for card in self.provider_cards:
            card.set_theme(self.current_theme)
        for index in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(index)
            widget = item.widget()
            if widget is not None and hasattr(widget, "set_theme"):
                widget.set_theme(self.current_theme)

    def _on_theme_changed(self) -> None:
        if self.theme_combo is None:
            return
        selected_theme = str(self.theme_combo.currentData() or THEME_DARK)
        if selected_theme == self.current_theme:
            return
        self.current_theme = selected_theme
        self.ui_settings = UiSettings(theme=selected_theme)
        save_ui_settings(self.ui_settings)
        self._apply_styles()

    def _set_discussion_state(self, state: str) -> None:
        if state == "idle":
            self.start_button.setEnabled(True)
            self.start_button.setText("开始讨论")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("待机")
            self.start_button.setProperty("state", "idle")
            self.stop_button.setProperty("state", "idle")
        elif state == "running":
            self.start_button.setEnabled(False)
            self.start_button.setText("讨论进行中")
            self.stop_button.setEnabled(True)
            self.stop_button.setText("停止讨论")
            self.start_button.setProperty("state", "running")
            self.stop_button.setProperty("state", "running")
        elif state == "stopping":
            self.start_button.setEnabled(False)
            self.start_button.setText("讨论进行中")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("正在停止")
            self.start_button.setProperty("state", "running")
            self.stop_button.setProperty("state", "stopping")

        for button in (self.start_button, self.stop_button):
            self.style().unpolish(button)
            self.style().polish(button)
            button.update()

    def _open_provider_manager(self, start_index: int = 0) -> None:
        if not self.providers:
            QMessageBox.information(self, "暂无角色", "当前没有可配置的角色。")
            return
        dialog = ProviderManagerDialog(
            self.providers,
            start_index=start_index,
            theme=self.current_theme,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self._refresh_provider_cards(rebuild=True)
        self._save_provider_config(show_feedback=False)

    def _populate_provider_summary_cards(self) -> None:
        if self.provider_summary_layout is None:
            return
        while self.provider_summary_layout.count():
            item = self.provider_summary_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.provider_cards = []
        for index, provider in enumerate(self.providers):
            card = ProviderSummaryCard(provider, index)
            card.edit_requested.connect(self._open_provider_manager)
            card.set_theme(self.current_theme)
            self.provider_cards.append(card)
            self.provider_summary_layout.addWidget(card)
        self.provider_summary_layout.addStretch(1)

    def _refresh_provider_cards(self, rebuild: bool = False) -> None:
        if rebuild or len(self.provider_cards) != len(self.providers):
            self._populate_provider_summary_cards()
            return
        for card in self.provider_cards:
            card.refresh_from_provider()

    def _save_provider_config(self, show_feedback: bool = False) -> None:
        for card in self.provider_cards:
            card.apply_to_provider()
        save_providers(self.providers)
        if show_feedback:
            QMessageBox.information(self, "配置已保存", "角色配置已写入 app_config.json。")

    def _refresh_workflow_summary(self) -> None:
        if self.workflow_summary_label is None:
            return
        self.workflow_summary_label.setText(render_workflow_settings_summary(self.workflow_config))

    def _refresh_environment_status(self, *, show_warning: bool) -> None:
        tectonic_path = shutil.which("tectonic")
        if self.environment_status_label is not None:
            if tectonic_path:
                self.environment_status_label.setText("环境：已检测到 Tectonic")
                self.environment_status_label.setToolTip(tectonic_path)
            else:
                self.environment_status_label.setText("环境：未检测到 Tectonic")
                self.environment_status_label.setToolTip("安装 Tectonic 并加入 PATH 后，才能启用本地 LaTeX 编译。")
        if show_warning and self.workflow_config.tooling.enable_latex_compile and tectonic_path is None:
            QMessageBox.warning(
                self,
                "未找到 Tectonic",
                "当前工作流设置允许本地 Tectonic 编译，但系统 PATH 中未找到 `tectonic`。请安装 Tectonic，或在工作流设置中关闭本地 TeX 编译。",
            )

    def _startup_environment_check(self) -> None:
        self._refresh_environment_status(show_warning=True)
        if shutil.which("tectonic") is None:
            self._append_status_card(
                "System",
                "启动环境检查：PATH 中未找到 Tectonic。讨论可以继续，但在安装编译器前会跳过本地 LaTeX 编译。",
            )

    def _reload_workflow_config(self, show_feedback: bool = False) -> None:
        self.workflow_config = load_workflow_config()
        self._refresh_workflow_summary()
        self._refresh_environment_status(show_warning=show_feedback)
        if show_feedback:
            QMessageBox.information(self, "已重新加载工作流配置", "工作流设置已从 workflow_config.json 重新加载。")

    def _export_workflow_graph_snapshot(self) -> None:
        WORKFLOW_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_graph = build_workflow_graph(self.workflow_config.workflow_template)
        graph_path = WORKFLOW_EXPORT_DIR / f"workflow_graph_{timestamp}.json"
        mermaid_path = WORKFLOW_EXPORT_DIR / f"workflow_graph_{timestamp}.mmd"
        policy_path = WORKFLOW_EXPORT_DIR / f"workflow_policy_{timestamp}.json"

        graph_path.write_text(json.dumps(workflow_graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        mermaid_path.write_text(render_workflow_graph_mermaid(workflow_graph), encoding="utf-8")
        policy_path.write_text(json.dumps(workflow_policy_snapshot(self.workflow_config), ensure_ascii=False, indent=2), encoding="utf-8")

        self._append_status_card(
            "System",
            "已导出工作流快照：\n"
            f"- JSON：`{graph_path}`\n"
            f"- Mermaid：`{mermaid_path}`\n"
            f"- 策略快照：`{policy_path}`",
        )
        QMessageBox.information(
            self,
            "工作流图已导出",
            "当前工作流图 JSON、Mermaid 图和策略快照已保存到 generated_artifacts/workflow_graph_exports。",
        )

    def _edit_workflow_settings(self) -> None:
        dialog = WorkflowSettingsDialog(self.workflow_config, self)
        if dialog.exec() != QDialog.Accepted:
            return

        try:
            self.workflow_config = apply_workflow_settings(self.workflow_config, dialog.settings_state())
            save_workflow_config(self.workflow_config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存工作流设置失败", str(exc))
            return

        self._refresh_workflow_summary()
        self._refresh_environment_status(show_warning=True)
        QMessageBox.information(
            self,
            "工作流配置已保存",
            "工作流设置已写入 workflow_config.json，并将在下一轮讨论中生效。",
        )

    def _refresh_attachment_list(self) -> None:
        self.attachment_list.clear()
        for attachment in self.attachments:
            kind_label = {
                "pdf": "PDF",
                "text": "文本",
                "image": "图片",
            }.get(attachment.kind, attachment.kind)
            label = f"{attachment.display_name} [{kind_label}]"
            if attachment.kind == "pdf":
                badge = pdf_reader_badge(attachment.path)
                if badge:
                    label += f" [{badge}]"
            self.attachment_list.addItem(QListWidgetItem(label))

    def _add_attachments(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择附件",
            "",
            "支持的文件 (*.pdf *.txt *.md *.png *.jpg *.jpeg *.webp *.bmp *.json *.csv)",
        )
        for path in paths:
            try:
                payload = load_attachment(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "加载附件失败", f"{Path(path).name}: {exc}")
                continue
            self.attachments.append(payload)
        self._refresh_attachment_list()

    def _remove_selected_attachment(self) -> None:
        row = self.attachment_list.currentRow()
        if row < 0:
            return
        del self.attachments[row]
        self._refresh_attachment_list()

    def _select_pdf_reader_provider(self) -> ProviderConfig | None:
        for provider in self.providers:
            if provider.duty == LITERATURE_DUTY and provider.api_key:
                return provider
        return None

    def _build_pdf_reader_cache(self) -> None:
        if self.reader_worker is not None:
            return
        self._save_provider_config(show_feedback=False)
        pdf_attachments = [attachment for attachment in self.attachments if attachment.kind == "pdf"]
        if not pdf_attachments:
            QMessageBox.warning(self, "未附加 PDF", "请先附加至少一个 PDF，再构建 PDF Reader 缓存。")
            return

        provider = self._select_pdf_reader_provider()
        if provider is None:
            self._append_status_card(
                "System",
                "未找到带 API Key 的文献综述角色，将以仅索引模式构建本地 PDF 章节索引。",
            )
        else:
            self._append_status_card(
                "System",
                f"正在使用 {provider.name} 为 {len(pdf_attachments)} 个 PDF 构建 PDF Reader 摘要。",
            )

        if self.pdf_reader_button is not None:
            self.pdf_reader_button.setEnabled(False)
            self.pdf_reader_button.setText("正在构建 PDF Reader")

        self.reader_thread = QThread()
        self.reader_worker = PdfReaderWorker(provider, pdf_attachments)
        self.reader_worker.moveToThread(self.reader_thread)
        self.reader_thread.started.connect(self.reader_worker.run)
        self.reader_worker.status_ready.connect(self._on_pdf_reader_status)
        self.reader_worker.finished.connect(self._on_pdf_reader_finished)
        self.reader_worker.failed.connect(self._on_pdf_reader_failed)
        self.reader_worker.finished.connect(self.reader_thread.quit)
        self.reader_worker.failed.connect(self.reader_thread.quit)
        self.reader_thread.finished.connect(self._cleanup_reader_worker)
        self.reader_thread.start()

    def _start_discussion(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "缺少问题", "请先输入研究问题或讨论目标。")
            return

        self._save_provider_config(show_feedback=False)
        active_providers = [provider for provider in self.providers if provider.enabled and provider.api_key]
        if not active_providers:
            QMessageBox.warning(self, "没有可用模型", "请至少启用并配置一个带 API Key 的模型。")
            return

        self.current_prompt = prompt
        self.stop_requested = False
        self.last_displayed_round = None
        self.session_messages = []
        self.session_status_lines = []
        self._set_discussion_state("running")
        self._reset_chat()
        self._append_time_divider()
        self._append_message(
            speaker="我",
            duty="USER",
            body=prompt,
            meta=f"用户输入 | {len(self.attachments)} 个附件",
            color=color_for_speaker("USER"),
            align_right=True,
        )
        self._append_status_card("System", "讨论已开始。Lead 会先按专长拆解任务，再由 Host 组织执行流程。")
        if self.literature_review_check.isChecked():
            self._append_status_card("System", "已启用文献综述。如果存在文献综述角色和参考附件，系统会在专家组介入前先生成综述。")
        self._append_status_card("System", "专家组会围绕具体子问题展开分析并相互复核，Reporter 会实时更新日志并在最后生成输出文件。")
        if self.local_execution_check is not None and self.local_execution_check.isChecked():
            execution_modes: list[str] = []
            if self.workflow_config.tooling.enable_python_execution_test:
                execution_modes.append("Python 冒烟测试")
            if self.workflow_config.tooling.enable_python_full_execution:
                execution_modes.append("Python 完整运行")
            if self.workflow_config.tooling.enable_latex_compile:
                execution_modes.append("Tectonic 编译")
            enabled_text = "、".join(execution_modes) or "本地执行步骤"
            self._append_status_card(
                "System",
                f"本轮已授权本地执行。应用可以使用当前解释器 `{sys.executable}` 运行：{enabled_text}。",
            )
        else:
            self._append_status_card(
                "System",
                "本轮未授权本地执行。生成的代码和 LaTeX 产物会被保存，但会跳过 Python 执行和 Tectonic 编译。",
            )

        self.worker_thread = QThread()
        self.worker = DiscussionWorker(
            active_providers,
            prompt,
            self.attachments,
            self.literature_review_check.isChecked(),
            bool(self.local_execution_check is not None and self.local_execution_check.isChecked()),
            self.workflow_config,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.message_ready.connect(self._on_discussion_message)
        self.worker.status_ready.connect(self._on_discussion_status)
        self.worker.finished.connect(self._on_discussion_finished)
        self.worker.failed.connect(self._on_discussion_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _stop_discussion(self) -> None:
        if self.worker is None:
            return
        self.stop_requested = True
        self.worker.stop()
        self._set_discussion_state("stopping")
        self._append_status_card("System", "已请求停止。当前模型调用会先完成，随后剩余工作流会停止。")

    def _on_discussion_message(self, message: DiscussionMessage) -> None:
        self.session_messages.append(message)
        if message.round_index > 0 and message.round_index != self.last_displayed_round:
            self.last_displayed_round = message.round_index
            self._append_task_divider(message.round_index)

        if message.duty == LEAD_DUTY:
            meta = "Lead 派工"
        elif message.duty == HOST_DUTY:
            meta = "Host 协调"
        elif message.duty == LITERATURE_DUTY:
            meta = "文献综述"
        elif message.duty == REPORT_DUTY:
            meta = "实时日志" if message.stage == "log" else "报告输出"
        elif message.stage == "review":
            meta = "专家复核"
        elif message.duty == EXPERT_DUTY:
            meta = "专家执行"
        else:
            meta = message.stage or "讨论消息"
        meta = f"{meta} | {message.model_name}"
        self._append_message(
            speaker=message.speaker,
            duty=message.duty,
            body=message.content,
            meta=meta,
            color=color_for_speaker(message.speaker),
            align_right=False,
            emphasis=message.content.startswith(("[Call Failed]", "[调用失败]")),
        )

    def _on_discussion_status(self, text: str) -> None:
        self._append_status_card("System", text)

    def _on_discussion_finished(self, result: DiscussionResult) -> None:
        try:
            literature_path, minutes_path, report_path = save_discussion_outputs(
                user_request=self.current_prompt,
                providers=self.providers,
                messages=result.messages,
                literature_review_text=result.literature_review or "",
                summary_text=result.final_summary or "未生成研究报告。",
                minutes_text=result.meeting_minutes or result.final_summary or "未生成会议纪要。",
                cancelled=result.cancelled,
                meeting_state=result.meeting_state,
                workflow_config=self.workflow_config,
            )
            result.literature_review_path = str(literature_path) if literature_path is not None else ""
            result.meeting_minutes_path = str(minutes_path)
            result.summary_path = str(report_path)
            result.report_path = str(report_path)
            saved_lines = []
            if literature_path is not None:
                saved_lines.append(f"- 文献综述：`{literature_path}`")
            saved_lines.append(f"- 会议纪要：`{minutes_path}`")
            saved_lines.append(f"- 研究报告：`{report_path}`")
            if result.meeting_state is not None:
                for artifact in result.meeting_state.generated_artifacts:
                    if artifact.path in {str(literature_path) if literature_path is not None else "", str(minutes_path), str(report_path)}:
                        continue
                    saved_lines.append(f"- {artifact.title}: `{artifact.path}`")
            self._append_status_card(
                "System",
                "已保存本地文件：\n" + "\n".join(saved_lines),
            )
        except Exception as exc:  # noqa: BLE001
            self._append_status_card("System", f"保存本地文件失败：{exc}")
        self._set_discussion_state("idle")
        self._scroll_chat_to_bottom()

    def _on_discussion_failed(self, payload: object) -> None:
        error_text = str(payload)
        traceback_text = error_text
        partial_result = None
        if isinstance(payload, dict):
            error_text = str(payload.get("error_text") or error_text)
            traceback_text = str(payload.get("traceback") or error_text)
            partial_result = payload.get("partial_result")

        self._append_status_card("System", f"讨论失败：\n\n{error_text}")
        try:
            saved_messages: list[DiscussionMessage] = []
            meeting_state = None
            literature_review_text = ""
            if isinstance(partial_result, DiscussionResult):
                saved_messages = partial_result.messages or []
                meeting_state = partial_result.meeting_state
                literature_review_text = partial_result.literature_review or ""
            if not saved_messages:
                saved_messages = list(self.session_messages)
            if not literature_review_text:
                literature_review_text = self._extract_session_literature_review(saved_messages)
            failure_path = save_failure_snapshot(
                user_request=self.current_prompt,
                providers=self.providers,
                messages=saved_messages,
                status_lines=self.session_status_lines,
                error_text=traceback_text,
                literature_review_text=literature_review_text,
                meeting_state=meeting_state,
                workflow_config=self.workflow_config,
            )
            self._append_status_card("System", f"已保存失败快照：`{failure_path}`")
        except Exception as exc:  # noqa: BLE001
            self._append_status_card("System", f"保存失败快照失败：{exc}")
        self._set_discussion_state("idle")
        self._scroll_chat_to_bottom()

    def _on_pdf_reader_status(self, text: str) -> None:
        self._append_status_card("System", text)

    def _on_pdf_reader_finished(self, payload: object) -> None:
        results = [item for item in (payload or []) if isinstance(item, PdfReaderBuildResult)]
        self._refresh_attachment_list()
        if results:
            saved_lines = [
                f"- {Path(item.source_pdf).name} | 摘要：`{item.digest_markdown_path}` | 索引：`{item.index_path}`"
                for item in results
            ]
            self._append_status_card("System", "PDF Reader 产物已保存：\n" + "\n".join(saved_lines))
        else:
            self._append_status_card("System", "PDF Reader 构建已完成，但没有生成 PDF 产物。")

    def _on_pdf_reader_failed(self, error_text: str) -> None:
        self._append_status_card("System", f"PDF Reader 构建失败：\n\n{error_text}")
    def _cleanup_reader_worker(self) -> None:
        self.reader_worker = None
        self.reader_thread = None
        if self.pdf_reader_button is not None:
            self.pdf_reader_button.setEnabled(True)
            self.pdf_reader_button.setText("构建 PDF Reader")

    def _cleanup_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.stop_requested = False

    def _reset_chat(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _append_message(
        self,
        speaker: str,
        duty: str,
        body: str,
        meta: str,
        color: str,
        align_right: bool,
        emphasis: bool = False,
    ) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        bubble = ChatBubble(
            speaker=speaker,
            duty=duty,
            body=body,
            meta=f"{meta} | {timestamp}",
            color=color,
            align_right=align_right,
            emphasis=emphasis,
            theme=self.current_theme,
        )
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._animate_message_entry(bubble)
        self._scroll_chat_to_bottom()

    def _append_status_card(self, speaker: str, body: str) -> None:
        display_speaker = {"System": "系统", "You": "我"}.get(speaker, speaker)
        self.session_status_lines.append(f"{display_speaker}: {body}")
        self._append_message(
            speaker=display_speaker,
            duty=HOST_DUTY,
            body=body,
            meta="系统提示",
            color=color_for_speaker("HOST"),
            align_right=False,
        )

    def _extract_session_literature_review(self, messages: list[DiscussionMessage]) -> str:
        for message in reversed(messages):
            if message.duty == LITERATURE_DUTY or message.stage == "literature_review":
                return message.content
        return ""

    def _append_time_divider(self) -> None:
        divider = TimelineDivider(datetime.now().strftime("%Y-%m-%d %H:%M"), accent=False, theme=self.current_theme)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, divider)
        self._scroll_chat_to_bottom()

    def _append_task_divider(self, round_index: int) -> None:
        divider = TimelineDivider(self._round_divider_label(round_index), accent=True, theme=self.current_theme)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, divider)
        self._scroll_chat_to_bottom()

    def _round_divider_label(self, round_index: int) -> str:
        return f"第 {round_index} 轮讨论"

    def _round_meta_label(self, round_index: int) -> str:
        if round_index <= 0:
            return ""
        return f"第 {round_index} 轮"

    def _scroll_chat_to_bottom(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _animate_message_entry(self, widget: QWidget) -> None:
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        self.entry_animations.append(animation)

        def _cleanup() -> None:
            if widget.graphicsEffect() is effect:
                widget.setGraphicsEffect(None)
            if animation in self.entry_animations:
                self.entry_animations.remove(animation)

        animation.finished.connect(_cleanup)
        animation.start()
