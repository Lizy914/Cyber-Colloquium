from __future__ import annotations

import re
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event

from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QThread, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
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
    QSplitter,
    QTextBrowser,
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


APP_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PHOTO_DIR = APP_ROOT / "Profile Photo"
APP_LOGO_PATH = APP_ROOT / "Overall Picture.png"
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
    ) -> None:
        super().__init__()
        self.providers = providers
        self.prompt = prompt
        self.attachments = attachments
        self.generate_literature_review = generate_literature_review
        self.cancel_event = Event()

    def run(self) -> None:
        orchestrator = DiscussionOrchestrator(self.providers)
        try:
            result = orchestrator.run_discussion(
                user_request=self.prompt,
                attachments=self.attachments,
                generate_literature_review=self.generate_literature_review,
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


class BubbleText(QTextBrowser):
    def __init__(self, body: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.document().setDocumentMargin(0)
        self.setStyleSheet(
            "background: transparent; border: none; color: #E6F6FF; font-size: 13px; font-family: 'Microsoft YaHei UI';"
        )
        self.setMarkdown(body)
        self._fit_height()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._fit_height()

    def _fit_height(self) -> None:
        width = max(320, self.viewport().width())
        self.document().setTextWidth(width)
        height = int(self.document().size().height()) + 10
        self.setMinimumHeight(max(52, height))
        self.setMaximumHeight(max(52, height))


class TimelineDivider(QFrame):
    def __init__(self, text: str, accent: bool = False) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(10)

        left_line = QFrame()
        left_line.setFrameShape(QFrame.HLine)
        left_line.setStyleSheet("color: rgba(83, 216, 255, 0.34);")
        right_line = QFrame()
        right_line.setFrameShape(QFrame.HLine)
        right_line.setStyleSheet("color: rgba(83, 216, 255, 0.34);")

        chip = QLabel(text)
        chip.setAlignment(Qt.AlignCenter)
        chip.setStyleSheet(
            "background: %s; color: %s; border-radius: 999px; padding: 6px 14px; font-size: 11px; font-weight: 700; font-family: 'Microsoft YaHei UI';"
            % (
                "rgba(17, 77, 122, 0.88)" if accent else "rgba(8, 29, 60, 0.88)",
                "#89E8FF" if accent else "#A7C5E5",
            )
        )

        layout.addWidget(left_line, 1)
        layout.addWidget(chip)
        layout.addWidget(right_line, 1)


class BrandHeaderFrame(QFrame):
    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 30, 30)

        base = QLinearGradient(rect.topLeft(), rect.bottomRight())
        base.setColorAt(0.0, QColor(8, 24, 49, 248))
        base.setColorAt(0.52, QColor(15, 43, 86, 242))
        base.setColorAt(1.0, QColor(22, 31, 72, 244))
        painter.fillPath(path, base)

        painter.setPen(Qt.NoPen)
        painter.setClipPath(path)

        glow_specs = [
            (rect.left() + 180, rect.top() + 40, 180, 120, QColor(93, 222, 255, 32)),
            (rect.center().x() - 90, rect.top() + 18, 240, 110, QColor(123, 184, 255, 28)),
            (rect.right() - 260, rect.bottom() - 72, 220, 120, QColor(118, 98, 255, 24)),
        ]
        for x, y, w, h, color in glow_specs:
            painter.setBrush(color)
            painter.drawEllipse(x, y, w, h)

        line_pen = QPen(QColor(119, 233, 255, 110), 1.4)
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
            painter.setBrush(QColor(137, 232, 255, 140))
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
            painter.setBrush(QColor(137, 232, 255, 140))
            painter.drawEllipse(node_x, node_y, 5, 5)

        painter.setClipping(False)
        border_pen = QPen(QColor(89, 214, 255, 56), 1.2)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 30, 30)
        painter.end()


class ChatSurfaceFrame(QFrame):
    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 28, 28)

        fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        fill.setColorAt(0.0, QColor(7, 17, 34, 232))
        fill.setColorAt(1.0, QColor(3, 10, 22, 240))
        painter.fillPath(path, fill)
        painter.setClipPath(path)

        painter.setPen(QPen(QColor(88, 204, 255, 18), 1))
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
            alpha = 52 if radius == 2 else 26
            painter.setBrush(QColor(153, 235, 255, alpha))
            painter.drawEllipse(x, y, radius + 1, radius + 1)

        circuit_pen = QPen(QColor(84, 221, 255, 36), 1.2)
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
        painter.setPen(QPen(QColor(89, 214, 255, 42), 1.0))
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


class ProviderConfigCard(QFrame):
    def __init__(self, provider: ProviderConfig, expanded: bool = False) -> None:
        super().__init__()
        self.provider = provider
        self.setObjectName("providerCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)

        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(provider.enabled)

        self.name_edit = QLineEdit(provider.name)
        self.name_edit.setObjectName("roleNameEdit")
        self.name_edit.setPlaceholderText("Role name")

        self.collapse_button = QToolButton()
        self.collapse_button.setObjectName("collapseToggle")
        self.collapse_button.setCheckable(True)
        self.collapse_button.toggled.connect(self.set_expanded)

        header.addWidget(self.enabled_check)
        header.addWidget(self.name_edit, 1)
        header.addWidget(self.collapse_button)
        layout.addLayout(header)

        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.model_edit = QLineEdit(provider.model)
        self.base_url_edit = QLineEdit(provider.base_url)
        self.api_key_edit = QLineEdit(provider.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.duty_combo = QComboBox()
        self.duty_combo.addItems(DUTY_OPTIONS)
        self.duty_combo.setCurrentText(provider.duty if provider.duty in DUTY_OPTIONS else EXPERT_DUTY)
        self.specialty_edit = QLineEdit(provider.specialty)
        self.specialty_edit.setPlaceholderText("e.g. literature review, derivation, experiment design, critical review")
        self.vision_check = QCheckBox("Enable vision")
        self.vision_check.setChecked(provider.supports_vision)

        form.addRow("Duty", self.duty_combo)
        form.addRow("Specialty", self.specialty_edit)
        form.addRow("Model", self.model_edit)
        form.addRow("Base URL", self.base_url_edit)
        form.addRow("API Key", self.api_key_edit)
        form.addRow("Capability", self.vision_check)

        layout.addWidget(self.body)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.collapse_button.blockSignals(True)
        self.collapse_button.setChecked(expanded)
        self.collapse_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.collapse_button.setText("Collapse" if expanded else "Expand")
        self.collapse_button.blockSignals(False)

    def apply_to_provider(self) -> ProviderConfig:
        self.provider.enabled = self.enabled_check.isChecked()
        self.provider.name = self.name_edit.text().strip() or self.provider.name
        self.provider.duty = self.duty_combo.currentText()
        self.provider.specialty = self.specialty_edit.text().strip()
        self.provider.model = self.model_edit.text().strip()
        self.provider.base_url = self.base_url_edit.text().strip()
        self.provider.api_key = self.api_key_edit.text().strip()
        self.provider.supports_vision = self.vision_check.isChecked()
        return self.provider


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
    ) -> None:
        super().__init__()
        self.align_right = align_right
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        outer.setAlignment(Qt.AlignRight if align_right else Qt.AlignLeft)

        avatar = QLabel(speaker[:1].upper())
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setFixedSize(42, 42)
        avatar_pixmap = circular_pixmap(avatar_photo_path(speaker, duty), 42)
        if avatar_pixmap is not None:
            avatar.setText("")
            avatar.setPixmap(avatar_pixmap)
            avatar.setStyleSheet("background: transparent; border: 2px solid rgba(126, 242, 255, 0.88); border-radius: 21px;")
        else:
            avatar.setStyleSheet(
                f"background:{color}; color:white; border-radius:21px; font-size: 16px; font-weight: 700; font-family: 'Microsoft YaHei UI';"
            )

        self.bubble = QFrame()
        self.bubble.setObjectName("messageCard")
        self.bubble.setMinimumWidth(460)
        self.bubble.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        self.bubble.setStyleSheet(
            "#messageCard {"
            f"background: {'rgba(34, 28, 56, 0.95)' if emphasis else 'rgba(8, 20, 42, 0.94)'};"
            f"border: 1px solid {'rgba(154, 110, 255, 0.55)' if emphasis else 'rgba(83, 216, 255, 0.28)'};"
            "border-radius: 24px;"
            "}"
        )

        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(18, 14, 18, 16)
        bubble_layout.setSpacing(8)

        title = QLabel(speaker)
        title.setStyleSheet("font-size: 14px; font-weight: 700; font-family: 'Microsoft YaHei UI'; color: #F2FBFF;")
        meta_label = QLabel(meta)
        meta_label.setStyleSheet("font-size: 11px; font-family: 'Microsoft YaHei UI'; color: #86A8CC;")
        content = BubbleText(body)

        bubble_layout.addWidget(title)
        bubble_layout.addWidget(meta_label)
        bubble_layout.addWidget(content)

        if align_right:
            outer.addStretch(1)
            outer.addWidget(self.bubble)
            outer.addWidget(avatar)
        else:
            outer.addWidget(avatar)
            outer.addWidget(self.bubble)
            outer.addStretch(1)

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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cyber Colloquium")
        if APP_LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_LOGO_PATH)))
        self.resize(1660, 1020)
        self.providers = load_providers()
        self.attachments: list[AttachmentPayload] = []
        self.provider_cards: list[ProviderConfigCard] = []
        self.worker_thread: QThread | None = None
        self.worker: DiscussionWorker | None = None
        self.reader_thread: QThread | None = None
        self.reader_worker: PdfReaderWorker | None = None
        self.pdf_reader_button: QPushButton | None = None
        self.stop_requested = False
        self.current_prompt = ""
        self.last_displayed_round: int | None = None
        self.session_messages: list[DiscussionMessage] = []
        self.session_status_lines: list[str] = []
        self.entry_animations: list[QPropertyAnimation] = []
        self._build_ui()
        self._apply_styles()
        self._set_discussion_state("idle")

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        root.setHandleWidth(10)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel())
        root.setSizes([360, 1300])
        self.setCentralWidget(root)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        title = QLabel("Discussion Console")
        title.setStyleSheet("font-size: 24px; font-weight: 800; font-family: 'Microsoft YaHei UI'; color: #F2FBFF;")
        subtitle = QLabel("Configure duties and specialties so the models can work like an academic team: divide tasks, challenge claims, review literature, and correct each other.")
        subtitle.setStyleSheet("font-size: 12px; font-family: 'Microsoft YaHei UI'; color: #9DB7D3;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._build_provider_group())
        layout.addWidget(self._build_attachment_group())
        layout.addStretch(1)
        return panel

    def _build_provider_group(self) -> QWidget:
        group = QGroupBox("Roles")
        wrapper_layout = QVBoxLayout(group)
        wrapper_layout.setSpacing(12)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        expand_button = QPushButton("Expand all")
        expand_button.setObjectName("ghostAction")
        expand_button.clicked.connect(lambda: self._set_provider_cards_expanded(True))
        collapse_button = QPushButton("Collapse all")
        collapse_button.setObjectName("ghostAction")
        collapse_button.clicked.connect(lambda: self._set_provider_cards_expanded(False))
        controls.addWidget(expand_button)
        controls.addWidget(collapse_button)
        controls.addStretch(1)
        wrapper_layout.addLayout(controls)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        for provider in self.providers:
            card = ProviderConfigCard(provider, expanded=provider.enabled)
            self.provider_cards.append(card)
            content_layout.addWidget(card)

        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        wrapper_layout.addWidget(scroll, 1)

        tips = QLabel("Add a specialty note for each expert. The lead can assign subproblems by specialty; the literature reviewer is optional and can read references before the expert team starts.")
        tips.setWordWrap(True)
        tips.setStyleSheet("font-size: 11px; font-family: 'Microsoft YaHei UI'; color: #88A6C7;")
        wrapper_layout.addWidget(tips)

        save_button = QPushButton("Save config")
        save_button.clicked.connect(lambda: self._save_provider_config(show_feedback=True))
        wrapper_layout.addWidget(save_button)
        return group

    def _build_attachment_group(self) -> QWidget:
        group = QGroupBox("Attachments")
        layout = QVBoxLayout(group)

        self.attachment_list = QListWidget()
        layout.addWidget(self.attachment_list)

        self.literature_review_check = QCheckBox("Enable literature review")
        self.literature_review_check.setChecked(False)
        self.literature_review_check.setToolTip("If enabled and a literature reviewer is configured, the system will generate a literature review from the attached references before the main team proceeds.")
        layout.addWidget(self.literature_review_check)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add files")
        remove_button = QPushButton("Remove selected")
        add_button.clicked.connect(self._add_attachments)
        remove_button.clicked.connect(self._remove_selected_attachment)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        layout.addLayout(button_row)

        self.pdf_reader_button = QPushButton("Build PDF reader")
        self.pdf_reader_button.setObjectName("ghostAction")
        self.pdf_reader_button.setToolTip("Build or refresh the PDF reader cache. Discussion will retrieve indexed sections, figures, and formulas from this cache.")
        self.pdf_reader_button.clicked.connect(self._build_pdf_reader_cache)
        layout.addWidget(self.pdf_reader_button)
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

        header_strip = BrandHeaderFrame()
        header_strip.setObjectName("headerStrip")
        header_strip_layout = QVBoxLayout(header_strip)
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
        title = QLabel("Cyber Colloquium")
        title.setObjectName("heroTitle")
        apply_outer_glow(title, QColor(116, 238, 255), blur_radius=24, alpha=118)
        subtitle = QLabel("--Create your own AI-powered academic meeting")
        subtitle.setObjectName("heroSubtitle")
        apply_outer_glow(subtitle, QColor(82, 204, 255), blur_radius=14, alpha=72)
        text_stack.addWidget(title)
        text_stack.addWidget(subtitle)
        brand_row.addLayout(text_stack, 1)
        header_strip_layout.addLayout(brand_row)
        top_layout.addWidget(header_strip)

        composer_card = QFrame()
        composer_card.setObjectName("composerCard")
        composer_layout = QHBoxLayout(composer_card)
        composer_layout.setContentsMargins(18, 18, 18, 18)
        composer_layout.setSpacing(14)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setObjectName("promptEdit")
        self.prompt_edit.setPlaceholderText("e.g. Explore a problem in quantitative finance, image processing, physics, or another field. Review the references first, then decompose the problem and collaborate on the analysis.")
        self.prompt_edit.setFixedHeight(100)
        composer_layout.addWidget(self.prompt_edit, 1)

        action_column = QVBoxLayout()
        action_column.setSpacing(10)
        flow_label = QLabel("Execution Mode")
        flow_label.setStyleSheet("font-size: 12px; font-weight: 700; font-family: 'Microsoft YaHei UI'; color: #A8D9F7;")
        flow_value = QLabel("Automatic workpackage flow")
        flow_value.setStyleSheet(
            "background: rgba(11, 46, 87, 0.86); color: #7FE2FF; border: 1px solid rgba(83, 216, 255, 0.28); border-radius: 14px; padding: 10px 12px; font-size: 12px; font-weight: 700; font-family: 'Microsoft YaHei UI';"
        )
        self.start_button = GlowButton("Start discussion", QColor(102, 231, 255))
        self.start_button.setObjectName("primaryAction")
        self.start_button.clicked.connect(self._start_discussion)
        self.stop_button = QPushButton("Standby")
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

        thread_frame = ChatSurfaceFrame()
        thread_frame.setObjectName("chatSurface")
        thread_layout = QVBoxLayout(thread_frame)
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
        layout.addWidget(thread_frame, 1)

        self._append_status_card("System", "Import materials and start the discussion. Optionally let the literature reviewer digest the references first, then let the lead assign work by specialty and have the expert team cross-check to reduce hallucinations.")
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
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
                font-family: 'Microsoft YaHei UI';
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 6px;
                color: #D7F7FF;
            }
            QFrame#providerCard {
                background: rgba(10, 24, 48, 0.92);
                border: 1px solid rgba(83, 216, 255, 0.16);
                border-radius: 18px;
            }
            QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QComboBox {
                background: rgba(4, 12, 26, 0.96);
                color: #EFFBFF;
                border: 1px solid rgba(110, 176, 255, 0.26);
                border-radius: 14px;
                padding: 9px 10px;
                selection-background-color: rgba(65, 175, 255, 0.82);
                selection-color: #04101F;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QListWidget:focus, QComboBox:focus {
                border: 1px solid rgba(111, 235, 255, 0.62);
            }
            QLineEdit#roleNameEdit {
                font-size: 13px;
                font-weight: 700;
                font-family: 'Microsoft YaHei UI';
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0F2A55, stop:1 #13407F);
                color: #F1FCFF;
                border: 1px solid rgba(91, 198, 255, 0.22);
                border-radius: 14px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 700;
                font-family: 'Microsoft YaHei UI';
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
                font-weight: 700;
                font-family: 'Microsoft YaHei UI';
            }
            QPushButton#ghostAction:hover {
                background: rgba(17, 44, 86, 0.92);
                border: 1px solid rgba(105, 233, 255, 0.32);
            }
            QToolButton#collapseToggle {
                background: transparent;
                color: #91D8FF;
                border: none;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 700;
                font-family: 'Microsoft YaHei UI';
            }
            QToolButton#collapseToggle:hover {
                color: #D7F8FF;
            }
            QCheckBox {
                spacing: 6px;
            }
            QFrame#topBar {
                background: transparent;
                border: none;
            }
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
            QFrame#headerStrip {
                background: transparent;
                border: none;
                border-radius: 30px;
            }
            QLabel#brandLogo {
                background: rgba(5, 18, 36, 0.92);
                border: 1px solid rgba(103, 232, 249, 0.32);
                border-radius: 20px;
                padding: 2px;
            }
            QLabel#heroTitle {
                color: #F1FCFF;
                font-size: 36px;
                font-weight: 800;
                font-family: 'Microsoft YaHei UI';
            }
            QLabel#heroSubtitle {
                color: #64E2FF;
                font-size: 15px;
                font-weight: 700;
                font-family: 'Microsoft YaHei UI';
            }
            QFrame#composerCard {
                background: rgba(7, 18, 37, 0.88);
                border: 1px solid rgba(83, 216, 255, 0.14);
                border-radius: 26px;
            }
            QFrame#chatSurface {
                background: transparent;
                border: none;
                border-radius: 28px;
            }
            QWidget#chatCanvas {
                background: transparent;
            }
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
            QPushButton#primaryAction[state="running"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #11487C, stop:1 #1F78B4);
                color: #E6FBFF;
            }
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
            QPushButton#secondaryAction:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #C53850, stop:1 #FF6A88);
                color: white;
                border: 1px solid rgba(255, 170, 196, 0.5);
            }
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
            QScrollBar::handle:vertical:hover {
                background: rgba(109, 228, 255, 0.72);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0px;
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #7CE7FF;
                margin-right: 8px;
            }
            QListView, QAbstractItemView {
                background: rgba(6, 17, 33, 0.98);
                color: #EFFBFF;
                border: 1px solid rgba(92, 181, 255, 0.24);
                selection-background-color: rgba(52, 157, 255, 0.55);
                selection-color: #F6FDFF;
            }
            QSplitter::handle {
                background: transparent;
            }
            """
        )

    def _set_discussion_state(self, state: str) -> None:
        if state == "idle":
            self.start_button.setEnabled(True)
            self.start_button.setText("Start discussion")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Standby")
            self.start_button.setProperty("state", "idle")
            self.stop_button.setProperty("state", "idle")
        elif state == "running":
            self.start_button.setEnabled(False)
            self.start_button.setText("Discussion running")
            self.stop_button.setEnabled(True)
            self.stop_button.setText("Stop discussion")
            self.start_button.setProperty("state", "running")
            self.stop_button.setProperty("state", "running")
        elif state == "stopping":
            self.start_button.setEnabled(False)
            self.start_button.setText("Discussion running")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping")
            self.start_button.setProperty("state", "running")
            self.stop_button.setProperty("state", "stopping")

        for button in (self.start_button, self.stop_button):
            self.style().unpolish(button)
            self.style().polish(button)
            button.update()

    def _set_provider_cards_expanded(self, expanded: bool) -> None:
        for card in self.provider_cards:
            card.set_expanded(expanded)

    def _save_provider_config(self, show_feedback: bool = False) -> None:
        for card in self.provider_cards:
            card.apply_to_provider()
        save_providers(self.providers)
        if show_feedback:
            QMessageBox.information(self, "Config Saved", "Provider settings were written to app_config.json")

    def _refresh_attachment_list(self) -> None:
        self.attachment_list.clear()
        for attachment in self.attachments:
            label = f"{attachment.display_name} [{attachment.kind}]"
            if attachment.kind == "pdf":
                badge = pdf_reader_badge(attachment.path)
                if badge:
                    label += f" [{badge}]"
            self.attachment_list.addItem(QListWidgetItem(label))

    def _add_attachments(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select attachments",
            "",
            "Supported Files (*.pdf *.txt *.md *.png *.jpg *.jpeg *.webp *.bmp *.json *.csv)",
        )
        for path in paths:
            try:
                payload = load_attachment(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Failed to load attachment", f"{Path(path).name}: {exc}")
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
            QMessageBox.warning(self, "No PDF attached", "Attach at least one PDF before building the PDF reader cache.")
            return

        provider = self._select_pdf_reader_provider()
        if provider is None:
            self._append_status_card(
                "System",
                "No literature-review provider with an API key was found. Building local PDF section indexes in index-only mode.",
            )
        else:
            self._append_status_card(
                "System",
                f"Building PDF reader digests with {provider.name} for {len(pdf_attachments)} PDF(s).",
            )

        if self.pdf_reader_button is not None:
            self.pdf_reader_button.setEnabled(False)
            self.pdf_reader_button.setText("Building PDF reader")

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
            QMessageBox.warning(self, "Missing prompt", "Enter a research question or discussion goal first.")
            return

        self._save_provider_config(show_feedback=False)
        active_providers = [provider for provider in self.providers if provider.enabled and provider.api_key]
        if not active_providers:
            QMessageBox.warning(self, "No active model", "Enable and configure at least one model with an API key.")
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
            speaker="You",
            duty="USER",
            body=prompt,
            meta=f"User prompt | {len(self.attachments)} attachment(s)",
            color=color_for_speaker("USER"),
            align_right=True,
        )
        self._append_status_card("System", "Discussion started. The lead will delegate by specialty first, then the host will arrange execution.")
        if self.literature_review_check.isChecked():
            self._append_status_card("System", "Literature review is enabled. If a literature reviewer and reference attachments are available, the system will generate a review before the expert team proceeds.")
        self._append_status_card("System", "The expert team will work on specific subproblems, cross-review each other, and the reporter will update the log in real time before generating the output files.")

        self.worker_thread = QThread()
        self.worker = DiscussionWorker(
            active_providers,
            prompt,
            self.attachments,
            self.literature_review_check.isChecked(),
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
        self._append_status_card("System", "Stop requested. The current model call will finish first, then the remaining workflow will stop.")

    def _on_discussion_message(self, message: DiscussionMessage) -> None:
        self.session_messages.append(message)
        if message.round_index > 0 and message.round_index != self.last_displayed_round:
            self.last_displayed_round = message.round_index
            self._append_task_divider(message.round_index)

        if message.duty == LEAD_DUTY:
            meta = "Lead delegation"
        elif message.duty == HOST_DUTY:
            meta = "Host coordination"
        elif message.duty == LITERATURE_DUTY:
            meta = "Literature review"
        elif message.duty == REPORT_DUTY:
            meta = "Live log" if message.stage == "log" else "Report delivery"
        elif message.stage == "review":
            meta = "Expert review"
        elif message.duty == EXPERT_DUTY:
            meta = "Expert execution"
        else:
            meta = message.stage or "Discussion message"
        meta = f"{meta} | {message.model_name}"
        self._append_message(
            speaker=message.speaker,
            duty=message.duty,
            body=message.content,
            meta=meta,
            color=color_for_speaker(message.speaker),
            align_right=False,
            emphasis=message.content.startswith("[Call Failed]"),
        )

    def _on_discussion_status(self, text: str) -> None:
        match = re.search(r"Task\s*(\d+)", text)
        if match:
            self._append_task_divider(int(match.group(1)))
            self.last_displayed_round = int(match.group(1))
        self._append_status_card("System", text)

    def _on_discussion_finished(self, result: DiscussionResult) -> None:
        try:
            literature_path, minutes_path, report_path = save_discussion_outputs(
                user_request=self.current_prompt,
                providers=self.providers,
                messages=result.messages,
                literature_review_text=result.literature_review or "",
                summary_text=result.final_summary or "No research report was generated.",
                minutes_text=result.meeting_minutes or result.final_summary or "No meeting minutes were generated.",
                cancelled=result.cancelled,
                meeting_state=result.meeting_state,
            )
            result.literature_review_path = str(literature_path) if literature_path is not None else ""
            result.meeting_minutes_path = str(minutes_path)
            result.summary_path = str(report_path)
            result.report_path = str(report_path)
            saved_lines = []
            if literature_path is not None:
                saved_lines.append(f"- Literature review: `{literature_path}`")
            saved_lines.append(f"- Meeting minutes: `{minutes_path}`")
            saved_lines.append(f"- Research report: `{report_path}`")
            self._append_status_card(
                "System",
                "Saved local files:\n" + "\n".join(saved_lines),
            )
        except Exception as exc:  # noqa: BLE001
            self._append_status_card("System", f"Failed to save local files: {exc}")
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

        self._append_status_card("System", f"Discussion failed:\n\n{error_text}")
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
            )
            self._append_status_card("System", f"Saved failure snapshot: `{failure_path}`")
        except Exception as exc:  # noqa: BLE001
            self._append_status_card("System", f"Failed to save failure snapshot: {exc}")
        self._set_discussion_state("idle")
        self._scroll_chat_to_bottom()

    def _on_pdf_reader_status(self, text: str) -> None:
        self._append_status_card("System", text)

    def _on_pdf_reader_finished(self, payload: object) -> None:
        results = [item for item in (payload or []) if isinstance(item, PdfReaderBuildResult)]
        self._refresh_attachment_list()
        if results:
            saved_lines = [
                f"- {Path(item.source_pdf).name} | digest: `{item.digest_markdown_path}` | index: `{item.index_path}`"
                for item in results
            ]
            self._append_status_card("System", "PDF reader artifacts saved:\n" + "\n".join(saved_lines))
        else:
            self._append_status_card("System", "PDF reader build finished, but no PDF artifact was produced.")

    def _on_pdf_reader_failed(self, error_text: str) -> None:
        self._append_status_card("System", f"PDF reader build failed:\n\n{error_text}")
    def _cleanup_reader_worker(self) -> None:
        self.reader_worker = None
        self.reader_thread = None
        if self.pdf_reader_button is not None:
            self.pdf_reader_button.setEnabled(True)
            self.pdf_reader_button.setText("Build PDF reader")

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
        )
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._animate_message_entry(bubble)
        self._scroll_chat_to_bottom()

    def _append_status_card(self, speaker: str, body: str) -> None:
        self.session_status_lines.append(f"{speaker}: {body}")
        self._append_message(
            speaker=speaker,
            duty=HOST_DUTY,
            body=body,
            meta="System note",
            color=color_for_speaker("HOST"),
            align_right=False,
        )

    def _extract_session_literature_review(self, messages: list[DiscussionMessage]) -> str:
        for message in reversed(messages):
            if message.duty == LITERATURE_DUTY or message.stage == "literature_review":
                return message.content
        return ""

    def _append_time_divider(self) -> None:
        divider = TimelineDivider(datetime.now().strftime("%Y-%m-%d %H:%M"), accent=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, divider)
        self._scroll_chat_to_bottom()

    def _append_task_divider(self, round_index: int) -> None:
        divider = TimelineDivider(f"Task {round_index}", accent=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, divider)
        self._scroll_chat_to_bottom()

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
            effect.deleteLater()
            if animation in self.entry_animations:
                self.entry_animations.remove(animation)

        animation.finished.connect(_cleanup)
        animation.start()
