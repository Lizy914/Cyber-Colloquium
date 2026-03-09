from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LEAD_DUTY = "Lead"
HOST_DUTY = "Host"
EXPERT_DUTY = "Expert"
LITERATURE_DUTY = "Literature Reviewer"
REPORT_DUTY = "Reporter"
DUTY_OPTIONS = [LEAD_DUTY, HOST_DUTY, EXPERT_DUTY, LITERATURE_DUTY, REPORT_DUTY]


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_key: str = ""
    enabled: bool = True
    supports_vision: bool = False
    temperature: float = 0.6
    duty: str = EXPERT_DUTY
    specialty: str = ""


@dataclass
class AttachmentPayload:
    path: Path
    kind: str
    content: str
    display_name: str


@dataclass
class AttachmentSnippet:
    evidence_id: str
    attachment_name: str
    kind: str
    chunk_index: int
    content: str
    page_hint: int | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class ReaderReference:
    ref_id: str
    attachment_name: str
    kind: str
    title: str
    content: str
    page_hint: int | None = None
    image_path: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class EvidenceCard:
    evidence_id: str
    summary: str
    source: str
    display_label: str = ""
    attachment_name: str = ""
    workpackage_index: int = 0


@dataclass
class StructuredLogEntry:
    workpackage_index: int
    workpackage_title: str
    speaker: str
    stage: str
    headline: str
    summary: str
    consensus_add: list[str] = field(default_factory=list)
    conflicts_add: list[str] = field(default_factory=list)
    resolved_conflicts: list[str] = field(default_factory=list)
    open_questions_add: list[str] = field(default_factory=list)
    resolved_questions: list[str] = field(default_factory=list)
    rejected_add: list[str] = field(default_factory=list)
    action_items_add: list[str] = field(default_factory=list)
    evidence_add: list[EvidenceCard] = field(default_factory=list)
    redundant: bool = False


@dataclass
class MeetingCheckpoint:
    checkpoint_id: str
    label: str
    workpackage_index: int
    summary: str
    consensus: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


@dataclass
class MeetingState:
    topic: str = ""
    domain: str = ""
    goal: str = ""
    assignment_summary: str = ""
    coordination_summary: str = ""
    current_stage: str = ""
    current_question: str = ""
    rules: list[str] = field(default_factory=list)
    stable_consensus: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    rejected_lines: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    log_entries: list[StructuredLogEntry] = field(default_factory=list)
    checkpoints: list[MeetingCheckpoint] = field(default_factory=list)


@dataclass
class DiscussionMessage:
    speaker: str
    role: str
    content: str
    round_index: int = 0
    model_name: str = ""
    duty: str = ""
    stage: str = ""


@dataclass
class DiscussionResult:
    messages: list[DiscussionMessage] = field(default_factory=list)
    literature_review: str = ""
    final_summary: str = ""
    meeting_minutes: str = ""
    literature_review_path: str = ""
    summary_path: str = ""
    report_path: str = ""
    meeting_minutes_path: str = ""
    cancelled: bool = False
    meeting_state: MeetingState | None = None
