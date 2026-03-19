from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import DUTY_OPTIONS, EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, LITERATURE_DUTY, REPORT_DUTY
from .roles import MODERATOR_ROLE_KEY, NOTETAKER_ROLE_KEY, RESEARCH_ANALYST_ROLE_KEY, REVIEWER_ROLE_KEY


WORKFLOW_CONFIG_PATH = Path("workflow_config.json")
WORKFLOW_CONFIG_VERSION = 7
WORKFLOW_INSERTED_STAGE_KEYS = {"discover_literature", "expand_literature_search", "run_experiment_cycle", "compile_latex_artifacts"}

DEFAULT_ROLE_ORDER = [LEAD_DUTY, HOST_DUTY, LITERATURE_DUTY, EXPERT_DUTY, REPORT_DUTY]
DEFAULT_SUMMARY_SLOTS = ["consensus", "conflicts", "open_questions", "recent_updates", "action_items"]
SUMMARY_SLOT_OPTIONS = set(DEFAULT_SUMMARY_SLOTS)
DEFAULT_TEAM_TEMPLATE_NAME = "Cyber Colloquium Basic Team"
DEFAULT_WORKFLOW_TEMPLATE_NAME = "Research Discussion Review Workflow"

DEFAULT_DUTY_ROLE_KEYS = {
    LEAD_DUTY: MODERATOR_ROLE_KEY,
    HOST_DUTY: MODERATOR_ROLE_KEY,
    EXPERT_DUTY: RESEARCH_ANALYST_ROLE_KEY,
    LITERATURE_DUTY: REVIEWER_ROLE_KEY,
    REPORT_DUTY: NOTETAKER_ROLE_KEY,
}


@dataclass(frozen=True)
class DiscussionPolicyConfig:
    max_rounds: int = 8
    checkpoint_every_n_rounds: int = 1
    enable_reviewer_role: bool = True
    max_checkpoints: int = 8
    max_followup_items: int = 2
    max_followup_attempts: int = 1
    max_literature_review_batches: int = 3


@dataclass(frozen=True)
class RoutingConfig:
    default_roles: list[str] = field(default_factory=lambda: list(DEFAULT_ROLE_ORDER))
    parallel_roles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextConfig:
    max_history_items: int = 12
    summary_slots: list[str] = field(default_factory=lambda: list(DEFAULT_SUMMARY_SLOTS))
    max_evidence_cards: int = 24
    max_log_entries: int = 40


@dataclass(frozen=True)
class ReportOptionsConfig:
    include_consensus: bool = True
    include_open_questions: bool = True
    include_action_items: bool = True


@dataclass(frozen=True)
class NotesOptionsConfig:
    include_role_labels: bool = True


@dataclass(frozen=True)
class ToolingConfig:
    enable_arxiv_discovery: bool = False
    download_arxiv_pdfs: bool = True
    arxiv_max_results: int = 3
    enable_python_artifact: bool = False
    enable_latex_artifact: bool = False
    enable_bibtex_artifact: bool = False
    enable_python_execution_test: bool = False
    enable_python_full_execution: bool = False
    python_execution_timeout_seconds: int = 20
    python_full_execution_timeout_seconds: int = 300
    python_workspace_input_limit_mb: int = 64
    enable_latex_compile: bool = False


@dataclass(frozen=True)
class TeamRoleConfig:
    duty: str
    role_key: str
    label: str
    enabled: bool = True
    required: bool = False
    specialty_hint: str = ""


@dataclass(frozen=True)
class TeamTemplateConfig:
    name: str = DEFAULT_TEAM_TEMPLATE_NAME
    description: str = ""
    roles: list[TeamRoleConfig] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowStageConfig:
    key: str
    label: str
    enabled: bool = True
    description: str = ""


@dataclass(frozen=True)
class WorkflowTemplateConfig:
    name: str = DEFAULT_WORKFLOW_TEMPLATE_NAME
    description: str = ""
    stages: list[WorkflowStageConfig] = field(default_factory=list)

    def enabled_stages(self) -> list[WorkflowStageConfig]:
        return [stage for stage in self.stages if stage.enabled]


@dataclass(frozen=True)
class WorkflowConfig:
    discussion: DiscussionPolicyConfig = field(default_factory=DiscussionPolicyConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    report: ReportOptionsConfig = field(default_factory=ReportOptionsConfig)
    notes: NotesOptionsConfig = field(default_factory=NotesOptionsConfig)
    tooling: ToolingConfig = field(default_factory=ToolingConfig)
    team_template: TeamTemplateConfig = field(default_factory=lambda: default_team_template())
    workflow_template: WorkflowTemplateConfig = field(default_factory=lambda: default_workflow_template())

    @property
    def team_roles(self) -> list[TeamRoleConfig]:
        return list(self.team_template.roles)

    def role_config_map(self) -> dict[str, TeamRoleConfig]:
        return {role.duty: role for role in self.team_roles}

    def role_config(self, duty: str) -> TeamRoleConfig | None:
        return self.role_config_map().get(duty)

    def is_role_enabled(self, duty: str) -> bool:
        role = self.role_config(duty)
        return role.enabled if role is not None else True

    def ordered_role_duties(self) -> list[str]:
        available = list(self.role_config_map())
        ordered = [duty for duty in self.routing.default_roles if duty in available]
        ordered.extend(duty for duty in available if duty not in ordered)
        return ordered


def default_team_roles() -> list[TeamRoleConfig]:
    return [
        TeamRoleConfig(
            duty=LEAD_DUTY,
            role_key=DEFAULT_DUTY_ROLE_KEYS[LEAD_DUTY],
            label="Lead",
            enabled=True,
            required=False,
            specialty_hint="Task decomposition, delegation, and quality control",
        ),
        TeamRoleConfig(
            duty=HOST_DUTY,
            role_key=DEFAULT_DUTY_ROLE_KEYS[HOST_DUTY],
            label="Moderator",
            enabled=True,
            required=False,
            specialty_hint="Coordination, pacing, and discussion control",
        ),
        TeamRoleConfig(
            duty=EXPERT_DUTY,
            role_key=DEFAULT_DUTY_ROLE_KEYS[EXPERT_DUTY],
            label="Research Analyst",
            enabled=True,
            required=True,
            specialty_hint="Analysis, evidence checking, and execution",
        ),
        TeamRoleConfig(
            duty=LITERATURE_DUTY,
            role_key=DEFAULT_DUTY_ROLE_KEYS[LITERATURE_DUTY],
            label="Reviewer",
            enabled=True,
            required=False,
            specialty_hint="Literature review, related-work mapping, and source digestion",
        ),
        TeamRoleConfig(
            duty=REPORT_DUTY,
            role_key=DEFAULT_DUTY_ROLE_KEYS[REPORT_DUTY],
            label="Notetaker",
            enabled=True,
            required=False,
            specialty_hint="Structured logging, synthesis, and final document writing",
        ),
    ]


def default_team_template() -> TeamTemplateConfig:
    return TeamTemplateConfig(
        name=DEFAULT_TEAM_TEMPLATE_NAME,
        description="Default multi-role research team that preserves the current Cyber Colloquium workflow.",
        roles=default_team_roles(),
    )


def default_workflow_template() -> WorkflowTemplateConfig:
    return WorkflowTemplateConfig(
        name=DEFAULT_WORKFLOW_TEMPLATE_NAME,
        description="Default configurable research workflow: ingest sources, discuss, review, update structured state, then generate notes and report.",
        stages=[
            WorkflowStageConfig(
                key="discover_literature",
                label="Discover arXiv Literature",
                enabled=True,
                description="Optionally search arXiv from the user request, download selected papers, and register metadata before ingesting sources.",
            ),
            WorkflowStageConfig(
                key="ingest_source_material",
                label="Ingest Source Material",
                enabled=True,
                description="Prepare user question, source material, PDF reader context, and optional literature review input.",
            ),
            WorkflowStageConfig(
                key="run_team_discussion",
                label="Run Team Discussion",
                enabled=True,
                description="Run lead delegation, host coordination, and primary analyst execution.",
            ),
            WorkflowStageConfig(
                key="expand_literature_search",
                label="Expand Literature Search",
                enabled=True,
                description="Let the lead or experts propose targeted arXiv queries from discussion gaps, then refresh the evidence pool before reviewer pass.",
            ),
            WorkflowStageConfig(
                key="run_reviewer_pass",
                label="Run Reviewer Pass",
                enabled=True,
                description="Run reviewer cross-checks for each workpackage when reviewer participation is enabled.",
            ),
            WorkflowStageConfig(
                key="update_structured_state",
                label="Update Structured State",
                enabled=True,
                description="Run state consolidation, checkpoints, and unresolved-issue follow-ups.",
            ),
            WorkflowStageConfig(
                key="run_experiment_cycle",
                label="Run Experiment Cycle",
                enabled=True,
                description="Generate executable experiment scaffolds, run authorized local tests, and feed outcomes back into the structured state.",
            ),
            WorkflowStageConfig(
                key="generate_meeting_notes",
                label="Generate Meeting Notes",
                enabled=True,
                description="Generate meeting notes from the structured intermediate state.",
            ),
            WorkflowStageConfig(
                key="generate_research_report",
                label="Generate Research Report",
                enabled=True,
                description="Generate the final research report from the structured intermediate state.",
            ),
            WorkflowStageConfig(
                key="compile_latex_artifacts",
                label="Compile LaTeX Artifacts",
                enabled=True,
                description="Optionally build BibTeX and LaTeX export artifacts and compile them with local Tectonic when authorized.",
            ),
        ],
    )


def default_workflow_config() -> WorkflowConfig:
    return WorkflowConfig(team_template=default_team_template(), workflow_template=default_workflow_template())


def load_workflow_config(path: Path = WORKFLOW_CONFIG_PATH) -> WorkflowConfig:
    if not path.exists():
        config = default_workflow_config()
        save_workflow_config(config, path)
        return config

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        config = default_workflow_config()
        save_workflow_config(config, path)
        return config

    config = workflow_config_from_dict(raw)
    save_workflow_config(config, path)
    return config


def save_workflow_config(config: WorkflowConfig, path: Path = WORKFLOW_CONFIG_PATH) -> None:
    payload = {
        "version": WORKFLOW_CONFIG_VERSION,
        "discussion": asdict(config.discussion),
        "routing": asdict(config.routing),
        "context": asdict(config.context),
        "report": asdict(config.report),
        "notes": asdict(config.notes),
        "tooling": asdict(config.tooling),
        "team_template": asdict(config.team_template),
        "workflow_template": asdict(config.workflow_template),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(path)


def workflow_config_from_dict(raw: object) -> WorkflowConfig:
    data = raw if isinstance(raw, dict) else {}

    discussion_data = _coerce_mapping(data.get("discussion"))
    routing_data = _coerce_mapping(data.get("routing"))
    context_data = _coerce_mapping(data.get("context"))
    report_data = _coerce_mapping(data.get("report"))
    notes_data = _coerce_mapping(data.get("notes"))
    tooling_data = _coerce_mapping(data.get("tooling"))
    template_data = _coerce_mapping(data.get("team_template"))
    workflow_template_data = _coerce_mapping(data.get("workflow_template"))
    merged_roles = _merge_team_roles(template_data.get("roles", data.get("team_roles")))
    team_template = TeamTemplateConfig(
        name=str(template_data.get("name") or DEFAULT_TEAM_TEMPLATE_NAME).strip() or DEFAULT_TEAM_TEMPLATE_NAME,
        description=str(template_data.get("description") or "Default multi-role research team.").strip(),
        roles=merged_roles,
    )
    workflow_template = WorkflowTemplateConfig(
        name=str(workflow_template_data.get("name") or DEFAULT_WORKFLOW_TEMPLATE_NAME).strip() or DEFAULT_WORKFLOW_TEMPLATE_NAME,
        description=str(workflow_template_data.get("description") or "Default configurable research workflow.").strip(),
        stages=_merge_workflow_stages(workflow_template_data.get("stages")),
    )

    config = WorkflowConfig(
        discussion=DiscussionPolicyConfig(
            max_rounds=_coerce_int(discussion_data.get("max_rounds"), 8, minimum=1),
            checkpoint_every_n_rounds=_coerce_int(discussion_data.get("checkpoint_every_n_rounds"), 1, minimum=1),
            enable_reviewer_role=_coerce_bool(discussion_data.get("enable_reviewer_role"), True),
            max_checkpoints=_coerce_int(discussion_data.get("max_checkpoints"), 8, minimum=1),
            max_followup_items=_coerce_int(discussion_data.get("max_followup_items"), 2, minimum=1),
            max_followup_attempts=_coerce_int(discussion_data.get("max_followup_attempts"), 1, minimum=1),
            max_literature_review_batches=_coerce_int(discussion_data.get("max_literature_review_batches"), 3, minimum=1),
        ),
        routing=RoutingConfig(
            default_roles=_coerce_duty_list(routing_data.get("default_roles"), DEFAULT_ROLE_ORDER),
            parallel_roles=_coerce_duty_list(routing_data.get("parallel_roles"), []),
        ),
        context=ContextConfig(
            max_history_items=_coerce_int(context_data.get("max_history_items"), 12, minimum=1),
            summary_slots=_coerce_summary_slots(context_data.get("summary_slots"), DEFAULT_SUMMARY_SLOTS),
            max_evidence_cards=_coerce_int(context_data.get("max_evidence_cards"), 24, minimum=1),
            max_log_entries=_coerce_int(context_data.get("max_log_entries"), 40, minimum=1),
        ),
        report=ReportOptionsConfig(
            include_consensus=_coerce_bool(report_data.get("include_consensus"), True),
            include_open_questions=_coerce_bool(report_data.get("include_open_questions"), True),
            include_action_items=_coerce_bool(report_data.get("include_action_items"), True),
        ),
        notes=NotesOptionsConfig(
            include_role_labels=_coerce_bool(notes_data.get("include_role_labels"), True),
        ),
        tooling=ToolingConfig(
            enable_arxiv_discovery=_coerce_bool(tooling_data.get("enable_arxiv_discovery"), False),
            download_arxiv_pdfs=_coerce_bool(tooling_data.get("download_arxiv_pdfs"), True),
            arxiv_max_results=_coerce_int(tooling_data.get("arxiv_max_results"), 3, minimum=1),
            enable_python_artifact=_coerce_bool(tooling_data.get("enable_python_artifact"), False),
            enable_latex_artifact=_coerce_bool(tooling_data.get("enable_latex_artifact"), False),
            enable_bibtex_artifact=_coerce_bool(tooling_data.get("enable_bibtex_artifact"), False),
            enable_python_execution_test=_coerce_bool(tooling_data.get("enable_python_execution_test"), False),
            enable_python_full_execution=_coerce_bool(tooling_data.get("enable_python_full_execution"), False),
            python_execution_timeout_seconds=_coerce_int(tooling_data.get("python_execution_timeout_seconds"), 20, minimum=5),
            python_full_execution_timeout_seconds=_coerce_int(tooling_data.get("python_full_execution_timeout_seconds"), 300, minimum=10),
            python_workspace_input_limit_mb=_coerce_int(tooling_data.get("python_workspace_input_limit_mb"), 64, minimum=1),
            enable_latex_compile=_coerce_bool(tooling_data.get("enable_latex_compile"), False),
        ),
        team_template=team_template,
        workflow_template=workflow_template,
    )
    return config


def _merge_team_roles(raw_roles: object) -> list[TeamRoleConfig]:
    roles_by_duty = {role.duty: role for role in default_team_roles()}
    if isinstance(raw_roles, list):
        for item in raw_roles:
            if not isinstance(item, dict):
                continue
            duty = str(item.get("duty", "")).strip()
            if duty not in DUTY_OPTIONS:
                continue
            default_role = roles_by_duty[duty]
            roles_by_duty[duty] = TeamRoleConfig(
                duty=duty,
                role_key=str(item.get("role_key") or default_role.role_key).strip() or default_role.role_key,
                label=str(item.get("label") or default_role.label).strip() or default_role.label,
                enabled=_coerce_bool(item.get("enabled"), default_role.enabled),
                required=_coerce_bool(item.get("required"), default_role.required),
                specialty_hint=str(item.get("specialty_hint") or default_role.specialty_hint).strip(),
            )

    ordered: list[TeamRoleConfig] = []
    for duty in DEFAULT_ROLE_ORDER:
        role = roles_by_duty.get(duty)
        if role is not None:
            ordered.append(role)
    for duty, role in roles_by_duty.items():
        if duty not in DEFAULT_ROLE_ORDER:
            ordered.append(role)
    return ordered


def _merge_workflow_stages(raw_stages: object) -> list[WorkflowStageConfig]:
    default_stages = default_workflow_template().stages
    default_order = [stage.key for stage in default_stages]
    stages_by_key = {stage.key: stage for stage in default_stages}
    if isinstance(raw_stages, list):
        ordered_keys: list[str] = []
        for item in raw_stages:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if key not in stages_by_key:
                continue
            default_stage = stages_by_key[key]
            stages_by_key[key] = WorkflowStageConfig(
                key=key,
                label=str(item.get("label") or default_stage.label).strip() or default_stage.label,
                enabled=_coerce_bool(item.get("enabled"), default_stage.enabled),
                description=str(item.get("description") or default_stage.description).strip(),
            )
            if key not in ordered_keys:
                ordered_keys.append(key)

        legacy_filtered_order = [key for key in ordered_keys if key not in WORKFLOW_INSERTED_STAGE_KEYS]
        if _is_subsequence(ordered_keys, default_order) or _is_subsequence(legacy_filtered_order, default_order):
            ordered: list[WorkflowStageConfig] = [stages_by_key[key] for key in default_order]
        else:
            merged_keys = list(ordered_keys)
            for stage in default_stages:
                if stage.key in merged_keys:
                    continue
                insert_at = len(merged_keys)
                default_index = default_order.index(stage.key)
                for later_key in default_order[default_index + 1 :]:
                    if later_key in merged_keys:
                        insert_at = merged_keys.index(later_key)
                        break
                merged_keys.insert(insert_at, stage.key)
            ordered = [stages_by_key[key] for key in merged_keys]
        return ordered
    return list(default_stages)


def _coerce_mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _coerce_int(value: object, default: int, *, minimum: int) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    if number < minimum:
        return default
    return number


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_duty_list(value: object, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    duties: list[str] = []
    for item in value:
        duty = str(item).strip()
        if duty in DUTY_OPTIONS and duty not in duties:
            duties.append(duty)
    return duties or list(default)


def _coerce_summary_slots(value: object, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    slots: list[str] = []
    for item in value:
        slot = str(item).strip()
        if slot in SUMMARY_SLOT_OPTIONS and slot not in slots:
            slots.append(slot)
    return slots or list(default)


def _is_subsequence(candidate: list[str], target: list[str]) -> bool:
    if not candidate:
        return True
    target_iter = iter(target)
    return all(any(item == candidate_item for item in target_iter) for candidate_item in candidate)
