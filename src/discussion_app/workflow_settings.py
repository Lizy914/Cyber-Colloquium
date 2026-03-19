from __future__ import annotations

from dataclasses import dataclass, replace

from .workflow_config import DEFAULT_SUMMARY_SLOTS, SUMMARY_SLOT_OPTIONS, WorkflowConfig
from .workflow_graph import build_workflow_graph, render_workflow_graph_summary


SUMMARY_SLOT_LABELS = {
    "consensus": "共识",
    "conflicts": "争议",
    "open_questions": "未决问题",
    "recent_updates": "近期更新",
    "action_items": "行动项",
}


@dataclass(frozen=True)
class WorkflowSettingsState:
    max_rounds: int
    checkpoint_every_n_rounds: int
    reviewer_enabled: bool
    enabled_roles: dict[str, bool]
    summary_slots: list[str]
    arxiv_discovery_enabled: bool
    arxiv_download_enabled: bool
    arxiv_max_results: int
    python_artifact_enabled: bool
    latex_artifact_enabled: bool
    bibtex_artifact_enabled: bool
    python_execution_test_enabled: bool
    python_full_execution_enabled: bool
    python_execution_timeout_seconds: int
    python_full_execution_timeout_seconds: int
    python_workspace_input_limit_mb: int
    latex_compile_enabled: bool


def workflow_settings_from_config(config: WorkflowConfig) -> WorkflowSettingsState:
    return WorkflowSettingsState(
        max_rounds=config.discussion.max_rounds,
        checkpoint_every_n_rounds=config.discussion.checkpoint_every_n_rounds,
        reviewer_enabled=config.discussion.enable_reviewer_role,
        enabled_roles={role.duty: role.enabled for role in config.team_roles},
        summary_slots=list(config.context.summary_slots),
        arxiv_discovery_enabled=config.tooling.enable_arxiv_discovery,
        arxiv_download_enabled=config.tooling.download_arxiv_pdfs,
        arxiv_max_results=config.tooling.arxiv_max_results,
        python_artifact_enabled=config.tooling.enable_python_artifact,
        latex_artifact_enabled=config.tooling.enable_latex_artifact,
        bibtex_artifact_enabled=config.tooling.enable_bibtex_artifact,
        python_execution_test_enabled=config.tooling.enable_python_execution_test,
        python_full_execution_enabled=config.tooling.enable_python_full_execution,
        python_execution_timeout_seconds=config.tooling.python_execution_timeout_seconds,
        python_full_execution_timeout_seconds=config.tooling.python_full_execution_timeout_seconds,
        python_workspace_input_limit_mb=config.tooling.python_workspace_input_limit_mb,
        latex_compile_enabled=config.tooling.enable_latex_compile,
    )


def validate_workflow_settings(settings: WorkflowSettingsState, available_role_duties: list[str]) -> list[str]:
    errors: list[str] = []
    if settings.max_rounds < 1:
        errors.append("最大讨论轮数至少为 1。")
    if settings.checkpoint_every_n_rounds < 1:
        errors.append("检查点频率至少为 1 轮。")
    if settings.arxiv_max_results < 1:
        errors.append("arXiv 最大结果数至少为 1。")
    if settings.python_execution_timeout_seconds < 5:
        errors.append("Python 执行超时至少为 5 秒。")
    if settings.python_full_execution_timeout_seconds < 10:
        errors.append("Python 完整运行超时至少为 10 秒。")
    if settings.python_workspace_input_limit_mb < 1:
        errors.append("Python 工作目录输入上限至少为 1 MB。")
    if not any(settings.enabled_roles.get(duty, False) for duty in available_role_duties):
        errors.append("至少保留一个启用角色。")

    selected_slots = [slot for slot in settings.summary_slots if slot in SUMMARY_SLOT_OPTIONS]
    if not selected_slots:
        errors.append("至少选择一个结构化摘要槽位。")
    return errors


def apply_workflow_settings(config: WorkflowConfig, settings: WorkflowSettingsState) -> WorkflowConfig:
    errors = validate_workflow_settings(settings, [role.duty for role in config.team_roles])
    if errors:
        raise ValueError(" ".join(errors))

    summary_slots = [slot for slot in settings.summary_slots if slot in SUMMARY_SLOT_OPTIONS]
    if not summary_slots:
        summary_slots = list(DEFAULT_SUMMARY_SLOTS)

    updated_roles = [
        replace(role, enabled=bool(settings.enabled_roles.get(role.duty, role.enabled)))
        for role in config.team_template.roles
    ]

    return replace(
        config,
        discussion=replace(
            config.discussion,
            max_rounds=max(1, int(settings.max_rounds)),
            checkpoint_every_n_rounds=max(1, int(settings.checkpoint_every_n_rounds)),
            enable_reviewer_role=bool(settings.reviewer_enabled),
        ),
        context=replace(
            config.context,
            summary_slots=summary_slots,
        ),
        tooling=replace(
            config.tooling,
            enable_arxiv_discovery=bool(settings.arxiv_discovery_enabled),
            download_arxiv_pdfs=bool(settings.arxiv_download_enabled),
            arxiv_max_results=max(1, int(settings.arxiv_max_results)),
            enable_python_artifact=bool(settings.python_artifact_enabled),
            enable_latex_artifact=bool(settings.latex_artifact_enabled),
            enable_bibtex_artifact=bool(settings.bibtex_artifact_enabled),
            enable_python_execution_test=bool(settings.python_artifact_enabled and settings.python_execution_test_enabled),
            enable_python_full_execution=bool(settings.python_artifact_enabled and settings.python_full_execution_enabled),
            python_execution_timeout_seconds=max(5, int(settings.python_execution_timeout_seconds)),
            python_full_execution_timeout_seconds=max(10, int(settings.python_full_execution_timeout_seconds)),
            python_workspace_input_limit_mb=max(1, int(settings.python_workspace_input_limit_mb)),
            enable_latex_compile=bool(settings.latex_artifact_enabled and settings.latex_compile_enabled),
        ),
        team_template=replace(
            config.team_template,
            roles=updated_roles,
        ),
    )


def render_workflow_settings_summary(config: WorkflowConfig) -> str:
    enabled_roles = [role.label or role.duty for role in config.team_roles if role.enabled]
    slot_labels = [SUMMARY_SLOT_LABELS.get(slot, slot) for slot in config.context.summary_slots]
    reviewer_status = "开启" if config.discussion.enable_reviewer_role else "关闭"
    graph_summary = render_workflow_graph_summary(build_workflow_graph(config.workflow_template))
    return (
        f"回合数: {config.discussion.max_rounds} | 检查点: {config.discussion.checkpoint_every_n_rounds}\n"
        f"复核阶段: {reviewer_status} | arXiv: {'开启' if config.tooling.enable_arxiv_discovery else '关闭'} ({config.tooling.arxiv_max_results})\n"
        f"Python 产物: {'开启' if config.tooling.enable_python_artifact else '关闭'} | 冒烟测试: {'开启' if config.tooling.enable_python_execution_test else '关闭'} ({config.tooling.python_execution_timeout_seconds}s) | 完整运行: {'开启' if config.tooling.enable_python_full_execution else '关闭'} ({config.tooling.python_full_execution_timeout_seconds}s)\n"
        f"LaTeX: {'开启' if config.tooling.enable_latex_artifact else '关闭'} | BibTeX: {'开启' if config.tooling.enable_bibtex_artifact else '关闭'} | Tectonic: {'开启' if config.tooling.enable_latex_compile else '关闭'}\n"
        f"角色: {', '.join(enabled_roles) or '无'}\n"
        f"摘要槽位: {', '.join(slot_labels) or '无'}\n"
        f"{graph_summary.replace('Graph:', '工作流图:')}"
    )
