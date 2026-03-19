from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .arxiv_client import ArxivPaper, build_arxiv_query_candidates, download_arxiv_pdf, render_bibtex_entry, save_arxiv_metadata, search_arxiv
from .attachments import (
    build_attachment_index,
    load_attachment,
    render_attachment_snippets,
    render_evidence_catalog,
    render_literature_review_context,
    select_attachment_snippets,
    select_literature_review_snippets,
    split_literature_review_packets,
    summarize_snippet_coverage,
)
from .language import choose_language, detect_primary_language, language_name
from .llm_client import LLMError, OpenAICompatibleClient
from .pdf_reader import (
    build_reader_reference_attachments,
    load_pdf_reader_references,
    render_cached_pdf_reader_context,
    render_pdf_reader_references,
    select_pdf_reader_references,
)
from .models import (
    EXPERT_DUTY,
    HOST_DUTY,
    LEAD_DUTY,
    LITERATURE_DUTY,
    REPORT_DUTY,
    AttachmentPayload,
    AttachmentSnippet,
    DiscussionMessage,
    DiscussionResult,
    EvidenceCard,
    ProviderConfig,
    ReaderReference,
    StructuredLogEntry,
)
from .state import ApprovalRecord, Checkpoint, DiscussionState, ExperimentRunRecord, PaperRecord, ProjectStateManager, ResearchProject, WorkflowTask
from .team import ResearchTeam, TeamMember, build_research_team
from .tool_runtime import (
    BIBTEX_GENERATION_TOOL_KEY,
    LATEX_GENERATION_TOOL_KEY,
    PYTHON_EXECUTION_TOOL_KEY,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolRuntime,
    default_tool_runtime,
)
from .workflow import DiscussionRunRecord, ResearchDiscussionReviewWorkflow, WorkflowRuntimeContext
from .workflow_config import WorkflowConfig, WorkflowStageConfig, load_workflow_config

ACADEMIC_COLLABORATION_PRINCIPLE = (
    "This is a general academic research team. The specific field is determined by the user's task and may involve quantitative finance, image processing, physics, or another discipline."
    " The team objective is to reduce single-model hallucinations and improve reliability through division of labor, cross-questioning, mutual correction, and structured logging."
)

MEETING_RULES = [
    "Important claims must be tied to evidence, theoretical support, or explicitly marked as pending verification.",
    "Consensus, conflicts, hypotheses, and open questions must be recorded separately.",
    "Each role should handle only its assigned subproblem and avoid replaying long history.",
    "Hallucinations, vague definitions, and missing evidence must be logged explicitly.",
    "If open questions or conflicts remain after the main flow, follow-up discussion passes are mandatory.",
]

GENERATED_ARTIFACTS_DIR = Path("generated_artifacts")
ARXIV_LIBRARY_DIR = Path("arxiv_library")
PYTHON_EXECUTION_RUNS_DIR = GENERATED_ARTIFACTS_DIR / "execution_runs"
LATEX_BUILD_RUNS_DIR = GENERATED_ARTIFACTS_DIR / "latex_builds"

LEAD_ASSIGNMENT_PROMPT = f"""You are the lead of this simulated academic seminar team. Your only job is to decompose the research task and assign work. Do not perform the analysis yourself.

{ACADEMIC_COLLABORATION_PRINCIPLE}

Use the team's specialties to delegate work. Keep the schema labels below in English exactly as written:
Research Goal: ...
Domain: ...
1. Subproblem title | Owner: role name | Reviewer: role name | Rationale: why this assignment fits
2. Subproblem title | Owner: role name | Reviewer: role name | Rationale: why this assignment fits
Assignment Principles: ...

Rules:
1. The explanatory prose after each English label should follow the user's language.
2. Each subproblem must be specific, executable, and reviewable.
3. Match assignments to specialties whenever possible.
4. Each subproblem must have an Owner and preferably a Reviewer.
5. Do not write the analysis conclusions for the assignees.
6. Keep the plan concise but complete. Prefer 3 to 5 subproblems unless the task is unusually broad.
7. Keep the plan concise but complete. Around 380 to 520 words is acceptable when the task is non-trivial.
"""

HOST_COORDINATION_PROMPT = f"""你是这个模拟学术研讨团队的主持人，只负责统筹规划和协作节奏，不直接给出研究结论。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你会拿到当前会议状态、派工结果和会议规则。请输出一个简洁的执行计划，要求：
1. 明确子问题顺序和切换条件。
2. 提醒各执行角色如何互相校验、避免幻觉和跳步。
3. 提醒统稿人应记录哪些证据、争议和未决问题。
4. 如果主流程结束后仍存在争议，明确要求进入深挖阶段。
5. 不要过短；需要给出可执行的顺序、校验条件和收束标准，建议控制在 280 到 420 字。
"""

LITERATURE_DISCOVERY_PLAN_PROMPT = f"""你是这个模拟学术研讨团队中负责统筹资料补充的角色，只负责决定是否需要追加 arXiv 检索，以及应该检索哪些关键词。

{ACADEMIC_COLLABORATION_PRINCIPLE}

请只输出一个 JSON 对象，不要使用 Markdown 代码块。格式如下：
{{
  "needs_search": true,
  "queries": ["english keyword query 1", "english keyword query 2"],
  "reason": "一句话说明为什么要检索这些关键词"
}}

规则：
1. 只根据当前讨论暴露出的证据缺口、未决问题和已有论文库来决定。
2. queries 最多 3 条，必须是适合 arXiv 的短英文关键词短语，不要写成长句。
3. 如果当前材料已经足够，把 needs_search 设为 false，并让 queries 为空列表。
4. 不要重复已经很明确覆盖过的关键词，优先补最关键的证据缺口。
5. 不输出研究结论，只输出检索决策。
"""

LITERATURE_PACKET_NOTES_PROMPT = f"""You are the literature-review specialist inside this simulated academic research team.

{ACADEMIC_COLLABORATION_PRINCIPLE}

You are reading one packet from the attached literature, not the whole raw PDF.
Write packet-level reading notes only from the provided packet.

Output in Markdown and include these sections in order:
## Packet Coverage
## Problem And Setting
## Method Details
## Experiments And Results
## Limits Or Missing Pieces
## Evidence Anchors

Rules:
1. Stay inside the packet. If details are missing, say they are missing.
2. Prefer concrete module names, losses, datasets, comparisons, and findings.
3. Use the packet's evidence labels or internal IDs when anchoring claims.
4. Do not produce a final whole-paper verdict in this step.
"""

LITERATURE_REVIEW_PROMPT = f"""You are the literature-review specialist inside this simulated academic research team.

{ACADEMIC_COLLABORATION_PRINCIPLE}

You will receive packet-level reading notes that together cover the attached paper(s).
Synthesize them into a literature review for the rest of the expert team.

Output in Markdown and include these sections in order:
## Coverage
## Research Scope
## Method And Evidence
## Main Findings
## Limitations And Open Questions
## What The Expert Team Can Reuse

Rules:
1. Base the review only on the packet notes and coverage summary.
2. If coverage is partial, explicitly mark the review as partial.
3. Merge repeated points and keep the section titles exactly.
4. Prefer evidence-linked reconstruction over generic praise.
5. Never claim to have read sections that are not supported by the packet notes.
"""

EXPERT_ANALYSIS_PROMPT = f"""你是模拟学术研讨团队中的专家组成员。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你只负责当前被分配到的子问题。你会拿到：当前会议状态快照、相关证据片段、必要的文献综述。

请严格按以下模板输出，不要加寒暄：
[Judgment]
一句话给出当前判断。
[Reasons]
- 论点 1
- 论点 2
- 论点 3（可选）
[Evidence]
- 使用了哪些证据 ID，或还缺什么证据
[Risk]
- 当前不确定性、可能幻觉点或边界条件
[Handoff]
- 需要复核专家重点检查什么

要求：
1. 不要复述任务背景。
2. 每条论点尽量绑定证据或理论依据。
3. 不扩展到未分配子问题。
4. 只能引用提供给你的 Evidence ID；如果证据不足，直接写“缺少对应证据”，不要自造新编号。
5. 如果当前任务要求讨论机制、公式、实验或风险中的某一类，就只回答这一类，不要滑向泛泛的综合评价。
6. 不要只写空泛判断；在保证聚焦的前提下，建议控制在 420 到 720 字。
"""

EXPERT_REVIEW_PROMPT = f"""你是模拟学术研讨团队中的专家复核成员。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你只负责复核当前子问题的上一条专家发言。你会拿到：当前会议状态快照、相关证据片段、待复核发言。

请严格按以下模板输出，不要加寒暄：
[Verdict]
一句话说明是否接受上一条发言。
[Corrections]
- 需要修正的点 1
- 需要修正的点 2（可选）
[Evidence Check]
- 哪些证据足够，哪些证据缺失
[Residual Risk]
- 仍未解决的问题或剩余风险

要求：
1. 不重复完整分析，只做纠错、补证和边界说明。
2. 如未发现实质性问题，也要说明为何暂时接受。
3. 上一条专家发言和证据目录已经提供在下方，除非对应区块确实为空，否则不要误报“缺少输入”。
4. 只能引用提供给你的 Evidence ID；如果证据不足，直接说明缺口，不要自造新编号。
5. 如果上一条发言偏离当前子问题，你要明确指出“偏离任务边界”，而不是顺着它继续扩写。
6. 保持紧凑但不要过短，建议控制在 260 到 520 字。
"""

LITERATURE_ANALYSIS_PROMPT = f"""你是模拟学术研讨团队中的综述专家，被分配处理一个具体子问题。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你需要从文献、相关工作和证据支持角度回答当前子问题。请严格按以下模板输出：
[Judgment]
一句话给出文献层面的判断。
[Support]
- 哪些文献结论或证据支持当前判断
[Gap]
- 目前文献还缺什么
[Risk]
- 文献外推到当前问题的风险
[Handoff]
- 建议后续角色重点检查什么

要求：
1. 不要编造引用来源。
2. 重点说清“已有文献支持什么，不支持什么”。
3. 只能引用提供给你的 Evidence ID；如果证据不足，直接写缺口，不要自造新编号。
4. 如果当前子问题要求实证、benchmark 或相关工作映射，就不要扩展到泛泛的模型优劣综述。
5. 需要覆盖支持、空白和风险，建议控制在 360 到 640 字。
"""

REPORT_SYNTHESIS_PROMPT = f"""你是统稿人，被临时指定负责某个子问题的整合。

{ACADEMIC_COLLABORATION_PRINCIPLE}

请严格按以下模板输出：
[Judgment]
一句话给出当前整合判断。
[Synthesis]
- 汇总现有分析与证据
- 指出哪些部分已经稳定
[Open Gap]
- 仍缺什么证据或推导
[Handoff]
- 建议复核角色或下一角色重点检查什么

要求：
1. 只能整合已有结论，不要凭空新增事实。
2. 若现有材料不足，要明确说不足。
3. 只能引用提供给你的 Evidence ID；如果证据不足，要明确说证据不足，不要自造新编号。
4. 只整合当前子问题范围内的材料，不要把其他子问题的判断混进来。
5. 不要只做一两句拼接，建议控制在 320 到 560 字。
"""

HOST_REVIEW_PROMPT = f"""你是主持人，负责判断某个子问题是否已经可以收束，或是否需要继续深挖。

{ACADEMIC_COLLABORATION_PRINCIPLE}

请严格按以下模板输出：
[Verdict]
一句话说明当前子问题是“可以暂时收束”还是“仍未达成共识”。
[Coordination Decision]
- 说明是否进入下一步，或继续深挖什么
[Need More Work?]
- 仍需验证的核心点
[Residual Risk]
- 如果现在收束，会留下什么风险

要求：
1. 不直接代替专家给学术结论。
2. 重点判断“是否形成足够共识”。
3. 需要明确写出收束条件、遗留风险和下一步，建议控制在 260 到 420 字。
"""

FOLLOWUP_HOST_PROMPT = f"""你是主持人，正在主持未决问题的深入讨论收束。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你会拿到一个未决问题、两位角色的深挖发言和当前状态。请严格按以下模板输出：
[Verdict]
一句话说明该问题现在是否达成阶段共识。
[Coordination Decision]
- 若达成共识，说明形成了什么范围内的结论
- 若仍未解决，说明需要什么额外证据或实验
[Need More Work?]
- 下一步最必要动作
[Residual Risk]
- 目前仍保留的风险或争议

要求：
1. 明确写出“达成阶段共识”或“仍未达成共识”。
2. 不要回放长历史，只对当前未决问题作出主持判断。
3. 需要交代结论边界与后续动作，建议控制在 260 到 420 字。
"""

REPORT_LOG_PROMPT = f"""你是统稿人，也是会议状态的唯一写入者。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你会拿到：当前会议状态快照、最新一条发言、相关证据片段。
请不要写散文，只输出一个 JSON 对象，不要使用 Markdown 代码块。格式如下：
{{
  "headline": "一句话标题",
  "summary": "一到两句概括最新发言的有效信息",
  "consensus_add": ["新增共识 1"],
  "conflicts_add": ["新增争议 1"],
  "resolved_conflicts": ["已解决争议"],
  "open_questions_add": ["新增未决问题 1"],
  "resolved_questions": ["已解决问题"],
  "rejected_add": ["被否决的路线或假设"],
  "action_items_add": ["下一步动作"],
  "evidence_ids": ["E1", "E7"],
  "redundant": false
}}

规则：
1. 只记录与当前子问题直接相关的新信息。
2. 如果最新发言没有新增有效信息，把 redundant 设为 true，并让其余列表尽量为空。
3. evidence_ids 只能从提供的证据片段里选择。
4. 所有字符串都要简洁，避免重复原文。
"""

REPORT_SUMMARY_PROMPT = f"""你是统稿人，负责把会议状态整理为正式研究报告。

{ACADEMIC_COLLABORATION_PRINCIPLE}

请基于会议状态、检查点、证据账本、文献综述、已检索论文和实验运行结果输出 Markdown 报告，至少包含：
1. 任务概述与所属领域
2. 团队分工与专长
3. 子问题执行路径
4. 已固化共识
5. 关键争议与纠错处理
6. 最终综合结论
7. 后续建议

只保留经过讨论、互相质疑和复核后仍成立的结论，不要使用占位日期。
"""

MEETING_MINUTES_PROMPT = f"""你是统稿人，负责输出会议纪要。

{ACADEMIC_COLLABORATION_PRINCIPLE}

请基于会议状态、检查点、日志账本、已检索论文和实验运行结果输出 Markdown 会议纪要，至少包含：
1. 任务背景
2. 角色分工与专长
3. 执行流程与阶段切换
4. 共识、争议、未决问题
5. 证据引用与纠错记录
6. 结论与待办

如果讨论中止，请明确写出“会议中止”，并保留阶段性成果。不要使用占位日期。
"""

PROVIDER_PROMPT_BUDGETS = {
    "qwen": 2600,
    "glm": 12000,
    "deepseek": 10000,
    "minimax": 10000,
    "kimi": 12000,
}


@dataclass
class WorkPackage:
    index: int
    title: str
    description: str
    owner_name: str
    reviewer_name: str

    @property
    def display_text(self) -> str:
        if self.description:
            return f"{self.title}：{self.description}"
        return self.title


class DiscussionOrchestrator:
    def __init__(self, providers: list[ProviderConfig], workflow_config: WorkflowConfig | None = None) -> None:
        self.workflow_config = workflow_config or load_workflow_config()
        self.discussion_config = self.workflow_config.discussion
        self.routing_config = self.workflow_config.routing
        self.context_config = self.workflow_config.context
        self.report_options = self.workflow_config.report
        self.notes_options = self.workflow_config.notes
        self.state_manager = ProjectStateManager()
        self.tool_runtime: ToolRuntime = default_tool_runtime()
        self.team: ResearchTeam = build_research_team(providers, self.workflow_config)
        self.team_members_by_name = {member.provider.name: member for member in self.team.members}
        self.enabled_providers = [member.provider for member in self.team.members]
        self.providers_by_name = {provider.name: provider for provider in self.enabled_providers}

        self.lead_member = self.team.primary_member_for_duty(LEAD_DUTY)
        self.host_member = self.team.primary_member_for_duty(HOST_DUTY)
        self.literature_member = self.team.primary_member_for_duty(LITERATURE_DUTY)
        self.report_member = self.team.primary_member_for_duty(REPORT_DUTY)
        self.expert_members = self.team.members_for_duty(EXPERT_DUTY)

        self.lead_provider = self.lead_member.provider if self.lead_member is not None else None
        self.host_provider = self.host_member.provider if self.host_member is not None else None
        self.literature_provider = self.literature_member.provider if self.literature_member is not None else None
        self.report_provider = self.report_member.provider if self.report_member is not None else None
        self.expert_providers = [member.provider for member in self.expert_members]
        self.attachment_index: list[AttachmentSnippet] = []
        self.reader_references: list[ReaderReference] = []
        self.cached_pdf_reader_context = ""
        self.latest_result: DiscussionResult | None = None
        self.latest_project: ResearchProject | None = None
        self.latest_state: DiscussionState | None = None
        self.output_language = "en"

        if self.report_provider is None:
            fallback_member = self._team_member_for_action("synthesize_report")
            if fallback_member is None:
                fallback_member = next((member for member in self.team.members if member.provider is not self.lead_provider), None)
            self.report_member = fallback_member
            self.report_provider = fallback_member.provider if fallback_member is not None else None

    def run_discussion(
        self,
        user_request: str,
        attachments: list[AttachmentPayload],
        rounds: int = 0,
        generate_literature_review: bool = False,
        local_execution_authorized: bool = False,
        on_message: Callable[[DiscussionMessage], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> DiscussionResult:
        max_rounds = rounds if rounds > 0 else self.discussion_config.max_rounds

        self.output_language = detect_primary_language(user_request)
        result = DiscussionResult()
        self.latest_result = result
        project = self._initialize_project(user_request, attachments)
        state = project.discussion_state
        self.latest_project = project
        self.latest_state = state
        result.research_project = project
        result.meeting_state = state
        self._refresh_retrieval_context(attachments)
        if on_status is not None and self.reader_references:
            section_count = sum(1 for item in self.reader_references if item.kind == "section")
            figure_count = sum(1 for item in self.reader_references if item.kind == "figure")
            formula_count = sum(1 for item in self.reader_references if item.kind == "formula")
            on_status(
                self._tr(
                    f"已加载 PDF Reader 缓存：可在讨论中检索 {section_count} 个章节、{figure_count} 张图示、{formula_count} 条公式。",
                    f"Loaded PDF reader cache: {section_count} sections, {figure_count} figures, {formula_count} formulas are available for retrieval during discussion.",
                )
            )
            if formula_count == 0:
                on_status(
                    self._tr(
                        "当前加载的 PDF Reader 缓存还没有公式索引。请重建 PDF Reader 缓存以刷新章节、图示和公式。",
                        "The loaded PDF reader cache does not contain formula references yet. Rebuild the PDF reader cache to refresh sections, figures, and formulas.",
                    )
                )
        context = WorkflowRuntimeContext(
            user_request=user_request,
            attachments=attachments,
            max_rounds=max_rounds,
            generate_literature_review=generate_literature_review,
            local_execution_authorized=local_execution_authorized,
            result=result,
            project=project,
            state=state,
            state_manager=self.state_manager,
            on_message=on_message,
            on_status=on_status,
            should_cancel=should_cancel,
            assignments_text=self._render_fallback_assignments(),
            workpackages=self._fallback_workpackages()[:max_rounds],
            team_roster=self._build_team_roster(),
        )
        workflow = ResearchDiscussionReviewWorkflow(self.workflow_config.workflow_template)
        result = workflow.execute(self, context)
        self.state_manager.update_token_usage(
            state,
            usage={
                "estimated_input_chars": len(user_request) + sum(len(item.content) for item in attachments if item.kind != "image"),
                "estimated_output_chars": sum(len(message.content) for message in result.messages) + len(result.meeting_minutes) + len(result.final_summary),
                "message_count": len(result.messages),
                "workflow_stage_count": len(state.workflow_stage_records),
            },
        )
        return result

    def _initialize_project(self, user_request: str, attachments: list[AttachmentPayload]) -> ResearchProject:
        topic = self._collapse_whitespace(self._truncate_text(user_request, 160))
        return self.state_manager.start_project(
            topic=topic,
            user_question=user_request,
            uploaded_sources=[attachment.display_name for attachment in attachments],
            language=self.output_language,
            rules=self._localized_meeting_rules(),
            current_stage="Initialization",
            current_question="Waiting for the lead to decompose the task",
        )

    def _refresh_retrieval_context(self, attachments: list[AttachmentPayload]) -> None:
        self.attachment_index = build_attachment_index(attachments, existing_snippets=self.attachment_index)
        self.reader_references = load_pdf_reader_references(attachments)
        self.cached_pdf_reader_context = render_cached_pdf_reader_context(attachments, max_chars=9000)

    def _halt_workflow_context(self, context: WorkflowRuntimeContext, message: str) -> str:
        context.result = self._build_cancelled_result(context.result, context.state, message)
        context.halted = True
        return message

    def _workflow_stage_discover_literature(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if not self.workflow_config.tooling.enable_arxiv_discovery:
            return "arXiv discovery is disabled by workflow config."
        if context.should_cancel is not None and context.should_cancel():
            return self._halt_workflow_context(context, "Discussion was stopped manually.")
        if context.on_status is not None:
            context.on_status(
                self._tr(
                    "已跳过基于原始问题的预检索，后续会在主讨论后根据 Lead/专家识别出的证据缺口再做按需 arXiv 检索。",
                    "Skipped upfront arXiv discovery from the raw request. Targeted discovery will run after the primary discussion based on gaps identified by the lead and experts.",
                )
            )
        return self._tr(
            "已延后 arXiv 检索，等待讨论阶段给出更聚焦的关键词。",
            "Deferred arXiv discovery until the discussion stage proposes more focused search keywords.",
        )

    def _workflow_stage_expand_literature_search(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if not self.workflow_config.tooling.enable_arxiv_discovery:
            return self._tr("已关闭按需 arXiv 检索。", "Discussion-guided arXiv discovery is disabled by workflow config.")
        if context.should_cancel is not None and context.should_cancel():
            return self._halt_workflow_context(context, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

        queries, reason, planner_name = self._plan_discussion_guided_search_queries(context)
        if not queries:
            if context.on_status is not None:
                context.on_status(
                    self._tr(
                        "Lead/专家判断当前材料暂时足够，本轮不追加 arXiv 检索。",
                        "The lead/expert planner judged the current material sufficient for now, so no extra arXiv search will run in this round.",
                    )
                )
            return self._tr(
                "讨论后判断暂不需要追加 arXiv 检索。",
                "No extra arXiv search was needed after the primary discussion.",
            )

        if context.on_status is not None:
            context.on_status(
                self._tr(
                    f"{planner_name} 建议按需检索 arXiv，关键词：{'；'.join(queries)}",
                    f"{planner_name} proposed targeted arXiv discovery with queries: {'; '.join(queries)}",
                )
            )

        added_papers, downloaded_count = self._discover_arxiv_for_queries(
            context,
            queries=queries,
            selection_reason=reason or self._tr("由主讨论暴露的证据缺口触发的定向文献补充。", "Triggered by evidence gaps exposed during the primary discussion."),
        )
        if not added_papers:
            if context.on_status is not None:
                context.on_status(
                    self._tr(
                        "按需 arXiv 检索未新增论文，后续复核将继续基于现有材料推进。",
                        "The targeted arXiv expansion did not add new papers, so reviewer pass will continue with the existing material.",
                    )
                )
            return self._tr(
                f"已执行按需 arXiv 检索，但未新增论文。关键词：{'；'.join(queries)}",
                f"Ran discussion-guided arXiv discovery but added no new papers. Queries: {'; '.join(queries)}",
            )

        return self._tr(
            f"按需 arXiv 检索新增 {len(added_papers)} 篇论文，下载 {downloaded_count} 个 PDF。关键词：{'；'.join(queries)}",
            f"Discussion-guided arXiv discovery added {len(added_papers)} paper(s) and downloaded {downloaded_count} PDF(s). Queries: {'; '.join(queries)}",
        )

    def _discover_arxiv_for_queries(
        self,
        context: WorkflowRuntimeContext,
        *,
        queries: list[str],
        selection_reason: str,
    ) -> tuple[list[ArxivPaper], int]:
        existing_ids = {paper.paper_id for paper in context.state.literature_library}
        project_dir = ARXIV_LIBRARY_DIR / context.project.project_id
        selected: list[tuple[str, ArxivPaper]] = []
        downloaded_count = 0

        for query in queries:
            if len(selected) >= self.workflow_config.tooling.arxiv_max_results:
                break
            try:
                papers = search_arxiv(query, max_results=self.workflow_config.tooling.arxiv_max_results)
            except Exception as exc:  # noqa: BLE001
                if context.on_status is not None:
                    context.on_status(f"Targeted arXiv search failed for '{query}': {exc}")
                continue
            for paper in papers:
                if paper.paper_id in existing_ids:
                    continue
                existing_ids.add(paper.paper_id)
                selected.append((query, paper))
                if len(selected) >= self.workflow_config.tooling.arxiv_max_results:
                    break

        if not selected:
            return [], 0

        bib_entries = [paper.bibtex_entry for paper in context.state.literature_library if paper.bibtex_entry.strip()]
        selected_titles: list[str] = []
        added_papers: list[ArxivPaper] = []

        for index, (query, paper) in enumerate(selected, start=1):
            local_pdf_path = ""
            if self.workflow_config.tooling.download_arxiv_pdfs:
                try:
                    pdf_path = download_arxiv_pdf(paper, project_dir / "pdfs")
                    local_pdf_path = str(pdf_path)
                    attachment = load_attachment(str(pdf_path))
                    context.attachments.append(attachment)
                    if attachment.display_name not in context.state.uploaded_sources:
                        context.state.uploaded_sources.append(attachment.display_name)
                    downloaded_count += 1
                except Exception as exc:  # noqa: BLE001
                    if context.on_status is not None:
                        context.on_status(f"Skipping PDF download for {paper.paper_id}: {exc}")

            bibtex_key, bibtex_entry = render_bibtex_entry(paper)
            bib_entries.append(bibtex_entry)
            self.state_manager.record_paper(
                context.state,
                PaperRecord(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors=list(paper.authors),
                    abstract=paper.abstract,
                    categories=list(paper.categories),
                    published_at=paper.published_at,
                    updated_at=paper.updated_at,
                    entry_url=paper.entry_url,
                    pdf_url=paper.pdf_url,
                    local_pdf_path=local_pdf_path,
                    bibtex_key=bibtex_key,
                    bibtex_entry=bibtex_entry,
                    selection_reason=f"{selection_reason} | query={query} | rank={index}",
                ),
            )
            selected_titles.append(f"- {paper.title} ({paper.paper_id}) | query={query}")
            added_papers.append(paper)

        metadata_path = save_arxiv_metadata(context.state.literature_library, project_dir / "arxiv_metadata.json")
        self.state_manager.record_artifact(
            context.state,
            artifact_type="arxiv_metadata",
            title="arXiv metadata",
            path=str(metadata_path),
            preview=self._truncate_text("\n".join(selected_titles), 240),
            metadata={"paper_count": str(len(context.state.literature_library))},
        )

        if self.workflow_config.tooling.enable_bibtex_artifact and bib_entries:
            bib_result = self._execute_bibtex_artifact_tool(
                context,
                bib_entries,
                artifact_stem=self._artifact_stem(context.state.topic or context.user_request),
            )
            self._persist_tool_result(context, bib_result)

        self._refresh_retrieval_context(context.attachments)
        if context.on_status is not None:
            context.on_status(
                f"Targeted arXiv discovery added {len(added_papers)} paper(s); downloaded {downloaded_count} PDF(s) into {project_dir}."
            )
        return added_papers, downloaded_count

    def _workflow_stage_ingest_source_material(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if context.should_cancel is not None and context.should_cancel():
            return self._halt_workflow_context(context, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

        if context.generate_literature_review:
            if self.literature_provider is not None and context.attachments:
                if context.on_status is not None:
                    context.on_status(self._tr("综述专家正在生成文献综述", "Literature reviewer is generating the literature review"))
                literature_message = self._generate_literature_review(context.user_request)
                if not self._is_failed_message(literature_message):
                    context.result.literature_review = literature_message.content
                    context.literature_review_text = literature_message.content
                self._push_message(context.result, context.successful_messages, context.on_message, literature_message)
            elif context.on_status is not None:
                context.on_status(
                    self._tr(
                        "已启用文献综述，但未找到可用的综述专家或参考附件，已跳过。",
                        "Literature review is enabled, but no usable literature reviewer or reference attachment was found. Skipping.",
                    )
                )
        return self._tr(
            f"已摄取 {len(context.attachments)} 个附件，并准备工作流输入。",
            f"Ingested {len(context.attachments)} attachment(s) and prepared workflow inputs.",
        )

    def _workflow_stage_run_team_discussion(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if self.lead_provider is not None:
            if context.on_status is not None:
                context.on_status(self._tr("总负责人正在根据团队专长拆解任务", "Lead is decomposing the task based on team specialties"))
            lead_message = self._lead_assign(context.user_request, context.team_roster, context.literature_review_text)
            context.assignments_text = lead_message.content
            parsed = self._extract_workpackages(lead_message.content)
            if parsed:
                context.workpackages = parsed[: context.max_rounds]
            self._update_state_from_assignment(context.state, lead_message.content, context.workpackages)
            self._push_message(context.result, context.successful_messages, context.on_message, lead_message)

        if self.host_provider is not None:
            if context.should_cancel is not None and context.should_cancel():
                return self._halt_workflow_context(context, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))
            if context.on_status is not None:
                context.on_status(self._tr("主持人正在准备协作安排", "Host is preparing the coordination plan"))
            host_message = self._host_coordinate(
                context.user_request,
                context.assignments_text,
                context.team_roster,
                context.state,
                context.literature_review_text,
            )
            self._update_state_from_coordination(context.state, host_message.content)
            self._push_message(context.result, context.successful_messages, context.on_message, host_message)

        kickoff_source = context.successful_messages[-1] if context.successful_messages else None
        kickoff_snippets = self._select_relevant_snippets(context.user_request, self.report_provider)
        kickoff_message, kickoff_entry = self._build_log_entry(
            user_request=context.user_request,
            state=context.state,
            source_message=kickoff_source,
            workpackage_title="项目启动",
            index=0,
            relevant_snippets=kickoff_snippets,
            fallback_text="已创建项目日志，等待各角色按专长执行子问题。",
        )
        self._record_log(
            context.result,
            context.successful_messages,
            context.on_message,
            context.log_messages,
            context.state,
            kickoff_message,
            kickoff_entry,
        )
        self._create_checkpoint(context.state, label="初始化", workpackage_index=0)

        if not self.expert_providers:
            return self._tr("当前没有可执行讨论的专家角色。", "No expert role is available for the discussion stage.")

        context.discussion_runs = []
        for workpackage in context.workpackages:
            if context.should_cancel is not None and context.should_cancel():
                return self._halt_workflow_context(context, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

            owner = self._resolve_owner(workpackage.owner_name, fallback_index=workpackage.index - 1)
            reviewer = self._resolve_reviewer(workpackage.reviewer_name, owner)
            self.state_manager.ensure_task(
                context.state,
                task_id=self._task_id_for_workpackage(workpackage),
                title=workpackage.title,
                description=workpackage.description,
                owner_name=owner.name,
                reviewer_name=reviewer.name if reviewer is not None else "",
                round_index=workpackage.index,
                source_kind="assignment",
            )
            self.state_manager.begin_task(
                context.state,
                task_id=self._task_id_for_workpackage(workpackage),
                stage_label=f"Task {workpackage.index}: {workpackage.title}",
                question=workpackage.display_text,
                round_index=workpackage.index,
            )

            if context.on_status is not None:
                context.on_status(
                    self._tr(
                        f"任务 {workpackage.index} 已启动：{workpackage.display_text}",
                        f"Task {workpackage.index} started: {workpackage.display_text}",
                    )
                )

            primary_snippets = self._select_relevant_snippets(
                f"{context.user_request}\n{workpackage.display_text}\n{owner.specialty}\n{' '.join(context.state.open_questions[-3:])}",
                owner,
            )
            primary_message = self._run_primary_assignment(
                provider=owner,
                user_request=context.user_request,
                assignments_text=context.assignments_text,
                team_roster=context.team_roster,
                literature_review_text=context.literature_review_text,
                workpackage=workpackage,
                state=context.state,
                relevant_snippets=primary_snippets,
                attachments=context.attachments,
            )
            self._push_message(context.result, context.successful_messages, context.on_message, primary_message)

            primary_log_message, primary_entry = self._build_log_entry(
                user_request=context.user_request,
                state=context.state,
                source_message=primary_message,
                workpackage_title=workpackage.display_text,
                index=workpackage.index,
                relevant_snippets=primary_snippets,
                fallback_text=self._fallback_log_line(primary_message, workpackage.display_text),
            )
            self._record_log(
                context.result,
                context.successful_messages,
                context.on_message,
                context.log_messages,
                context.state,
                primary_log_message,
                primary_entry,
            )
            context.discussion_runs.append(
                DiscussionRunRecord(
                    workpackage=workpackage,
                    owner=owner,
                    reviewer=reviewer,
                    primary_snippets=primary_snippets,
                    primary_message=primary_message,
                )
            )

        return self._tr(
            f"已完成 {len(context.discussion_runs)} 个任务的主讨论轮次。",
            f"Completed primary discussion passes for {len(context.discussion_runs)} workpackage(s).",
        )

    def _workflow_stage_run_reviewer_pass(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if not self.discussion_config.enable_reviewer_role:
            return self._tr("配置已关闭 reviewer pass。", "Reviewer pass is disabled by workflow config.")
        if not context.discussion_runs:
            return self._tr("没有待复核的任务。", "No workpackage is available for reviewer pass.")

        reviewed_count = 0
        for run in context.discussion_runs:
            if run.reviewer is None:
                continue
            if context.should_cancel is not None and context.should_cancel():
                return self._halt_workflow_context(context, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

            review_snippets = self._select_relevant_snippets(
                f"{context.user_request}\n{run.workpackage.display_text}\n{run.reviewer.specialty}\n{run.primary_message.content}",
                run.reviewer,
            )
            review_snippets = self._augment_review_snippets(
                review_snippets,
                state=context.state,
                workpackage=run.workpackage,
                previous_message=run.primary_message,
                provider=run.reviewer,
            )
            review_message = self._run_review_assignment(
                provider=run.reviewer,
                user_request=context.user_request,
                assignments_text=context.assignments_text,
                team_roster=context.team_roster,
                literature_review_text=context.literature_review_text,
                workpackage=run.workpackage,
                previous_message=run.primary_message,
                state=context.state,
                relevant_snippets=review_snippets,
            )
            self._push_message(context.result, context.successful_messages, context.on_message, review_message)

            review_log_message, review_entry = self._build_log_entry(
                user_request=context.user_request,
                state=context.state,
                source_message=review_message,
                workpackage_title=run.workpackage.display_text,
                index=run.workpackage.index,
                relevant_snippets=review_snippets,
                fallback_text=self._fallback_log_line(review_message, run.workpackage.display_text),
            )
            self._record_log(
                context.result,
                context.successful_messages,
                context.on_message,
                context.log_messages,
                context.state,
                review_log_message,
                review_entry,
            )
            run.review_snippets = review_snippets
            run.review_message = review_message
            reviewed_count += 1

        return self._tr(
            f"已完成 {reviewed_count} 个 reviewer pass。",
            f"Completed reviewer passes for {reviewed_count} workpackage(s).",
        )

    def _workflow_stage_update_structured_state(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if not context.discussion_runs and not self.expert_providers:
            self.state_manager.update_summary(context.state, self._build_fallback_report(context.state))
            return self._tr("已基于当前状态生成阶段总结。", "Generated an intermediate summary from the current structured state.")

        for run in context.discussion_runs:
            context.completed_rounds += 1
            self.state_manager.complete_task(
                context.state,
                task_id=self._task_id_for_workpackage(run.workpackage),
                notes=self._truncate_text(run.primary_message.content, 240),
            )
            checkpoint = self._maybe_create_checkpoint(
                context.state,
                label=run.workpackage.title,
                workpackage_index=run.workpackage.index,
                completed_rounds=context.completed_rounds,
            )
            if checkpoint is not None and context.on_status is not None:
                context.on_status(
                    self._tr(
                        f"{self._checkpoint_label(checkpoint.checkpoint_id)} 已记录：{checkpoint.label}",
                        f"{self._checkpoint_label(checkpoint.checkpoint_id)} recorded: {checkpoint.label}",
                    )
                )

        context.completed_rounds = self._run_consensus_followups(
            result=context.result,
            state=context.state,
            user_request=context.user_request,
            attachments=context.attachments,
            assignments_text=context.assignments_text,
            team_roster=context.team_roster,
            literature_review_text=context.literature_review_text,
            successful_messages=context.successful_messages,
            log_messages=context.log_messages,
            on_message=context.on_message,
            on_status=context.on_status,
            should_cancel=context.should_cancel,
            completed_rounds=context.completed_rounds,
        )
        if context.halted:
            return self._tr("工作流已中止。", "Workflow was halted.")

        self.state_manager.update_summary(context.state, self._build_fallback_report(context.state))
        return self._tr("结构化状态已完成收束与更新。", "Structured state was consolidated and updated.")

    def _workflow_stage_run_experiment_cycle(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if context.halted:
            return "Workflow halted; skipped experiment cycle."
        if not self.workflow_config.tooling.enable_python_artifact:
            return "Python artifact generation is disabled by workflow config."

        artifact_stem = self._artifact_stem(context.state.topic or context.user_request)
        if context.on_status is not None:
            context.on_status("Experiment cycle started: generating a Python draft artifact from the current structured state.")
        result = self._execute_structured_artifact_tool(
            context=context,
            tool_key=PYTHON_EXECUTION_TOOL_KEY,
            title=f"{artifact_stem}_analysis_draft",
            artifact_type="python_script",
        )
        created_paths = self._persist_tool_result(context, result, allow_local_execution=context.local_execution_authorized)
        if not created_paths:
            return "Experiment cycle did not produce any executable artifact."
        return f"Experiment cycle generated {len(created_paths)} artifact(s)."

    def _workflow_stage_generate_meeting_notes(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if context.halted:
            return self._tr("工作流已中止，跳过会议纪要。", "Workflow halted; skipped meeting notes generation.")
        if context.on_status is not None:
            context.on_status(self._tr("统稿人正在根据会议状态撰写会议纪要", "Reporter is drafting the meeting minutes from the meeting state"))
        context.result.meeting_minutes = self._generate_meeting_minutes(
            user_request=context.user_request,
            team_roster=context.team_roster,
            state=context.state,
            literature_review_text=context.literature_review_text,
            final_report="",
            cancelled=False,
        )
        return self._tr("会议纪要已生成。", "Meeting notes were generated.")

    def _workflow_stage_generate_research_report(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if context.halted:
            return self._tr("工作流已中止，跳过研究报告。", "Workflow halted; skipped research report generation.")
        if context.on_status is not None:
            context.on_status(self._tr("统稿人正在根据会议状态生成研究报告", "Reporter is synthesizing the research report from the meeting state"))
        context.result.final_summary = self._generate_report(
            context.user_request,
            context.team_roster,
            context.state,
            context.literature_review_text,
        )
        self.state_manager.update_summary(context.state, context.result.final_summary)
        generated = self._generate_optional_artifacts(context)
        if not generated:
            return self._tr("研究报告已生成。", "Research report was generated.")
        return self._tr(
            f"研究报告已生成，并额外生成 {len(generated)} 个产物。",
            f"Research report was generated, plus {len(generated)} additional artifact(s).",
        )

    def _workflow_stage_compile_latex_artifacts(self, context: WorkflowRuntimeContext, stage: WorkflowStageConfig) -> str:
        del stage
        if context.halted:
            return "Workflow halted; skipped Tectonic build."
        if not self.workflow_config.tooling.enable_latex_artifact:
            return "LaTeX artifact generation is disabled by workflow config."
        if not self.workflow_config.tooling.enable_latex_compile:
            return "Tectonic compile is disabled by workflow config."

        latest_tex = next((artifact for artifact in reversed(context.state.generated_artifacts) if artifact.artifact_type == "latex_document"), None)
        if latest_tex is None:
            return "No LaTeX document artifact is available to compile."
        if not context.local_execution_authorized:
            self.state_manager.record_approval(
                context.state,
                approval_type="local_execution",
                scope=f"latex_compile:{Path(latest_tex.path).name}",
                granted=False,
                details="Local execution was not authorized for this discussion run.",
            )
            if context.on_status is not None:
                context.on_status("Tectonic compile is enabled, but the current discussion run did not grant local execution authorization.")
            return "Tectonic compile was skipped because local execution was not authorized."

        build_outputs = self._compile_latex_artifact(context, Path(latest_tex.path))
        if not build_outputs:
            return "Tectonic compile finished without producing a build log or PDF."
        return f"Tectonic compile produced {len(build_outputs)} build artifact(s)."

    def _plan_discussion_guided_search_queries(self, context: WorkflowRuntimeContext) -> tuple[list[str], str, str]:
        planner = (
            self.lead_provider
            or self.literature_provider
            or self.host_provider
            or (self.expert_providers[0] if self.expert_providers else None)
        )
        fallback_queries = self._fallback_discussion_guided_queries(context)
        planner_name = planner.name if planner is not None else self._tr("系统回退策略", "system fallback")
        if planner is None:
            return fallback_queries, self._tr("未找到可用的 Lead/专家，改用状态回退策略生成检索词。", "No lead or expert was available, so a state-based fallback planned the search queries."), planner_name

        prompt = (
            f"用户任务：\n{context.user_request}\n\n"
            f"当前会议状态：\n{self._build_state_snapshot(context.state, mode='host')}\n\n"
            f"最近任务更新：\n{self._render_recent_entries(self._select_recent_entries(context.state, None))}\n\n"
            f"当前论文库：\n{self._build_literature_library_snapshot(context.state)}\n\n"
            "请判断是否需要为了下一步 reviewer pass 追加 arXiv 检索，并给出最多 3 个英文关键词查询。"
        )
        raw = self._chat(
            provider=planner,
            system_prompt=LITERATURE_DISCOVERY_PLAN_PROMPT,
            user_prompt=prompt,
            max_tokens=260,
            max_continuations=0,
        )
        payload = self._extract_json_object(raw)
        if payload is None:
            return fallback_queries, self._tr("检索规划输出不规范，已回退到状态驱动关键词。", "Search planning output was malformed, so the workflow fell back to state-driven queries."), planner_name

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return fallback_queries, self._tr("检索规划 JSON 解析失败，已回退到状态驱动关键词。", "Search planning JSON could not be parsed, so the workflow fell back to state-driven queries."), planner_name

        needs_search = bool(data.get("needs_search"))
        queries = self._normalize_search_queries(self._coerce_str_list(data.get("queries")))
        reason = str(data.get("reason") or "").strip()
        if not needs_search:
            return [], reason, planner_name
        if queries:
            return queries, reason, planner_name
        return fallback_queries, reason or self._tr("检索规划未给出有效关键词，已回退到状态驱动关键词。", "The planner did not return usable queries, so the workflow fell back to state-driven queries."), planner_name

    def _fallback_discussion_guided_queries(self, context: WorkflowRuntimeContext) -> list[str]:
        candidate_texts: list[str] = []
        candidate_texts.extend(entry.summary for entry in context.state.log_entries[-4:] if entry.summary.strip())
        candidate_texts.extend(context.state.open_questions[-3:])
        candidate_texts.extend(task.display_text for task in context.state.workflow_tasks[-3:])
        candidate_texts.append(context.user_request)
        return self._normalize_search_queries(candidate_texts)

    def _normalize_search_queries(self, raw_queries: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_query in raw_queries:
            candidates = build_arxiv_query_candidates(raw_query)
            candidate = candidates[0] if candidates else " ".join(raw_query.split()).strip()
            candidate = " ".join(candidate.split()).strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(candidate)
            if len(normalized) >= 3:
                break
        return normalized

    def _tr(self, zh_text: str, en_text: str) -> str:
        return choose_language(self.output_language, zh_text, en_text)

    def _localized_meeting_rules(self) -> list[str]:
        return [
            "Important claims must be tied to evidence, theoretical support, or explicitly marked as pending verification.",
            "Consensus, conflicts, hypotheses, and open questions must be recorded separately.",
            "Each role should focus only on its assigned subproblem and avoid replaying long history.",
            "If hallucinations, vague definitions, or missing evidence appear, they must be logged explicitly.",
            "If open questions or conflicts remain after the main flow, follow-up discussion passes are mandatory.",
        ]

    def _team_member_for_action(self, action: str, *, fallback_duty: str = "") -> TeamMember | None:
        if fallback_duty:
            duty_member = self.team.primary_member_for_duty(fallback_duty)
            if duty_member is not None and duty_member.supports_action(action):
                return duty_member
        member = self.team.primary_member_for_action(action)
        if member is not None:
            return member
        if fallback_duty:
            return self.team.primary_member_for_duty(fallback_duty)
        return None

    def _member_for_provider(self, provider: ProviderConfig | None) -> TeamMember | None:
        if provider is None:
            return None
        return self.team_members_by_name.get(provider.name)

    def _provider_for_action(self, action: str, *, fallback: ProviderConfig | None = None, fallback_duty: str = "") -> ProviderConfig | None:
        member = self._team_member_for_action(action, fallback_duty=fallback_duty)
        if member is not None:
            return member.provider
        return fallback

    def _task_id_for_workpackage(self, workpackage: WorkPackage) -> str:
        return f"task_{workpackage.index}_{self._topic_key(workpackage.title).replace(' ', '_') or 'workpackage'}"

    def _language_policy(self) -> str:
        if self.output_language == "zh":
            return (
                "Language requirement: the model-generated reply body, literature review body, report body, minutes body, and JSON string values should follow the same primary language as the user's request."
                " The detected primary language is Chinese."
                " Keep paper titles, model names, formulas, evidence IDs, and fixed template markers such as [Judgment], [Risk], and ## Coverage in their original form."
                " System labels and framework text outside the generated reply body may remain in English."
            )
        return (
            "Language requirement: the model-generated reply body, literature review body, report body, minutes body, and JSON string values should follow the same primary language as the user's request."
            f" The detected primary language is {language_name(self.output_language)}."
            " You may keep paper titles, model names, formulas, evidence IDs, and fixed template markers such as [Judgment], [Risk], and ## Coverage in their original form."
            " System labels and framework text outside the generated reply body may remain in English."
        )

    def _generate_optional_artifacts(self, context: WorkflowRuntimeContext) -> list[Path]:
        created_paths: list[Path] = []
        artifact_stem = self._artifact_stem(context.state.topic or context.user_request)
        if self.workflow_config.tooling.enable_bibtex_artifact and context.state.literature_library:
            bib_entries = [paper.bibtex_entry for paper in context.state.literature_library if paper.bibtex_entry.strip()]
            if bib_entries:
                result = self._execute_bibtex_artifact_tool(context, bib_entries, artifact_stem=artifact_stem)
                created_paths.extend(self._persist_tool_result(context, result))
        if self.workflow_config.tooling.enable_latex_artifact:
            result = self._execute_structured_artifact_tool(
                context=context,
                tool_key=LATEX_GENERATION_TOOL_KEY,
                title=f"{artifact_stem}_report_draft",
                artifact_type="latex_document",
            )
            created_paths.extend(self._persist_tool_result(context, result))
        return created_paths

    def _execute_structured_artifact_tool(
        self,
        *,
        context: WorkflowRuntimeContext,
        tool_key: str,
        title: str,
        artifact_type: str,
    ) -> ToolExecutionResult:
        member = self._team_member_for_tool(tool_key)
        if member is None:
            return ToolExecutionResult(
                tool_key=tool_key,
                status="failed",
                message="No enabled role can use this tool.",
                error_message=f"No enabled role can use tool '{tool_key}'.",
            )
        evidence_labels = [card.display_label or card.evidence_id for card in context.state.evidence_cards[-8:]]
        artifact_stem = self._artifact_stem(context.state.topic or context.user_request)
        payload = {
            "title": title,
            "topic": context.state.topic or context.user_request,
            "summary": context.result.final_summary or context.state.summary,
            "minutes": context.result.meeting_minutes,
            "consensus": list(context.state.consensus_points),
            "open_questions": list(context.state.open_questions),
            "action_items": list(context.state.action_items),
            "evidence_labels": evidence_labels,
            "source_names": list(context.state.uploaded_sources),
            "bibtex_keys": [paper.bibtex_key for paper in context.state.literature_library if paper.bibtex_key],
            "bibliography_basename": f"{artifact_stem}_references",
            "artifact_type": artifact_type,
        }
        request = ToolExecutionRequest(
            tool_key=tool_key,
            role_key=member.role_key,
            payload=payload,
            action="generate_artifact",
            project_id=context.project.project_id,
            user_request=context.user_request,
            working_directory=str(GENERATED_ARTIFACTS_DIR),
        )
        return self.tool_runtime.execute(request)

    def _execute_bibtex_artifact_tool(self, context: WorkflowRuntimeContext, bib_entries: list[str], *, artifact_stem: str) -> ToolExecutionResult:
        member = self._team_member_for_tool(BIBTEX_GENERATION_TOOL_KEY)
        if member is None:
            return ToolExecutionResult(
                tool_key=BIBTEX_GENERATION_TOOL_KEY,
                status="failed",
                message="No enabled role can generate BibTeX artifacts.",
                error_message=f"No enabled role can use tool '{BIBTEX_GENERATION_TOOL_KEY}'.",
            )
        request = ToolExecutionRequest(
            tool_key=BIBTEX_GENERATION_TOOL_KEY,
            role_key=member.role_key,
            payload={
                "title": f"{artifact_stem}_references",
                "bibtex_entries": bib_entries,
                "path_hint": f"generated_artifacts/{artifact_stem}_references.bib",
            },
            action="generate_artifact",
            project_id=context.project.project_id,
            user_request=context.user_request,
            working_directory=str(GENERATED_ARTIFACTS_DIR),
        )
        return self.tool_runtime.execute(request)

    def _persist_tool_result(
        self,
        context: WorkflowRuntimeContext,
        result: ToolExecutionResult,
        *,
        allow_local_execution: bool = False,
    ) -> list[Path]:
        if not result.ok or not result.artifacts:
            if context.on_status is not None:
                context.on_status(
                    f"Optional artifact step skipped for {result.tool_key}: {result.error_message or result.message or 'no output'}"
                )
            return []

        GENERATED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        created_paths: list[Path] = []
        for artifact in result.artifacts:
            target = self._artifact_output_path(artifact.path_hint, artifact.title, artifact.artifact_type)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(artifact.content, encoding="utf-8")
            created_paths.append(target)
            self.state_manager.record_artifact(
                context.state,
                artifact_type=artifact.artifact_type,
                title=artifact.title,
                path=str(target),
                preview=self._truncate_text(artifact.content, 240),
                metadata=dict(artifact.metadata),
            )
            if artifact.artifact_type == "python_script":
                python_exec_enabled = (
                    self.workflow_config.tooling.enable_python_execution_test
                    or self.workflow_config.tooling.enable_python_full_execution
                )
                if python_exec_enabled and allow_local_execution:
                    smoke_passed = True
                    if self.workflow_config.tooling.enable_python_execution_test:
                        smoke_outputs, smoke_passed = self._run_python_smoke_test(context, target)
                        created_paths.extend(smoke_outputs)
                    if self.workflow_config.tooling.enable_python_full_execution:
                        if smoke_passed:
                            full_outputs, _ = self._run_python_full_execution(context, target)
                            created_paths.extend(full_outputs)
                        else:
                            self.state_manager.record_approval(
                                context.state,
                                approval_type="local_execution",
                                scope=f"python_execution:{target.name}:full",
                                granted=False,
                                details="Full run was skipped because the smoke test reported issues.",
                            )
                            if context.on_status is not None:
                                context.on_status(
                                    f"Full Python run was skipped for {target.name} because the smoke test reported issues. Review the smoke-test log first."
                                )
                elif python_exec_enabled:
                    enabled_modes: list[str] = []
                    if self.workflow_config.tooling.enable_python_execution_test:
                        enabled_modes.append("smoke")
                    if self.workflow_config.tooling.enable_python_full_execution:
                        enabled_modes.append("full")
                    scope_suffix = "+".join(enabled_modes) or "local"
                    self.state_manager.record_approval(
                        context.state,
                        approval_type="local_execution",
                        scope=f"python_execution:{target.name}:{scope_suffix}",
                        granted=False,
                        details="Local execution was not authorized for this discussion run.",
                    )
                    if context.on_status is not None:
                        context.on_status(
                            f"Python artifact {target.name} was generated, but local {scope_suffix} execution was skipped because this run did not grant execution authorization."
                        )
        if context.on_status is not None and created_paths:
            created_text = ", ".join(path.name for path in created_paths)
            context.on_status(f"Generated optional artifacts: {created_text}")
        elif context.on_status is not None and result.artifacts:
            created_text = ", ".join(
                self._artifact_output_path(artifact.path_hint, artifact.title, artifact.artifact_type).name
                for artifact in result.artifacts
            )
            context.on_status(f"Generated optional artifacts: {created_text}")
        return created_paths

    def _artifact_output_path(self, path_hint: str, title: str, artifact_type: str) -> Path:
        candidate = Path(path_hint) if path_hint else GENERATED_ARTIFACTS_DIR / self._topic_key(title)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == GENERATED_ARTIFACTS_DIR.name:
            return candidate
        suffix_map = {
            "latex_document": ".tex",
            "bibtex_library": ".bib",
            "python_script": ".py",
        }
        suffix = suffix_map.get(artifact_type, ".txt")
        stem = self._topic_key(title) or artifact_type
        return GENERATED_ARTIFACTS_DIR / f"{stem}{suffix}"

    def _team_member_for_tool(self, tool_key: str) -> TeamMember | None:
        for member in self.team.members:
            if self.tool_runtime.permission_policy.can_use(member.role_key, tool_key):
                return member
        return None

    def _run_python_smoke_test(self, context: WorkflowRuntimeContext, script_path: Path) -> tuple[list[Path], bool]:
        return self._run_python_execution(
            context,
            script_path,
            run_mode="smoke",
            timeout_seconds=self.workflow_config.tooling.python_execution_timeout_seconds,
            approval_details="User authorized local Python smoke testing for this discussion run.",
            status_label="smoke test",
        )

    def _run_python_full_execution(self, context: WorkflowRuntimeContext, script_path: Path) -> tuple[list[Path], bool]:
        return self._run_python_execution(
            context,
            script_path,
            run_mode="full",
            timeout_seconds=self.workflow_config.tooling.python_full_execution_timeout_seconds,
            approval_details="User authorized full local Python execution for this discussion run.",
            status_label="full run",
        )

    def _run_python_execution(
        self,
        context: WorkflowRuntimeContext,
        script_path: Path,
        *,
        run_mode: str,
        timeout_seconds: int,
        approval_details: str,
        status_label: str,
    ) -> tuple[list[Path], bool]:
        self.state_manager.record_approval(
            context.state,
            approval_type="local_execution",
            scope=f"python_execution:{script_path.name}:{run_mode}",
            granted=True,
            details=f"{approval_details} Interpreter: {sys.executable}",
        )
        if context.on_status is not None:
            context.on_status(f"Authorized Python {status_label} started for {script_path.name} in the current interpreter")

        workspace = self._prepare_runtime_workspace(
            context=context,
            category_dir=PYTHON_EXECUTION_RUNS_DIR,
            stem=f"{script_path.stem}_{run_mode}",
        )
        execution_script = workspace / script_path.name
        shutil.copy2(script_path, execution_script)
        manifest_path, mapped_inputs, skipped_inputs = self._map_workspace_inputs(context, workspace)
        runtime_env = self._build_python_runtime_env(workspace=workspace, manifest_path=manifest_path, run_mode=run_mode)

        compile_cmd = [sys.executable, "-m", "py_compile", execution_script.name]
        run_cmd = [sys.executable, execution_script.name]
        compile_result = self._run_subprocess(compile_cmd, cwd=workspace, timeout_seconds=timeout_seconds, env=runtime_env)
        run_result = self._run_subprocess(run_cmd, cwd=workspace, timeout_seconds=timeout_seconds, env=runtime_env)

        log_text = (
            f"Python {status_label} for: {script_path.name}\n"
            f"Run mode: {run_mode}\n"
            f"Interpreter: {sys.executable}\n\n"
            f"Source script: {script_path}\n"
            f"Execution workspace: {workspace}\n\n"
            f"Input manifest: {manifest_path}\n"
            f"Mapped inputs: {len(mapped_inputs)}\n"
            f"Skipped inputs: {len(skipped_inputs)}\n"
            f"Execution timeout: {timeout_seconds} seconds\n"
            f"Mapped input limit: {self.workflow_config.tooling.python_workspace_input_limit_mb} MB\n\n"
            "== Compile Check ==\n"
            f"Exit code: {compile_result['returncode']}\n"
            f"Stdout:\n{compile_result['stdout'] or '(empty)'}\n\n"
            f"Stderr:\n{compile_result['stderr'] or '(empty)'}\n\n"
            "== Runtime Check ==\n"
            f"Exit code: {run_result['returncode']}\n"
            f"Stdout:\n{run_result['stdout'] or '(empty)'}\n\n"
            f"Stderr:\n{run_result['stderr'] or '(empty)'}\n"
        )
        if skipped_inputs:
            skipped_text = "\n".join(
                f"- {item['display_name']} | reason={item['reason']}"
                for item in skipped_inputs
            )
            log_text += f"\n\n== Skipped Inputs ==\n{skipped_text}\n"
        log_path = workspace / f"{script_path.stem}_{run_mode}_run_log.txt"
        log_path.write_text(log_text, encoding="utf-8")
        self.state_manager.record_artifact(
            context.state,
            artifact_type="python_input_manifest",
            title=f"{script_path.stem} input manifest",
            path=str(manifest_path),
            preview=self._truncate_text(manifest_path.read_text(encoding="utf-8"), 240),
            metadata={"source_script": str(script_path), "working_directory": str(workspace), "run_mode": run_mode},
        )
        self.state_manager.record_artifact(
            context.state,
            artifact_type="python_run_log",
            title=f"{script_path.stem} {status_label} log",
            path=str(log_path),
            preview=self._truncate_text(log_text, 240),
            metadata={"source_script": str(script_path), "working_directory": str(workspace), "run_mode": run_mode},
        )
        run_status = "passed" if compile_result["returncode"] == 0 and run_result["returncode"] == 0 else "needs_attention"
        self.state_manager.record_experiment_run(
            context.state,
            ExperimentRunRecord(
                run_id=f"run_{len(context.state.experiment_runs) + 1}",
                script_path=str(execution_script),
                working_directory=str(workspace),
                interpreter_path=sys.executable,
                run_mode=run_mode,
                command=[sys.executable, execution_script.name],
                compile_returncode=int(compile_result["returncode"]),
                runtime_returncode=int(run_result["returncode"]),
                log_path=str(log_path),
                stdout_excerpt=self._truncate_text(str(run_result["stdout"] or ""), 240),
                stderr_excerpt=self._truncate_text(
                    "\n".join(
                        part for part in [str(compile_result["stderr"] or ""), str(run_result["stderr"] or "")] if part
                    ),
                    240,
                ),
                status=run_status,
                authorized=True,
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        self._record_experiment_state_feedback(
            context,
            script_path=execution_script,
            log_path=log_path,
            compile_result=compile_result,
            run_result=run_result,
            run_mode=run_mode,
        )
        success = compile_result["returncode"] == 0 and run_result["returncode"] == 0
        if context.on_status is not None:
            if success:
                context.on_status(
                    f"Python {status_label} passed for {script_path.name} inside {workspace.name} with {len(mapped_inputs)} mapped input(s)"
                )
            else:
                context.on_status(f"Python {status_label} finished with issues for {script_path.name}; see {log_path.name}")
        return [manifest_path, log_path], success

    def _run_subprocess(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: int = 20,
        env: dict[str, str] | None = None,
    ) -> dict[str, str | int]:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -9,
                "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
                "stderr": ((exc.stderr or "").strip() if isinstance(exc.stderr, str) else "") or f"Process timed out after {timeout_seconds} seconds.",
            }
        except Exception as exc:  # noqa: BLE001
            return {"returncode": -1, "stdout": "", "stderr": str(exc)}

    def _record_experiment_state_feedback(
        self,
        context: WorkflowRuntimeContext,
        *,
        script_path: Path,
        log_path: Path,
        compile_result: dict[str, str | int],
        run_result: dict[str, str | int],
        run_mode: str,
    ) -> None:
        success = int(compile_result["returncode"]) == 0 and int(run_result["returncode"]) == 0
        mode_label = "full run" if run_mode == "full" else "smoke test"
        headline = f"Experiment {mode_label} {'passed' if success else 'needs attention'}"
        summary = (
            f"Local {mode_label} for {script_path.name} "
            f"{'passed both compile and runtime checks' if success else 'reported compile/runtime issues'}."
        )
        entry = StructuredLogEntry(
            workpackage_index=max(context.completed_rounds, 1),
            workpackage_title="Experiment cycle",
            speaker=self.report_provider.name if self.report_provider is not None else "System",
            stage="experiment",
            headline=headline,
            summary=summary,
            consensus_add=[f"{script_path.name} passed the authorized local {mode_label}."] if success else [],
            conflicts_add=[] if success else [f"{script_path.name} produced issues during the authorized local {mode_label}."],
            open_questions_add=[] if success else [f"What changes are needed before {script_path.name} can complete a clean {mode_label}?"],
            action_items_add=[] if success else [f"Inspect {log_path.name} and revise the generated experiment scaffold."],
        )
        self.state_manager.apply_log_entry(
            context.state,
            entry=entry,
            max_history_items=self.context_config.max_history_items,
            max_log_entries=self.context_config.max_log_entries,
            max_evidence_cards=self.context_config.max_evidence_cards,
        )

    def _compile_latex_artifact(self, context: WorkflowRuntimeContext, tex_path: Path) -> list[Path]:
        self.state_manager.record_approval(
            context.state,
            approval_type="local_execution",
            scope=f"latex_compile:{tex_path.name}:tectonic",
            granted=True,
            details="User authorized local Tectonic compilation for this discussion run.",
        )
        tectonic = shutil.which("tectonic")
        if tectonic is None:
            if context.on_status is not None:
                context.on_status("Tectonic compile skipped because tectonic was not found on PATH.")
            return []

        bib_path = next(
            (Path(artifact.path) for artifact in reversed(context.state.generated_artifacts) if artifact.artifact_type == "bibtex_library"),
            None,
        )
        build_dir = self._prepare_runtime_workspace(
            context=context,
            category_dir=LATEX_BUILD_RUNS_DIR,
            stem=tex_path.stem,
        )
        command = [
            tectonic,
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(build_dir),
            "--untrusted",
            tex_path.name,
        ]
        result = self._run_subprocess(command, cwd=tex_path.parent, timeout_seconds=90)
        log_parts = [
            "== Tectonic Build ==",
            f"Command: {' '.join(command)}",
            f"Source document: {tex_path}",
            f"Build directory: {build_dir}",
            f"Bibliography: {bib_path if bib_path is not None and bib_path.exists() else 'not provided'}",
            f"Exit code: {result['returncode']}",
            f"Stdout:\n{result['stdout'] or '(empty)'}",
            f"Stderr:\n{result['stderr'] or '(empty)'}",
        ]
        engine_log_path = build_dir / f"{tex_path.stem}.log"
        if engine_log_path.exists():
            log_parts.extend(
                [
                    "Engine log:",
                    engine_log_path.read_text(encoding="utf-8", errors="replace").strip() or "(empty)",
                ]
            )

        log_text = "\n\n".join(log_parts).strip() + "\n"
        log_path = build_dir / f"{tex_path.stem}_tectonic_build_log.txt"
        log_path.write_text(log_text, encoding="utf-8")
        self.state_manager.record_artifact(
            context.state,
            artifact_type="latex_build_log",
            title=f"{tex_path.stem} build log",
            path=str(log_path),
            preview=self._truncate_text(log_text, 240),
            metadata={
                "source_document": str(tex_path),
                "compiler": "tectonic",
                "build_directory": str(build_dir),
            },
        )

        created_paths = [log_path]
        pdf_path = build_dir / f"{tex_path.stem}.pdf"
        if pdf_path.exists():
            self.state_manager.record_artifact(
                context.state,
                artifact_type="latex_pdf",
                title=f"{tex_path.stem} pdf",
                path=str(pdf_path),
                preview="Compiled PDF output from the LaTeX draft.",
                metadata={
                    "source_document": str(tex_path),
                    "compiler": "tectonic",
                    "build_directory": str(build_dir),
                },
            )
            created_paths.append(pdf_path)
        if context.on_status is not None:
            if pdf_path.exists():
                context.on_status(f"Tectonic compile completed for {tex_path.name}; built {pdf_path.name}.")
            else:
                context.on_status(f"Tectonic compile finished for {tex_path.name}; see {log_path.name} for diagnostics.")
        return created_paths

    def _prepare_runtime_workspace(self, *, context: WorkflowRuntimeContext, category_dir: Path, stem: str) -> Path:
        project_key = self._artifact_stem(context.project.project_id or "project")
        stem_key = self._artifact_stem(stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        workspace = category_dir / project_key / f"{timestamp}_{stem_key}"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _map_workspace_inputs(
        self,
        context: WorkflowRuntimeContext,
        workspace: Path,
    ) -> tuple[Path, list[dict[str, object]], list[dict[str, str]]]:
        inputs_dir = workspace / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = self.workflow_config.tooling.python_workspace_input_limit_mb * 1024 * 1024
        used_bytes = 0
        mapped_inputs: list[dict[str, object]] = []
        skipped_inputs: list[dict[str, str]] = []

        for attachment in context.attachments:
            source_path = attachment.path
            if not source_path.exists():
                skipped_inputs.append({"display_name": attachment.display_name, "reason": "source_missing"})
                continue
            file_size = source_path.stat().st_size
            if used_bytes + file_size > max_bytes:
                skipped_inputs.append({"display_name": attachment.display_name, "reason": "input_limit_exceeded"})
                continue

            target = self._unique_workspace_input_path(inputs_dir, source_path.name)
            shutil.copy2(source_path, target)
            used_bytes += file_size
            mapped_inputs.append(
                {
                    "display_name": attachment.display_name,
                    "kind": attachment.kind,
                    "source_path": str(source_path),
                    "mapped_path": str(target),
                    "relative_path": str(target.relative_to(workspace)),
                    "size_bytes": file_size,
                }
            )

        manifest = {
            "project_id": context.project.project_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "workspace": str(workspace),
            "input_limit_mb": self.workflow_config.tooling.python_workspace_input_limit_mb,
            "mapped_input_count": len(mapped_inputs),
            "mapped_total_bytes": used_bytes,
            "mapped_inputs": mapped_inputs,
            "skipped_inputs": skipped_inputs,
        }
        manifest_path = workspace / "input_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path, mapped_inputs, skipped_inputs

    def _build_python_runtime_env(self, *, workspace: Path, manifest_path: Path, run_mode: str) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CYBER_COLLOQUIUM_WORKSPACE"] = str(workspace)
        env["CYBER_COLLOQUIUM_INPUT_DIR"] = str(workspace / "inputs")
        env["CYBER_COLLOQUIUM_INPUT_MANIFEST"] = str(manifest_path)
        env["CYBER_COLLOQUIUM_RUN_MODE"] = run_mode
        env["CYBER_COLLOQUIUM_SMOKE_TEST"] = "1" if run_mode == "smoke" else "0"
        env["CYBER_COLLOQUIUM_FULL_RUN"] = "1" if run_mode == "full" else "0"
        env["CYBER_COLLOQUIUM_PYTHON_EXECUTABLE"] = sys.executable
        return env

    def _unique_workspace_input_path(self, directory: Path, file_name: str) -> Path:
        candidate = directory / file_name
        if not candidate.exists():
            return candidate
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix
        counter = 2
        while True:
            candidate = directory / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _lead_assign(
        self,
        user_request: str,
        team_roster: str,
        literature_review_text: str,
    ) -> DiscussionMessage:
        lead_provider = self._provider_for_action("decompose_task", fallback=self.lead_provider, fallback_duty=LEAD_DUTY)
        assert lead_provider is not None
        snippets = self._select_relevant_snippets(user_request, lead_provider, max_snippets=4)
        reader_context = self._build_reader_context(user_request, lead_provider, max_chars=1800, max_items=4)
        reader_attachments = self._build_reader_attachments(user_request, lead_provider, max_items=2)
        prompt = (
            f"User task:\n{user_request}\n\n"
            f"Team roster and specialties:\n{team_roster}\n\n"
            f"Relevant attachment snippets:\n{render_attachment_snippets(snippets, max_chars=2200) or 'No attachment snippets.'}\n\n"
            f"PDF reader retrieval:\n{reader_context}\n\n"
            f"Literature review context:\n{self._build_literature_context(literature_review_text, 1400)}\n\n"
            "Generate the delegation plan using the fixed English schema labels."
        )
        content = self._chat(
            provider=lead_provider,
            system_prompt=LEAD_ASSIGNMENT_PROMPT,
            user_prompt=prompt,
            max_tokens=900,
            attachments=reader_attachments,
            max_continuations=2,
        )
        return DiscussionMessage(
            speaker=lead_provider.name,
            role="assistant",
            content=content,
            round_index=0,
            model_name=lead_provider.model,
            duty=lead_provider.duty,
            stage="assignment",
        )

    def _host_coordinate(
        self,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        state: DiscussionState,
        literature_review_text: str,
    ) -> DiscussionMessage:
        host_provider = self._provider_for_action("coordinate_workflow", fallback=self.host_provider, fallback_duty=HOST_DUTY)
        assert host_provider is not None
        host_query = f"{user_request}\n{assignments_text}\n{state.current_question}"
        reader_context = self._build_reader_context(host_query, host_provider, max_chars=1600, max_items=4)
        reader_attachments = self._build_reader_attachments(host_query, host_provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前会议状态：\n{self._build_state_snapshot(state, workpackage=None, mode='host')}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1200)}\n\n"
            "请输出执行安排。"
        )
        content = self._chat(
            provider=host_provider,
            system_prompt=HOST_COORDINATION_PROMPT,
            user_prompt=prompt,
            max_tokens=720,
            attachments=reader_attachments,
            max_continuations=1,
        )
        return DiscussionMessage(
            speaker=host_provider.name,
            role="assistant",
            content=content,
            round_index=0,
            model_name=host_provider.model,
            duty=host_provider.duty,
            stage="coordination",
        )

    def _run_primary_assignment(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
        attachments: list[AttachmentPayload],
    ) -> DiscussionMessage:
        if provider.duty == LITERATURE_DUTY:
            return self._run_literature_assignment(
                provider=provider,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                literature_review_text=literature_review_text,
                workpackage=workpackage,
                state=state,
                relevant_snippets=relevant_snippets,
            )
        if provider.duty == REPORT_DUTY:
            return self._run_report_assignment(
                provider=provider,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                literature_review_text=literature_review_text,
                workpackage=workpackage,
                state=state,
                relevant_snippets=relevant_snippets,
            )
        if provider.duty == HOST_DUTY:
            return self._run_host_assignment(
                provider=provider,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                workpackage=workpackage,
                state=state,
            )
        return self._run_primary_expert(
            provider=provider,
            user_request=user_request,
            assignments_text=assignments_text,
            team_roster=team_roster,
            literature_review_text=literature_review_text,
            workpackage=workpackage,
            state=state,
            relevant_snippets=relevant_snippets,
            attachments=attachments,
        )

    def _run_review_assignment(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        previous_message: DiscussionMessage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        if provider.duty == HOST_DUTY:
            return self._run_host_review(
                provider=provider,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                workpackage=workpackage,
                previous_message=previous_message,
                state=state,
            )
        if provider.duty == REPORT_DUTY:
            return self._run_report_review(
                provider=provider,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                workpackage=workpackage,
                previous_message=previous_message,
                state=state,
            )
        if provider.duty == LITERATURE_DUTY:
            return self._run_literature_review_assignment(
                provider=provider,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                literature_review_text=literature_review_text,
                workpackage=workpackage,
                previous_message=previous_message,
                state=state,
                relevant_snippets=relevant_snippets,
            )
        return self._run_reviewer(
            provider=provider,
            user_request=user_request,
            assignments_text=assignments_text,
            team_roster=team_roster,
            literature_review_text=literature_review_text,
            workpackage=workpackage,
            previous_message=previous_message,
            state=state,
            relevant_snippets=relevant_snippets,
        )

    def _run_primary_expert(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
        attachments: list[AttachmentPayload],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{provider.specialty}\n{state.current_question}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1800, max_items=5)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        evidence_catalog = render_evidence_catalog(relevant_snippets, max_items=8) or self._tr("暂无可引用的 Evidence ID。", "No Evidence ID is available for citation.")
        allowed_evidence_ids = {snippet.evidence_id for snippet in relevant_snippets}
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"任务边界：本轮只允许回答“{workpackage.title}”，不要扩展到其他子问题或泛化结论。\n"
            f"主责角色：{provider.name}\n"
            f"你的专长：{provider.specialty or '未填写'}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='expert')}\n\n"
            f"相关证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=self._evidence_budget(provider)) or '暂无可检索证据片段。'}\n\n"
            f"允许引用的 Evidence ID：\n{evidence_catalog}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1200)}\n\n"
            "请只处理当前子问题。只能引用上面证据目录中的 Evidence ID；如果证据不够，请直接说明缺少证据。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=EXPERT_ANALYSIS_PROMPT,
            user_prompt=prompt,
            max_tokens=1100,
            attachments=self._merge_chat_attachments(attachments, reader_attachments),
            max_continuations=2,
            required_sections=["[Judgment]", "[Reasons]", "[Evidence]", "[Risk]", "[Handoff]"],
            min_chars=180,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="analysis",
        )

    def _run_reviewer(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        previous_message: DiscussionMessage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{provider.specialty}\n{previous_message.content}"
        provider_key = self._provider_key(provider)
        review_snippets = self._augment_review_snippets(
            relevant_snippets,
            state=state,
            workpackage=workpackage,
            previous_message=previous_message,
            provider=provider,
        )
        reader_context = self._build_reader_context(
            reader_query,
            provider,
            max_chars=700 if provider_key == "qwen" else 1000,
            max_items=3,
        )
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        evidence_catalog = render_evidence_catalog(review_snippets, max_items=6) or self._tr("暂无可引用的 Evidence ID。", "No Evidence ID is available for citation.")
        allowed_evidence_ids = {snippet.evidence_id for snippet in review_snippets}
        state_snapshot = self._truncate_text(
            self._build_state_snapshot(state, workpackage=workpackage, mode="reviewer"),
            540 if provider_key == "qwen" else 900,
        )
        evidence_snippets = render_attachment_snippets(
            review_snippets,
            max_chars=720 if provider_key == "qwen" else min(1500, self._evidence_budget(provider)),
        ) or "暂无可检索证据片段。"
        literature_context = self._build_literature_context(
            literature_review_text,
            260 if provider_key == "qwen" else 600,
        )
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"任务边界：只复核“{workpackage.title}”是否成立，不要顺着上一条发言扩展到其他子问题。\n"
            f"复核角色：{provider.name}\n"
            f"你的专长：{provider.specialty or '未填写'}\n\n"
            f"待复核发言：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 880 if provider_key == 'qwen' else 1400)}\n\n"
            f"当前会议状态快照：\n{state_snapshot}\n\n"
            f"允许引用的 Evidence ID：\n{evidence_catalog}\n\n"
            f"相关证据片段：\n{evidence_snippets}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{literature_context}\n\n"
            f"团队成员简表：\n{self._truncate_text(team_roster, 220)}\n\n"
            f"派工摘要：\n{self._truncate_text(assignments_text, 260)}\n\n"
            "请基于你的专长直接完成复核。你已经收到了待复核发言、会议状态快照和证据目录，不要回答“输入缺失”或“未收到待复核内容”；如果证据不足，只指出哪条主张缺证据。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=EXPERT_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=720,
            attachments=reader_attachments,
            max_continuations=1,
            required_sections=["[Verdict]", "[Corrections]", "[Evidence Check]", "[Residual Risk]"],
            min_chars=140,
            allowed_evidence_ids=allowed_evidence_ids,
            reject_missing_review_input=True,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="review",
        )

    def _run_literature_assignment(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{provider.specialty}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=2200, max_items=6)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=3)
        evidence_catalog = render_evidence_catalog(relevant_snippets, max_items=8) or self._tr("暂无可引用的 Evidence ID。", "No Evidence ID is available for citation.")
        allowed_evidence_ids = {snippet.evidence_id for snippet in relevant_snippets}
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"任务边界：只梳理与“{workpackage.title}”直接相关的文献支持、空白和风险。\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='expert')}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1500)}\n\n"
            f"相关文献证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=2200) or '暂无可检索证据片段。'}\n\n"
            f"允许引用的 Evidence ID：\n{evidence_catalog}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            "请从文献支持角度回答当前子问题。只能引用上面证据目录中的 Evidence ID；如果证据不够，请直接说明缺口。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=LITERATURE_ANALYSIS_PROMPT,
            user_prompt=prompt,
            max_tokens=900,
            attachments=reader_attachments,
            max_continuations=2,
            required_sections=["[Judgment]", "[Support]", "[Gap]", "[Risk]", "[Handoff]"],
            min_chars=170,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="literature_analysis",
        )

    def _run_literature_review_assignment(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        previous_message: DiscussionMessage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{previous_message.content}"
        provider_key = self._provider_key(provider)
        review_snippets = self._augment_review_snippets(
            relevant_snippets,
            state=state,
            workpackage=workpackage,
            previous_message=previous_message,
            provider=provider,
        )
        reader_context = self._build_reader_context(
            reader_query,
            provider,
            max_chars=760 if provider_key == "qwen" else 1100,
            max_items=4,
        )
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=3)
        evidence_catalog = render_evidence_catalog(review_snippets, max_items=6) or self._tr("暂无可引用的 Evidence ID。", "No Evidence ID is available for citation.")
        allowed_evidence_ids = {snippet.evidence_id for snippet in review_snippets}
        state_snapshot = self._truncate_text(
            self._build_state_snapshot(state, workpackage=workpackage, mode="reviewer"),
            560 if provider_key == "qwen" else 940,
        )
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"任务边界：只从相关工作角度复核“{workpackage.title}”，不要扩展到泛化综述。\n"
            f"待复核发言：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 920 if provider_key == 'qwen' else 1500)}\n\n"
            f"当前会议状态快照：\n{state_snapshot}\n\n"
            f"允许引用的 Evidence ID：\n{evidence_catalog}\n\n"
            f"相关文献证据片段：\n{render_attachment_snippets(review_snippets, max_chars=760 if provider_key == 'qwen' else 1600) or '暂无可检索证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 320 if provider_key == 'qwen' else 720)}\n\n"
            f"团队成员简表：\n{self._truncate_text(team_roster, 220)}\n\n"
            f"派工摘要：\n{self._truncate_text(assignments_text, 260)}\n\n"
            "请从文献支持和相关工作角度直接复核。你已经收到了待复核发言、会议状态快照和证据目录，不要误报“输入缺失”；只能引用上面证据目录中的 Evidence ID。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=EXPERT_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=680,
            attachments=reader_attachments,
            max_continuations=1,
            required_sections=["[Verdict]", "[Corrections]", "[Evidence Check]", "[Residual Risk]"],
            min_chars=140,
            allowed_evidence_ids=allowed_evidence_ids,
            reject_missing_review_input=True,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="literature_review",
        )

    def _run_report_assignment(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        workpackage: WorkPackage,
        state: DiscussionState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{state.current_question}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1500, max_items=4)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        evidence_catalog = render_evidence_catalog(relevant_snippets, max_items=8) or self._tr("暂无可引用的 Evidence ID。", "No Evidence ID is available for citation.")
        allowed_evidence_ids = {snippet.evidence_id for snippet in relevant_snippets}
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"任务边界：只整合“{workpackage.title}”范围内已有材料，不要把其他任务的判断混入本轮结论。\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='report')}\n\n"
            f"相关证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=1800) or '暂无可检索证据片段。'}\n\n"
            f"允许引用的 Evidence ID：\n{evidence_catalog}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1000)}\n\n"
            "请整合已有观点，给出当前子问题的结构化综合判断。只能引用上面证据目录中的 Evidence ID；如果证据不足，要明确说不足。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=REPORT_SYNTHESIS_PROMPT,
            user_prompt=prompt,
            max_tokens=780,
            attachments=reader_attachments,
            max_continuations=1,
            required_sections=["[Judgment]", "[Synthesis]", "[Open Gap]", "[Handoff]"],
            min_chars=150,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="synthesis",
        )

    def _run_report_review(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        workpackage: WorkPackage,
        previous_message: DiscussionMessage,
        state: DiscussionState,
    ) -> DiscussionMessage:
        provider_key = self._provider_key(provider)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"待复核整合稿：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 920 if provider_key == 'qwen' else 1400)}\n\n"
            f"当前会议状态快照：\n{self._truncate_text(self._build_state_snapshot(state, workpackage=workpackage, mode='report'), 520 if provider_key == 'qwen' else 860)}\n\n"
            f"团队成员简表：\n{self._truncate_text(team_roster, 220)}\n\n"
            f"派工摘要：\n{self._truncate_text(assignments_text, 260)}\n\n"
            "请判断当前整合稿是否足够支撑收束。待复核整合稿和会议状态都已提供，不要误报输入缺失。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=HOST_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=640,
            max_continuations=1,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
            min_chars=120,
            reject_missing_review_input=True,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="review",
        )

    def _run_host_assignment(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        workpackage: WorkPackage,
        state: DiscussionState,
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{state.current_question}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1400, max_items=4)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='host')}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            "请从主持与协调角度给出该子问题的执行收束建议。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=HOST_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=640,
            attachments=reader_attachments,
            max_continuations=1,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
            min_chars=120,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="coordination_review",
        )

    def _run_host_review(
        self,
        *,
        provider: ProviderConfig,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        workpackage: WorkPackage,
        previous_message: DiscussionMessage,
        state: DiscussionState,
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{previous_message.content}"
        provider_key = self._provider_key(provider)
        reader_context = self._build_reader_context(
            reader_query,
            provider,
            max_chars=680 if provider_key == "qwen" else 900,
            max_items=3,
        )
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"待主持复核发言：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 920 if provider_key == 'qwen' else 1400)}\n\n"
            f"当前会议状态快照：\n{self._truncate_text(self._build_state_snapshot(state, workpackage=workpackage, mode='host'), 520 if provider_key == 'qwen' else 820)}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"团队成员简表：\n{self._truncate_text(team_roster, 220)}\n\n"
            f"派工摘要：\n{self._truncate_text(assignments_text, 260)}\n\n"
            "请判断该子问题是否可以暂时收束。待主持复核发言和会议状态都已提供，不要误报输入缺失。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=HOST_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=640,
            attachments=reader_attachments,
            max_continuations=1,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
            min_chars=120,
            reject_missing_review_input=True,
        )
        return DiscussionMessage(
            speaker=provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=provider.model,
            duty=provider.duty,
            stage="coordination_review",
        )

    def _build_log_entry(
        self,
        *,
        user_request: str,
        state: DiscussionState,
        source_message: DiscussionMessage | None,
        workpackage_title: str,
        index: int,
        relevant_snippets: list[AttachmentSnippet],
        fallback_text: str,
    ) -> tuple[DiscussionMessage, StructuredLogEntry]:
        log_provider = self._provider_for_action("log_state", fallback=self.report_provider, fallback_duty=REPORT_DUTY)
        if source_message is None or log_provider is None or self._is_failed_message(source_message):
            fallback_entry = self._fallback_structured_log_entry(
                source_message=source_message,
                workpackage_title=workpackage_title,
                index=index,
                relevant_snippets=relevant_snippets,
                fallback_text=fallback_text,
            )
            return self._log_message_from_entry(fallback_entry), fallback_entry

        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"当前子问题：任务 {index} - {workpackage_title}\n\n"
            f"当前会议状态：\n{self._build_state_snapshot(state, workpackage_index=index, mode='logger')}\n\n"
            f"候选证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=1500) or '暂无证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{self._build_reader_context(f'{user_request}\\n{workpackage_title}\\n{source_message.content}', log_provider, max_chars=1400, max_items=4)}\n\n"
            f"最新讨论内容：\n[{source_message.speaker} | {source_message.stage}]\n{self._truncate_text(source_message.content, 1200)}\n\n"
            "请输出状态补丁 JSON。"
        )
        content = self._chat(
            provider=log_provider,
            system_prompt=REPORT_LOG_PROMPT,
            user_prompt=prompt,
            max_tokens=480,
            attachments=self._build_reader_attachments(f"{user_request}\n{workpackage_title}\n{source_message.content}", log_provider, max_items=2),
            max_continuations=0,
        )
        entry = self._parse_structured_log_entry(
            raw_text=content,
            source_message=source_message,
            workpackage_title=workpackage_title,
            index=index,
            relevant_snippets=relevant_snippets,
            fallback_text=fallback_text,
        )
        return self._log_message_from_entry(entry), entry

    def _generate_report(
        self,
        user_request: str,
        team_roster: str,
        state: DiscussionState,
        literature_review_text: str,
    ) -> str:
        report_provider = self._provider_for_action("synthesize_report", fallback=self.report_provider or self.host_provider or self.lead_provider)
        if report_provider is None:
            return self._build_fallback_report(state)

        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"报告生成选项：\n{self._report_policy_hint()}\n\n"
            f"会议状态总览：\n{self._build_state_snapshot(state, workpackage=None, mode='report')}\n\n"
            f"已检索论文库：\n{self._build_literature_library_snapshot(state)}\n\n"
            f"实验运行记录：\n{self._build_experiment_run_snapshot(state)}\n\n"
            f"检查点序列：\n{self._build_checkpoint_timeline(state)}\n\n"
            f"证据账本：\n{self._build_evidence_ledger(state)}\n\n"
            f"PDF reader 索引检索：\n{self._build_reader_context(user_request + chr(10) + state.current_question, report_provider, max_chars=1800, max_items=5)}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1600)}\n\n"
            "请输出最终研究报告。"
        )
        content = self._chat(
            provider=report_provider,
            system_prompt=REPORT_SUMMARY_PROMPT,
            user_prompt=prompt,
            max_tokens=2200,
            attachments=self._build_reader_attachments(user_request + "\n" + state.current_question, report_provider, max_items=2),
            max_continuations=2,
        )
        return self._sanitize_document(content)

    def _generate_meeting_minutes(
        self,
        *,
        user_request: str,
        team_roster: str,
        state: DiscussionState,
        literature_review_text: str,
        final_report: str,
        cancelled: bool,
    ) -> str:
        minutes_provider = self._provider_for_action("write_minutes", fallback=self.report_provider or self.host_provider or self.lead_provider)
        if minutes_provider is None:
            if cancelled:
                return self._build_cancelled_minutes(state)
            summary_source = state.summary or self._truncate_text(self._collapse_whitespace(final_report), 400)
            return self._build_fallback_minutes(state, summary_source)

        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"会议纪要选项：\n{self._notes_policy_hint()}\n\n"
            f"会议状态总览：\n{self._build_state_snapshot(state, workpackage=None, mode='minutes')}\n\n"
            f"已检索论文库：\n{self._build_literature_library_snapshot(state)}\n\n"
            f"实验运行记录：\n{self._build_experiment_run_snapshot(state)}\n\n"
            f"检查点序列：\n{self._build_checkpoint_timeline(state)}\n\n"
            f"证据账本：\n{self._build_evidence_ledger(state)}\n\n"
            f"PDF reader 索引检索：\n{self._build_reader_context(user_request + chr(10) + state.current_question, minutes_provider, max_chars=1600, max_items=5)}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1200)}\n\n"
            f"最终报告摘要：\n{self._truncate_text(final_report, 2000)}\n\n"
            f"讨论状态：{'会议中止' if cancelled else '会议完成'}"
        )
        content = self._chat(
            provider=minutes_provider,
            system_prompt=MEETING_MINUTES_PROMPT,
            user_prompt=prompt,
            max_tokens=1800,
            attachments=self._build_reader_attachments(user_request + "\n" + state.current_question, minutes_provider, max_items=2),
            max_continuations=2,
        )
        return self._sanitize_document(content)

    def _report_policy_hint(self) -> str:
        lines = [
            f"- include_consensus: {'yes' if self.report_options.include_consensus else 'no'}",
            f"- include_open_questions: {'yes' if self.report_options.include_open_questions else 'no'}",
            f"- include_action_items: {'yes' if self.report_options.include_action_items else 'no'}",
        ]
        return "\n".join(lines)

    def _notes_policy_hint(self) -> str:
        return f"- include_role_labels: {'yes' if self.notes_options.include_role_labels else 'no'}"

    def _chat(
        self,
        *,
        provider: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        attachments: list[AttachmentPayload] | None = None,
        max_continuations: int = 1,
    ) -> str:
        client = OpenAICompatibleClient(provider)
        try:
            return client.chat(
                system_prompt=f"{self._language_policy()}\n\n{system_prompt}",
                user_prompt=self._truncate_text(f"{self._language_policy()}\n\n{user_prompt}", self._prompt_budget(provider)),
                attachments=attachments,
                max_tokens=max_tokens,
                max_continuations=max_continuations,
            )
        except LLMError as exc:
            return f"[Call Failed] {exc}"

    def _chat_with_sections(
        self,
        *,
        provider: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        required_sections: list[str],
        attachments: list[AttachmentPayload] | None = None,
        max_continuations: int = 0,
        min_chars: int = 96,
        allowed_evidence_ids: set[str] | None = None,
        reject_missing_review_input: bool = False,
    ) -> str:
        content = self._chat(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            attachments=attachments,
            max_continuations=max_continuations,
        )
        invalid_evidence_ids = self._invalid_evidence_ids(content, allowed_evidence_ids)
        missing_input_claim = reject_missing_review_input and self._claims_missing_review_input(content)
        if not self._response_needs_repair(
            content,
            required_sections,
            min_chars=min_chars,
            allowed_evidence_ids=allowed_evidence_ids,
            reject_missing_review_input=reject_missing_review_input,
        ):
            return content

        repair_prompt = (
            f"{user_prompt}\n\n"
            + self._tr(
                "你上一条输出格式不合格，存在内容过短、字段缺失或只输出标签的问题。",
                "Your previous output did not follow the required structure. It was too short, missed fields, or only repeated the labels.",
            )
            + " "
            + self._tr(
                f"你的输出必须包含这些区块：{', '.join(required_sections)}。\n\n",
                f"Your output must contain these sections: {', '.join(required_sections)}.\n\n",
            )
            + (
                self._tr(
                    f"你还引用了未提供的 Evidence ID：{', '.join(invalid_evidence_ids)}。只能使用当前证据目录中的编号；如果证据不够，请明确写缺少证据。\n\n",
                    f"You also cited Evidence IDs that were not provided: {', '.join(invalid_evidence_ids)}. Only use IDs from the current evidence catalog; if evidence is missing, say so explicitly.\n\n",
                )
                if invalid_evidence_ids
                else ""
            )
            + (
                self._tr(
                    "你还错误声称缺少待复核输入。实际上，本次请求已经提供了待复核发言、会议状态快照和证据目录。请直接完成复核；如果证据不足，只能指出具体哪条主张缺证据，不能再说“输入缺失”。\n\n",
                    "You also incorrectly claimed that the review input was missing. In fact, this request already included the statement under review, the meeting-state snapshot, and the evidence catalog. Complete the review directly; if evidence is insufficient, identify the unsupported claim instead of saying the input is missing.\n\n",
                )
                if missing_input_claim
                else ""
            )
            + self._tr("上一条输出：\n", "Previous output:\n")
            + f"{self._truncate_text(content, 800)}\n\n"
            + self._tr(
                "请严格按模板完整重写，不要省略任何区块。",
                "Rewrite the full answer strictly in the required template and do not omit any section.",
            )
        )
        repaired = self._chat(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=repair_prompt,
            max_tokens=max_tokens,
            attachments=attachments,
            max_continuations=max_continuations,
        )
        if not self._response_needs_repair(
            repaired,
            required_sections,
            min_chars=min_chars,
            allowed_evidence_ids=allowed_evidence_ids,
            reject_missing_review_input=reject_missing_review_input,
        ):
            return repaired
        if missing_input_claim:
            return repaired
        return content

    def _chat_with_repair(
        self,
        *,
        provider: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        required_sections: list[str],
        attachments: list[AttachmentPayload] | None = None,
        max_continuations: int = 0,
        min_chars: int = 96,
    ) -> str:
        return self._chat_with_sections(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            required_sections=required_sections,
            attachments=attachments,
            max_continuations=max_continuations,
            min_chars=min_chars,
        )

    def _generate_literature_review(self, user_request: str) -> DiscussionMessage:
        literature_provider = self._provider_for_action("review_literature", fallback=self.literature_provider, fallback_duty=LITERATURE_DUTY)
        assert literature_provider is not None
        snippets = select_literature_review_snippets(
            self.attachment_index,
            max_snippets=16,
            max_chars=15000,
        )
        cached_context = self._truncate_text(self.cached_pdf_reader_context, 8500)
        if not snippets and cached_context:
            return DiscussionMessage(
                speaker=literature_provider.name,
                role="assistant",
                content=f"## {self._tr('PDF Reader 缓存摘要', 'Cached PDF Reader Digest')}\n\n{cached_context}",
                round_index=0,
                model_name=literature_provider.model,
                duty=literature_provider.duty,
                stage="literature_review",
            )
        if not snippets:
            return DiscussionMessage(
                speaker=literature_provider.name,
                role="assistant",
                content=self._tr(
                    "[调用失败] 没有可用于文献综述的文本附件。",
                    "[Call Failed] No text attachment was available for literature review.",
                ),
                round_index=0,
                model_name=literature_provider.model,
                duty=literature_provider.duty,
                stage="literature_review",
            )

        packets = split_literature_review_packets(
            snippets,
            max_chars_per_packet=4200,
            max_packets=self.discussion_config.max_literature_review_batches,
        )
        packet_notes: list[str] = []
        required_packet_sections = [
            "## Packet Coverage",
            "## Problem And Setting",
            "## Method Details",
            "## Experiments And Results",
            "## Limits Or Missing Pieces",
            "## Evidence Anchors",
        ]
        for packet_index, packet in enumerate(packets, start=1):
            packet_context = render_literature_review_context(packet, max_chars=4200) or self._tr(
                "没有可用的文献分段。",
                "No literature packet was available.",
            )
            note_prompt = (
                f"{self._tr('用户任务', 'User task')}:\n{user_request}\n\n"
                f"{self._tr('文献分段', 'Packet')} {packet_index} {self._tr('共', 'of')} {len(packets)}\n\n"
                f"{packet_context}\n\n"
                + self._tr(
                    "请仔细阅读这个分段，只写分段级阅读笔记。保留固定 Markdown 标题，但正文内容必须使用与用户提问相同的主语言。",
                    "Read this packet carefully and write packet-level reading notes. Keep the fixed Markdown section titles, but write the section content in the same primary language as the user request.",
                )
            )
            notes = self._chat_with_repair(
                provider=literature_provider,
                system_prompt=LITERATURE_PACKET_NOTES_PROMPT,
                user_prompt=note_prompt,
                max_tokens=850,
                required_sections=required_packet_sections,
                max_continuations=1,
                min_chars=220,
            )
            packet_notes.append(f"### {self._tr('阅读分段', 'Reading Packet')} {packet_index}\n\n{notes}")

        synthesis_prompt = (
            f"{self._tr('用户任务', 'User task')}:\n{user_request}\n\n"
            + (
                self._tr("整篇论文主来源：\n", "Primary whole-paper source:\n")
                + f"{cached_context}\n\n"
                if cached_context
                else ""
            )
            + self._tr("分段覆盖摘要：\n", "Packet-level coverage summary:\n")
            + f"{summarize_snippet_coverage(snippets)}\n\n"
            + self._tr("分段阅读笔记：\n\n", "Packet-level reading notes:\n\n")
            + f"{self._truncate_text(chr(10).join(packet_notes), 12000)}\n\n"
            + self._tr(
                "请将这些材料整合为一份供后续专家组使用的文献综述。优先使用 PDF Reader 缓存摘要把握整篇结构、章节顺序，以及其中涉及的图示或流程图解释。保留固定 Markdown 标题，但正文内容必须使用与用户提问相同的主语言。",
                "Synthesize these sources into one literature review for the downstream expert team. Prefer the cached PDF reader digest for whole-paper structure, section order, and any figure or flowchart interpretation when it is available. Keep the fixed Markdown section titles, but write the section content in the same primary language as the user request.",
            )
        )
        content = self._chat_with_repair(
            provider=literature_provider,
            system_prompt=LITERATURE_REVIEW_PROMPT,
            user_prompt=synthesis_prompt,
            max_tokens=1700,
            required_sections=[
                "## Coverage",
                "## Research Scope",
                "## Method And Evidence",
                "## Main Findings",
                "## Limitations And Open Questions",
                "## What The Expert Team Can Reuse",
            ],
            max_continuations=2,
            min_chars=320,
        )
        return DiscussionMessage(
            speaker=literature_provider.name,
            role="assistant",
            content=content,
            round_index=0,
            model_name=literature_provider.model,
            duty=literature_provider.duty,
            stage="literature_review",
        )

    def _run_consensus_followups(
        self,
        *,
        result: DiscussionResult,
        state: DiscussionState,
        user_request: str,
        attachments: list[AttachmentPayload],
        assignments_text: str,
        team_roster: str,
        literature_review_text: str,
        successful_messages: list[DiscussionMessage],
        log_messages: list[DiscussionMessage],
        on_message: Callable[[DiscussionMessage], None] | None,
        on_status: Callable[[str], None] | None,
        should_cancel: Callable[[], bool] | None,
        completed_rounds: int,
    ) -> int:
        topic_attempts: dict[str, int] = {}

        for pass_index in range(1, self.discussion_config.max_followup_attempts + 1):
            followups = self._build_followup_workpackages(state, topic_attempts)
            if not followups:
                return completed_rounds

            if on_status is not None:
                on_status(self._tr(f"主持人正在组织第 {pass_index} 轮未决问题深挖", f"Host is organizing unresolved-issue pass {pass_index}"))

            for workpackage in followups:
                if should_cancel is not None and should_cancel():
                    return completed_rounds

                topic_key = self._topic_key(workpackage.display_text)
                topic_attempts[topic_key] = topic_attempts.get(topic_key, 0) + 1
                owner, reviewer = self._resolve_followup_pair(workpackage.display_text)
                self.state_manager.ensure_task(
                    state,
                    task_id=self._task_id_for_workpackage(workpackage),
                    title=workpackage.title,
                    description=workpackage.description,
                    owner_name=owner.name,
                    reviewer_name=reviewer.name if reviewer is not None else "",
                    round_index=workpackage.index,
                    source_kind="followup",
                )
                self.state_manager.begin_task(
                    state,
                    task_id=self._task_id_for_workpackage(workpackage),
                    stage_label=f"Follow-up {pass_index}: {workpackage.title}",
                    question=workpackage.display_text,
                    round_index=workpackage.index,
                )

                deep_snippets = self._select_relevant_snippets(
                    f"{user_request}\n{workpackage.display_text}\n{owner.specialty}\n{reviewer.specialty if reviewer is not None else ''}",
                    owner,
                    max_snippets=5,
                    max_chars=2800,
                )
                deep_message = self._run_primary_assignment(
                    provider=owner,
                    user_request=user_request,
                    assignments_text=assignments_text,
                    team_roster=team_roster,
                    literature_review_text=literature_review_text,
                    workpackage=workpackage,
                    state=state,
                    relevant_snippets=deep_snippets,
                    attachments=attachments,
                )
                self._push_message(result, successful_messages, on_message, deep_message)

                deep_log_message, deep_entry = self._build_log_entry(
                    user_request=user_request,
                    state=state,
                    source_message=deep_message,
                    workpackage_title=workpackage.display_text,
                    index=workpackage.index,
                    relevant_snippets=deep_snippets,
                    fallback_text=self._fallback_log_line(deep_message, workpackage.display_text),
                )
                self._record_log(result, successful_messages, on_message, log_messages, state, deep_log_message, deep_entry)

                review_message = None
                if reviewer is not None:
                    review_snippets = self._select_relevant_snippets(
                        f"{user_request}\n{workpackage.display_text}\n{reviewer.specialty}\n{deep_message.content}",
                        reviewer,
                        max_snippets=5,
                        max_chars=2600,
                    )
                    review_snippets = self._augment_review_snippets(
                        review_snippets,
                        state=state,
                        workpackage=workpackage,
                        previous_message=deep_message,
                        provider=reviewer,
                    )
                    review_message = self._run_review_assignment(
                        provider=reviewer,
                        user_request=user_request,
                        assignments_text=assignments_text,
                        team_roster=team_roster,
                        literature_review_text=literature_review_text,
                        workpackage=workpackage,
                        previous_message=deep_message,
                        state=state,
                        relevant_snippets=review_snippets,
                    )
                    self._push_message(result, successful_messages, on_message, review_message)

                    review_log_message, review_entry = self._build_log_entry(
                        user_request=user_request,
                        state=state,
                        source_message=review_message,
                        workpackage_title=workpackage.display_text,
                        index=workpackage.index,
                        relevant_snippets=review_snippets,
                        fallback_text=self._fallback_log_line(review_message, workpackage.display_text),
                    )
                    self._record_log(result, successful_messages, on_message, log_messages, state, review_log_message, review_entry)

                if self.host_provider is not None:
                    resolution_message = self._run_host_resolution(
                        user_request=user_request,
                        assignments_text=assignments_text,
                        team_roster=team_roster,
                        workpackage=workpackage,
                        state=state,
                        primary_message=deep_message,
                        review_message=review_message,
                    )
                    self._push_message(result, successful_messages, on_message, resolution_message)

                    resolution_log_message, resolution_entry = self._build_log_entry(
                        user_request=user_request,
                        state=state,
                        source_message=resolution_message,
                        workpackage_title=workpackage.display_text,
                        index=workpackage.index,
                        relevant_snippets=deep_snippets,
                        fallback_text=self._fallback_log_line(resolution_message, workpackage.display_text),
                    )
                    self._record_log(result, successful_messages, on_message, log_messages, state, resolution_log_message, resolution_entry)
                    self._apply_followup_resolution(state, workpackage.display_text, resolution_message.content)

                completed_rounds += 1
                self.state_manager.complete_task(
                    state,
                    task_id=self._task_id_for_workpackage(workpackage),
                    notes=self._truncate_text(deep_message.content, 240),
                )
                checkpoint = self._maybe_create_checkpoint(
                    state,
                    label=workpackage.title,
                    workpackage_index=workpackage.index,
                    completed_rounds=completed_rounds,
                )
                if checkpoint is not None and on_status is not None:
                    on_status(
                        self._tr(
                            f"未决问题 {self._checkpoint_label(checkpoint.checkpoint_id)} 已更新：{checkpoint.label}",
                            f"Unresolved-issue {self._checkpoint_label(checkpoint.checkpoint_id)} updated: {checkpoint.label}",
                        )
                    )

            if not self._has_remaining_followups(state, topic_attempts):
                return completed_rounds

        return completed_rounds

    def _run_host_resolution(
        self,
        *,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        workpackage: WorkPackage,
        state: DiscussionState,
        primary_message: DiscussionMessage,
        review_message: DiscussionMessage | None,
    ) -> DiscussionMessage:
        resolution_provider = self._provider_for_action("resolve_followup", fallback=self.host_provider, fallback_duty=HOST_DUTY)
        assert resolution_provider is not None
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前未决问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='host')}\n\n"
            f"深挖发言 1：\n[{primary_message.speaker}]\n{self._truncate_text(primary_message.content, 1000)}\n\n"
            f"深挖发言 2：\n[{review_message.speaker if review_message else '无复核'}]\n{self._truncate_text(review_message.content, 900) if review_message else '无复核发言'}\n\n"
            "请判断该未决问题现在是否达到阶段共识。"
        )
        content = self._chat_with_sections(
            provider=resolution_provider,
            system_prompt=FOLLOWUP_HOST_PROMPT,
            user_prompt=prompt,
            max_tokens=720,
            max_continuations=1,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
            min_chars=120,
        )
        return DiscussionMessage(
            speaker=resolution_provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=resolution_provider.model,
            duty=resolution_provider.duty,
            stage="followup_resolution",
        )

    def _update_state_from_assignment(self, state: DiscussionState, assignment_text: str, workpackages: list[WorkPackage]) -> None:
        state.assignment_summary = self._truncate_text(assignment_text, 1000)
        extracted_goal = self._extract_named_value(assignment_text, "研究目标") or self._extract_named_value(assignment_text, "Research Goal")
        extracted_domain = self._extract_named_value(assignment_text, "领域判断") or self._extract_named_value(assignment_text, "Domain")
        if extracted_goal:
            state.goal = extracted_goal
        if extracted_domain:
            state.domain = extracted_domain
        state.action_items = []
        for workpackage in workpackages:
            state.action_items.append(
                self._tr(
                    f"任务 {workpackage.index}: {workpackage.title} -> 主责 {workpackage.owner_name or '待定'} / 复核 {workpackage.reviewer_name or '待定'}",
                    f"Task {workpackage.index}: {workpackage.title} -> Owner {workpackage.owner_name or 'TBD'} / Reviewer {workpackage.reviewer_name or 'TBD'}",
                )
            )
        state.current_question = workpackages[0].display_text if workpackages else state.current_question
        self.state_manager.sync_workflow_tasks(
            state,
            tasks=[
                WorkflowTask(
                    task_id=self._task_id_for_workpackage(workpackage),
                    title=workpackage.title,
                    description=workpackage.description,
                    owner_name=workpackage.owner_name,
                    reviewer_name=workpackage.reviewer_name,
                    round_index=workpackage.index,
                    source_kind="assignment",
                )
                for workpackage in workpackages
            ],
            replace_for_source_kind="assignment",
        )

    def _update_state_from_coordination(self, state: DiscussionState, coordination_text: str) -> None:
        state.coordination_summary = self._truncate_text(coordination_text, 800)

    def _record_log(
        self,
        result: DiscussionResult,
        successful_messages: list[DiscussionMessage],
        on_message: Callable[[DiscussionMessage], None] | None,
        log_messages: list[DiscussionMessage],
        state: DiscussionState,
        log_message: DiscussionMessage,
        entry: StructuredLogEntry,
    ) -> None:
        self._apply_log_entry(state, entry)
        self._push_message(result, successful_messages, on_message, log_message)
        log_messages.append(log_message)

    def _apply_log_entry(self, state: DiscussionState, entry: StructuredLogEntry) -> None:
        self.state_manager.apply_log_entry(
            state,
            entry=entry,
            max_history_items=self.context_config.max_history_items,
            max_log_entries=self.context_config.max_log_entries,
            max_evidence_cards=self.context_config.max_evidence_cards,
        )

    def _create_checkpoint(self, state: DiscussionState, *, label: str, workpackage_index: int) -> Checkpoint:
        return self.state_manager.create_checkpoint(
            state,
            label=label,
            workpackage_index=workpackage_index,
            max_checkpoints=self.discussion_config.max_checkpoints,
        )

    def _maybe_create_checkpoint(
        self,
        state: DiscussionState,
        *,
        label: str,
        workpackage_index: int,
        completed_rounds: int,
    ) -> Checkpoint | None:
        frequency = self.discussion_config.checkpoint_every_n_rounds
        if frequency <= 1 or completed_rounds % frequency == 0:
            return self._create_checkpoint(state, label=label, workpackage_index=workpackage_index)
        return None

    def _build_state_snapshot(
        self,
        state: DiscussionState,
        *,
        workpackage: WorkPackage | None = None,
        workpackage_index: int | None = None,
        mode: str,
    ) -> str:
        current_index = workpackage.index if workpackage is not None else workpackage_index
        checkpoint = state.checkpoints[-1] if state.checkpoints else None
        relevant_entries = self._select_recent_entries(state, current_index)
        summary_slots = set(self.context_config.summary_slots)
        rule_limit = min(5, self.context_config.max_history_items)
        history_limit = min(6, self.context_config.max_history_items)

        parts = [
            f"{self._tr('主题', 'Topic')}: {state.topic or self._tr('未设定', 'Not set')}",
            f"{self._tr('用户问题', 'User Question')}: {state.user_question or self._tr('未记录', 'Not recorded')}",
            f"{self._tr('领域', 'Domain')}: {state.domain or self._tr('待讨论判断', 'To be determined during discussion')}",
            f"{self._tr('研究目标', 'Research Goal')}: {state.goal or self._tr('未提炼', 'Not distilled yet')}",
            f"{self._tr('当前阶段', 'Current Stage')}: {state.current_stage or self._tr('未开始', 'Not started')}",
            f"{self._tr('当前轮次', 'Current Round')}: {state.current_round or 0}",
            f"{self._tr('当前子问题', 'Current Workpackage')}: {workpackage.display_text if workpackage is not None else state.current_question or self._tr('未设定', 'Not set')}",
            f"{self._tr('已上传资料', 'Uploaded Sources')}: {', '.join(state.uploaded_sources) if state.uploaded_sources else self._tr('无', 'None')}",
            f"{self._tr('会议规则', 'Meeting Rules')}:",
            self._render_indexed_lines(state.rules, prefix="R", limit=rule_limit),
        ]

        if state.assignment_summary:
            parts.append(f"{self._tr('派工摘要', 'Assignment Summary')}:\n{self._truncate_text(state.assignment_summary, 320)}")
        if mode in {"host", "report", "minutes"} and state.coordination_summary:
            parts.append(f"{self._tr('主持安排摘要', 'Coordination Summary')}:\n{self._truncate_text(state.coordination_summary, 260)}")
        if mode in {"report", "minutes"} and state.summary:
            parts.append(f"{self._tr('当前总结', 'Current Summary')}:\n{self._truncate_text(state.summary, 320)}")
        if checkpoint is not None:
            parts.append(f"{self._tr('最近检查点', 'Latest Checkpoint')} {checkpoint.checkpoint_id}: {checkpoint.summary}")

        if "consensus" in summary_slots:
            parts.append(f"{self._tr('稳定共识', 'Stable Consensus')}:")
            parts.append(self._render_indexed_lines(state.stable_consensus, prefix="K", limit=history_limit))
        if "conflicts" in summary_slots:
            parts.append(f"{self._tr('当前争议', 'Active Conflicts')}:")
            parts.append(self._render_indexed_lines(state.conflicts, prefix="C", limit=history_limit))
        if "open_questions" in summary_slots:
            parts.append(f"{self._tr('未决问题', 'Open Questions')}:")
            parts.append(self._render_indexed_lines(state.open_questions, prefix="Q", limit=history_limit))

        if mode in {"expert", "reviewer", "logger", "report", "minutes", "host"} and "recent_updates" in summary_slots:
            parts.append(f"{self._tr('近期增量', 'Recent Updates')}:")
            parts.append(self._render_recent_entries(relevant_entries))

        if mode in {"report", "minutes", "host"} and "action_items" in summary_slots:
            parts.append(f"{self._tr('行动项', 'Action Items')}:")
            parts.append(self._render_indexed_lines(state.action_items, prefix="A", limit=history_limit))

        if mode in {"host", "report", "minutes"} and state.workflow_tasks:
            parts.append(f"{self._tr('任务状态', 'Workflow Tasks')}:")
            parts.append(self._render_workflow_tasks(state.workflow_tasks[-max(4, history_limit):]))

        if mode in {"report", "minutes"} and state.literature_library:
            parts.append(f"{self._tr('论文库', 'Literature Library')}:")
            parts.append(self._build_literature_library_snapshot(state))

        if mode in {"report", "minutes"} and state.experiment_runs:
            parts.append(f"{self._tr('实验运行', 'Experiment Runs')}:")
            parts.append(self._build_experiment_run_snapshot(state))

        if mode in {"report", "minutes"} and state.approval_records:
            parts.append(f"{self._tr('授权记录', 'Approval Records')}:")
            parts.append(self._build_approval_snapshot(state))

        if mode in {"report", "minutes"} and state.generated_artifacts:
            parts.append(f"{self._tr('已生成产物', 'Generated Artifacts')}:")
            parts.append(self._render_generated_artifacts(state))

        return "\n\n".join(part for part in parts if part)

    def _parse_structured_log_entry(
        self,
        *,
        raw_text: str,
        source_message: DiscussionMessage,
        workpackage_title: str,
        index: int,
        relevant_snippets: list[AttachmentSnippet],
        fallback_text: str,
    ) -> StructuredLogEntry:
        if raw_text.startswith("[Call Failed]"):
            return self._fallback_structured_log_entry(
                source_message=source_message,
                workpackage_title=workpackage_title,
                index=index,
                relevant_snippets=relevant_snippets,
                fallback_text=fallback_text,
            )

        payload = self._extract_json_object(raw_text)
        if payload is None:
            return self._fallback_structured_log_entry(
                source_message=source_message,
                workpackage_title=workpackage_title,
                index=index,
                relevant_snippets=relevant_snippets,
                fallback_text=fallback_text,
            )

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return self._fallback_structured_log_entry(
                source_message=source_message,
                workpackage_title=workpackage_title,
                index=index,
                relevant_snippets=relevant_snippets,
                fallback_text=fallback_text,
            )

        evidence_lookup = {snippet.evidence_id: snippet for snippet in relevant_snippets}
        evidence_cards: list[EvidenceCard] = []
        for evidence_id in self._coerce_str_list(data.get("evidence_ids")):
            snippet = evidence_lookup.get(evidence_id)
            if snippet is None:
                continue
            evidence_cards.append(
                EvidenceCard(
                    evidence_id=snippet.evidence_id,
                    summary=self._summarize_snippet(snippet),
                    source=self._snippet_source(snippet),
                    display_label=self._snippet_display_label(snippet),
                    attachment_name=snippet.attachment_name,
                    workpackage_index=index,
                )
            )
        if not evidence_cards and relevant_snippets:
            for snippet in relevant_snippets[:2]:
                evidence_cards.append(
                    EvidenceCard(
                        evidence_id=snippet.evidence_id,
                        summary=self._summarize_snippet(snippet),
                        source=f"{snippet.attachment_name}#chunk{snippet.chunk_index}",
                        attachment_name=snippet.attachment_name,
                        workpackage_index=index,
                    )
                )

        return StructuredLogEntry(
            workpackage_index=index,
            workpackage_title=workpackage_title,
            speaker=source_message.speaker,
            stage=source_message.stage,
            headline=str(data.get("headline") or f"{source_message.speaker} 更新").strip(),
            summary=str(data.get("summary") or fallback_text).strip(),
            consensus_add=self._coerce_str_list(data.get("consensus_add")),
            conflicts_add=self._coerce_str_list(data.get("conflicts_add")),
            resolved_conflicts=self._coerce_str_list(data.get("resolved_conflicts")),
            open_questions_add=self._coerce_str_list(data.get("open_questions_add")),
            resolved_questions=self._coerce_str_list(data.get("resolved_questions")),
            rejected_add=self._coerce_str_list(data.get("rejected_add")),
            action_items_add=self._coerce_str_list(data.get("action_items_add")),
            evidence_add=evidence_cards,
            redundant=bool(data.get("redundant", False)),
        )

    def _fallback_structured_log_entry(
        self,
        *,
        source_message: DiscussionMessage | None,
        workpackage_title: str,
        index: int,
        relevant_snippets: list[AttachmentSnippet],
        fallback_text: str,
    ) -> StructuredLogEntry:
        if source_message is None:
            return StructuredLogEntry(
                workpackage_index=index,
                workpackage_title=workpackage_title,
                speaker=self.report_provider.name if self.report_provider else self._tr("统稿日志", "Reporter Log"),
                stage="log",
                headline=self._tr(f"任务 {index} 初始化", f"Task {index} initialized"),
                summary=fallback_text,
                action_items_add=[self._tr(f"启动任务 {index}：{workpackage_title}", f"Start task {index}: {workpackage_title}")],
            )

        summary = self._collapse_whitespace(self._truncate_text(source_message.content, 180))
        sections = self._extract_structured_sections(source_message.content)
        candidate_lines = self._candidate_lines(source_message.content)

        consensus = [
            *sections.get("judgment", [])[:1],
            *sections.get("verdict", [])[:1],
            *sections.get("reasons", [])[:2],
            *sections.get("support", [])[:2],
            *sections.get("synthesis", [])[:2],
        ]
        conflicts = [
            *sections.get("risk", [])[:2],
            *sections.get("gap", [])[:2],
            *sections.get("open gap", [])[:2],
            *sections.get("corrections", [])[:2],
            *sections.get("residual risk", [])[:2],
            *sections.get("evidence check", [])[:1],
        ]
        questions = list(sections.get("need more work", [])[:2])
        actions = [
            *sections.get("handoff", [])[:2],
            *sections.get("coordination decision", [])[:2],
        ]

        if not consensus and not conflicts and not questions:
            for line in candidate_lines:
                if "?" in line or "?" in line:
                    questions.append(line)
                elif any(token in line for token in ["\u98ce\u9669", "\u4e0d\u8db3", "\u95ee\u9898", "\u4e89\u8bae", "\u5e7b\u89c9", "\u4e0d\u786e\u5b9a", "\u9700\u8981", "\u672a\u8fbe\u6210", "\u7f3a\u5c11", "\u7f3a\u4e4f", "risk", "insufficient", "issue", "conflict", "need"]):
                    conflicts.append(line)
                else:
                    consensus.append(line)

        if self._is_failed_message(source_message):
            conflicts = [self._tr(
                f"{source_message.speaker} 在任务 {index} 的调用失败，需要重试或替换。",
                f"{source_message.speaker} failed while working on task {index}; retry or model replacement is needed.",
            )]
            questions = [self._tr(
                f"是否需要为任务 {index} 更换模型或缩短上下文。",
                f"Should task {index} switch models or shorten the context window?",
            )]
            consensus = []
            actions = [self._tr(f"重试任务 {index}：{workpackage_title}", f"Retry task {index}: {workpackage_title}")]

        evidence_cards = [
            EvidenceCard(
                evidence_id=snippet.evidence_id,
                summary=self._summarize_snippet(snippet),
                source=self._snippet_source(snippet),
                display_label=self._snippet_display_label(snippet),
                attachment_name=snippet.attachment_name,
                workpackage_index=index,
            )
            for snippet in relevant_snippets[:2]
        ]

        return StructuredLogEntry(
            workpackage_index=index,
            workpackage_title=workpackage_title,
            speaker=source_message.speaker,
            stage=source_message.stage,
            headline=f"{source_message.speaker} {self._stage_label(source_message.stage)}",
            summary=summary or fallback_text,
            consensus_add=consensus[:2],
            conflicts_add=conflicts[:2],
            open_questions_add=questions[:2],
            action_items_add=(actions[:2] if actions else [self._tr(f"继续推进任务 {index}：{workpackage_title}", f"Continue task {index}: {workpackage_title}")]),
            evidence_add=evidence_cards,
            redundant=False,
        )

    def _log_message_from_entry(self, entry: StructuredLogEntry) -> DiscussionMessage:
        lines = [f"- {self._tr('摘要', 'Summary')}: {entry.summary}"]
        if entry.consensus_add:
            lines.append(f"- {self._tr('共识增量', 'Consensus delta')}: " + self._tr("；", "; ").join(entry.consensus_add[:3]))
        if entry.conflicts_add:
            lines.append(f"- {self._tr('争议增量', 'Conflict delta')}: " + self._tr("；", "; ").join(entry.conflicts_add[:3]))
        if entry.open_questions_add:
            lines.append(f"- {self._tr('未决问题', 'Open questions')}: " + self._tr("；", "; ").join(entry.open_questions_add[:3]))
        if entry.action_items_add:
            lines.append(f"- {self._tr('下一步', 'Next step')}: " + self._tr("；", "; ").join(entry.action_items_add[:3]))
        if entry.evidence_add:
            lines.append(
                f"- {self._tr('证据引用', 'Evidence references')}: " + self._tr("；", "; ").join(
                    card.display_label or self._fallback_evidence_label(card.evidence_id, card.source)
                    for card in entry.evidence_add
                )
            )
        if entry.redundant:
            lines.append(self._tr("- 判定：本条发言新增信息有限。", "- Verdict: this message added limited new information."))
        return DiscussionMessage(
            speaker=self.report_provider.name if self.report_provider else self._tr("统稿日志", "Reporter Log"),
            role="assistant",
            content="\n".join(lines),
            round_index=entry.workpackage_index,
            model_name=self.report_provider.model if self.report_provider else "",
            duty=REPORT_DUTY,
            stage="log",
        )

    def _select_relevant_snippets(
        self,
        query: str,
        provider: ProviderConfig | None,
        *,
        max_snippets: int | None = None,
        max_chars: int | None = None,
    ) -> list[AttachmentSnippet]:
        if provider is None:
            provider_budget = 2200
            snippet_limit = 4
        else:
            provider_budget = self._evidence_budget(provider)
            snippet_limit = 3 if self._provider_key(provider) == "qwen" else 4
        return select_attachment_snippets(
            self.attachment_index,
            query,
            max_chars=max_chars or provider_budget,
            max_snippets=max_snippets or snippet_limit,
        )

    def _augment_review_snippets(
        self,
        snippets: list[AttachmentSnippet],
        *,
        state: DiscussionState,
        workpackage: WorkPackage,
        previous_message: DiscussionMessage,
        provider: ProviderConfig | None,
    ) -> list[AttachmentSnippet]:
        provider_budget = self._evidence_budget(provider)
        snippet_limit = 3 if provider is not None and self._provider_key(provider) == "qwen" else 4
        merged = self._merge_snippet_lists(snippets)
        if len(merged) >= min(2, snippet_limit):
            return self._cap_snippets(merged, max_snippets=snippet_limit, max_chars=provider_budget)

        attachment_lookup = {snippet.evidence_id: snippet for snippet in self.attachment_index}
        candidate_ids: list[str] = []
        candidate_ids.extend(
            card.evidence_id
            for card in reversed(state.evidence_cards)
            if card.workpackage_index == workpackage.index
        )
        candidate_ids.extend(self._extract_evidence_ids(previous_message.content))
        candidate_ids.extend(card.evidence_id for card in reversed(state.evidence_cards[-self.context_config.max_evidence_cards :]))
        fallback_snippets = [attachment_lookup[evidence_id] for evidence_id in candidate_ids if evidence_id in attachment_lookup]
        merged = self._merge_snippet_lists(merged, fallback_snippets)

        if len(merged) < min(2, snippet_limit):
            fallback_query = "\n".join(
                part
                for part in [
                    state.user_question,
                    workpackage.display_text,
                    previous_message.content[:480],
                    state.current_question,
                ]
                if part
            )
            merged = self._merge_snippet_lists(
                merged,
                self._select_relevant_snippets(
                    fallback_query,
                    provider,
                    max_snippets=snippet_limit,
                    max_chars=provider_budget,
                ),
            )

        if not merged and self.attachment_index:
            merged = self._merge_snippet_lists(merged, self.attachment_index[:snippet_limit])
        return self._cap_snippets(merged, max_snippets=snippet_limit, max_chars=provider_budget)

    def _merge_snippet_lists(self, *groups: list[AttachmentSnippet]) -> list[AttachmentSnippet]:
        merged: list[AttachmentSnippet] = []
        seen: set[tuple[str, str, int]] = set()
        for group in groups:
            for snippet in group:
                key = (snippet.evidence_id, snippet.attachment_name, snippet.chunk_index)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(snippet)
        return merged

    def _cap_snippets(
        self,
        snippets: list[AttachmentSnippet],
        *,
        max_snippets: int,
        max_chars: int,
    ) -> list[AttachmentSnippet]:
        selected: list[AttachmentSnippet] = []
        used = 0
        for snippet in snippets:
            if len(selected) >= max_snippets:
                break
            snippet_cost = len(snippet.content) + len(snippet.attachment_name) + 32
            if selected and used + snippet_cost > max_chars:
                continue
            selected.append(snippet)
            used += snippet_cost
        return selected or snippets[:max_snippets]

    def _select_reader_references(
        self,
        query: str,
        provider: ProviderConfig | None,
        *,
        max_items: int | None = None,
        max_chars: int = 1800,
    ) -> list[ReaderReference]:
        if not self.reader_references:
            return []
        if provider is None:
            item_limit = 4
        else:
            item_limit = 3 if self._provider_key(provider) == "qwen" else 5
        return select_pdf_reader_references(
            self.reader_references,
            query,
            max_chars=max_chars,
            max_items=max_items or item_limit,
        )

    def _build_reader_context(
        self,
        query: str,
        provider: ProviderConfig | None,
        *,
        max_chars: int = 1800,
        max_items: int | None = None,
    ) -> str:
        references = self._select_reader_references(
            query,
            provider,
            max_chars=max_chars,
            max_items=max_items,
        )
        return render_pdf_reader_references(references, max_chars=max_chars) or "No indexed PDF reader references were retrieved."

    def _build_reader_attachments(
        self,
        query: str,
        provider: ProviderConfig | None,
        *,
        max_items: int = 2,
    ) -> list[AttachmentPayload]:
        if provider is None or not provider.supports_vision:
            return []
        references = self._select_reader_references(query, provider, max_items=max_items, max_chars=1500)
        return build_reader_reference_attachments(references, max_images=max_items)

    def _merge_chat_attachments(
        self,
        base_attachments: list[AttachmentPayload] | None,
        extra_attachments: list[AttachmentPayload] | None,
    ) -> list[AttachmentPayload]:
        merged: list[AttachmentPayload] = []
        seen: set[str] = set()
        for attachment in (base_attachments or []) + (extra_attachments or []):
            key = str(attachment.path)
            if key in seen:
                continue
            seen.add(key)
            merged.append(attachment)
        return merged

    def _build_team_roster(self) -> str:
        if not self.team.members:
            return self._tr("- 暂无有效角色", "- No active roles")
        template_header = self._tr(
            f"团队模板：{self.team.template.name}",
            f"Team template: {self.team.template.name}",
        )
        roster_lines = [template_header]
        roster_lines.extend(member.roster_line() for member in self.team.members)
        return "\n".join(roster_lines)

    def _build_literature_context(self, literature_review_text: str, max_chars: int) -> str:
        parts: list[str] = []
        if self.cached_pdf_reader_context.strip():
            parts.append(f"[{self._tr('PDF Reader 缓存摘要', 'Cached PDF Reader Digest')}]\n{self.cached_pdf_reader_context.strip()}")
        if literature_review_text.strip():
            parts.append(literature_review_text.strip())
        if not parts:
            return self._tr(
                "尚未生成文献综述，且没有可用的 PDF Reader 缓存摘要。",
                "Literature review was not generated and no cached PDF reader digest is available.",
            )
        return self._truncate_text("\n\n".join(parts), max_chars)

    def _build_literature_library_snapshot(self, state: DiscussionState) -> str:
        if not state.literature_library:
            return "- No discovered literature records."
        lines = []
        for index, paper in enumerate(state.literature_library[-8:], start=max(1, len(state.literature_library) - 7)):
            authors = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors += ", et al."
            lines.append(
                f"- Paper {index}: {paper.title} | {paper.paper_id} | authors={authors or 'Unknown'} | "
                f"local_pdf={paper.local_pdf_path or 'not downloaded'} | bibkey={paper.bibtex_key or 'n/a'}"
            )
        return "\n".join(lines)

    def _build_experiment_run_snapshot(self, state: DiscussionState) -> str:
        if not state.experiment_runs:
            return "- No experiment runs recorded."
        return "\n".join(
            f"- {run.run_id} | status={run.status} | script={Path(run.script_path).name} | "
            f"compile_rc={run.compile_returncode} | run_rc={run.runtime_returncode} | log={Path(run.log_path).name if run.log_path else 'n/a'}"
            for run in state.experiment_runs[-8:]
        )

    def _build_approval_snapshot(self, state: DiscussionState) -> str:
        if not state.approval_records:
            return "- No approval records."
        return "\n".join(
            f"- {approval.approval_type} | scope={approval.scope} | granted={'yes' if approval.granted else 'no'} | {approval.created_at}"
            for approval in state.approval_records[-8:]
        )

    def _build_checkpoint_timeline(self, state: DiscussionState) -> str:
        if not state.checkpoints:
            return self._tr("- 暂无检查点。", "- No checkpoints yet.")
        return "\n".join(
            f"- {self._checkpoint_label(checkpoint.checkpoint_id)} | {self._tr('轮次', 'Round')} {checkpoint.round_index} | {checkpoint.label} | {checkpoint.summary}"
            for checkpoint in state.checkpoints[-self.discussion_config.max_checkpoints:]
        )

    def _build_evidence_ledger(self, state: DiscussionState) -> str:
        if not state.evidence_cards:
            return self._tr("- 暂无已固化证据引用。", "- No evidence references have been consolidated yet.")
        return "\n".join(
            f"- {card.display_label or self._fallback_evidence_label(card.evidence_id, card.source)} | {self._tr('轮次', 'Round')} {card.workpackage_index} | {card.summary}"
            for card in state.evidence_cards[-self.context_config.max_evidence_cards:]
        )

    def _checkpoint_label(self, checkpoint_id: str) -> str:
        number = ''.join(char for char in checkpoint_id if char.isdigit()) or checkpoint_id
        return self._tr(f"检查点 {number}", f"Checkpoint {number}")

    def _state_item_label(self, prefix: str, index: int) -> str:
        mapping = {
            "R": self._tr("规则", "Rule"),
            "K": self._tr("共识", "Consensus"),
            "C": self._tr("争议", "Conflict"),
            "Q": self._tr("未决问题", "Open Question"),
            "A": self._tr("行动项", "Action Item"),
        }
        return f"{mapping.get(prefix, prefix)} {index}"

    def _snippet_source(self, snippet: AttachmentSnippet) -> str:
        parts = [snippet.attachment_name]
        if snippet.page_hint is not None:
            parts.append(self._tr(f"第 {snippet.page_hint} 页", f"page {snippet.page_hint}"))
        parts.append(self._tr(f"分段 {snippet.chunk_index}", f"chunk {snippet.chunk_index}"))
        return " | ".join(parts)

    def _snippet_display_label(self, snippet: AttachmentSnippet) -> str:
        number = ''.join(char for char in snippet.evidence_id if char.isdigit()) or snippet.evidence_id
        parts = [snippet.attachment_name]
        if snippet.page_hint is not None:
            parts.append(self._tr(f"第 {snippet.page_hint} 页", f"page {snippet.page_hint}"))
        parts.append(self._tr(f"分段 {snippet.chunk_index}", f"chunk {snippet.chunk_index}"))
        if self.output_language == "zh":
            return f"证据 {number}（{'\uff0c'.join(parts)}）"
        return f"Evidence {number} ({', '.join(parts)})"

    def _fallback_evidence_label(self, evidence_id: str, source: str) -> str:
        number = ''.join(char for char in evidence_id if char.isdigit()) or evidence_id
        cleaned_source = source.replace("|", self._tr("，", ",")).strip()
        if self.output_language == "zh":
            return f"证据 {number}（{cleaned_source}）" if cleaned_source else f"证据 {number}"
        return f"Evidence {number} ({cleaned_source})" if cleaned_source else f"Evidence {number}"


    def _push_message(
        self,
        result: DiscussionResult,
        successful_messages: list[DiscussionMessage],
        on_message: Callable[[DiscussionMessage], None] | None,
        message: DiscussionMessage,
    ) -> None:
        result.messages.append(message)
        if not self._is_failed_message(message):
            successful_messages.append(message)
        if on_message is not None:
            on_message(message)

    def _resolve_owner(self, assigned_name: str, fallback_index: int) -> ProviderConfig:
        provider = self._match_provider_by_name(assigned_name)
        if provider is not None:
            return provider
        if self.expert_providers:
            return self.expert_providers[fallback_index % len(self.expert_providers)]
        fallback = self.report_provider or self.host_provider or self.literature_provider or self.lead_provider
        if fallback is None:
            raise RuntimeError("没有可用的执行角色。")
        return fallback

    def _resolve_reviewer(self, assigned_name: str, owner: ProviderConfig) -> ProviderConfig | None:
        if not self.discussion_config.enable_reviewer_role:
            return None
        provider = self._match_provider_by_name(assigned_name)
        if provider is not None and provider is not owner:
            return provider
        if owner.duty == EXPERT_DUTY and len(self.expert_providers) >= 2:
            owner_index = self.expert_providers.index(owner)
            return self.expert_providers[(owner_index + 1) % len(self.expert_providers)]
        for candidate in [self.host_provider, self.report_provider, self.literature_provider, *self.expert_providers]:
            if candidate is not None and candidate is not owner:
                return candidate
        return None

    def _resolve_followup_pair(self, topic: str) -> tuple[ProviderConfig, ProviderConfig | None]:
        lowered = topic.lower()
        if (
            any(token in topic for token in ["证据", "文献", "引用", "Evidence ID", "输入缺失"])
            or any(token in lowered for token in ["evidence", "literature", "citation", "input missing"])
        ):
            owner = self.literature_provider or self.report_provider or self.host_provider
            reviewer = next((candidate for candidate in [self.host_provider, self.lead_provider, *self.expert_providers] if candidate is not None and candidate is not owner), None)
            if owner is not None:
                return owner, reviewer
        if (
            any(token in topic for token in ["偏离", "偏题", "流程", "协调", "复核阻塞"])
            or any(token in lowered for token in ["off-topic", "scope", "workflow", "coordination", "reviewer blocked"])
        ):
            owner = self.host_provider or self.lead_provider or self.report_provider
            reviewer = next((candidate for candidate in [self.report_provider, self.literature_provider, *self.expert_providers] if candidate is not None and candidate is not owner), None)
            if owner is not None:
                return owner, reviewer
        ranked = sorted(
            self.expert_providers,
            key=lambda provider: self._provider_relevance_score(provider, topic),
            reverse=True,
        )
        if not ranked:
            fallback = self.report_provider or self.host_provider or self.literature_provider or self.lead_provider
            if fallback is None:
                raise RuntimeError("没有可用于深挖未决问题的角色。")
            return fallback, None
        owner = ranked[0]
        if not self.discussion_config.enable_reviewer_role:
            return owner, None
        reviewer = next((provider for provider in ranked[1:] if provider is not owner), None)
        if reviewer is None:
            reviewer = self._resolve_reviewer("", owner)
        return owner, reviewer

    def _match_provider_by_name(self, assigned_name: str) -> ProviderConfig | None:
        normalized = assigned_name.strip()
        if not normalized:
            return None
        direct = self.providers_by_name.get(normalized)
        if direct is not None:
            return direct

        lowered = normalized.lower()
        for provider in self.providers_by_name.values():
            provider_name = provider.name.lower()
            if provider_name in lowered or lowered in provider_name:
                return provider
        return None

    def _provider_relevance_score(self, provider: ProviderConfig, topic: str) -> int:
        terms = self._extract_terms(topic)
        haystack = f"{provider.name} {provider.specialty} {provider.model}".lower()
        score = 0
        for term in terms:
            if term in haystack:
                score += 6
        if provider.duty == EXPERT_DUTY:
            score += 3
        if provider.duty == REPORT_DUTY:
            score += 1
        return score

    def _build_followup_workpackages(self, state: DiscussionState, topic_attempts: dict[str, int]) -> list[WorkPackage]:
        candidates: list[tuple[int, str, str, str]] = []
        for item in state.conflicts[-self.context_config.max_history_items:]:
            normalized = self._normalize_followup_topic(item)
            if normalized:
                title = self._followup_title_for_topic(normalized, source_kind="conflict")
                priority = self._followup_priority_score(normalized, source_kind="conflict")
                candidates.append((priority, "conflict", title, normalized))
        for item in state.open_questions[-self.context_config.max_history_items:]:
            normalized = self._normalize_followup_topic(item)
            if normalized:
                title = self._followup_title_for_topic(normalized, source_kind="open_question")
                priority = self._followup_priority_score(normalized, source_kind="open_question")
                candidates.append((priority, "open_question", title, normalized))

        workpackages: list[WorkPackage] = []
        next_index = max((task.round_index for task in state.workflow_tasks), default=state.current_round) + 1
        seen_topics: set[str] = set()
        for _, _, title, topic in sorted(candidates, key=lambda item: item[0], reverse=True):
            key = self._topic_key(topic)
            if not key or key in seen_topics:
                continue
            if topic_attempts.get(key, 0) >= self.discussion_config.max_followup_attempts:
                continue
            seen_topics.add(key)
            workpackages.append(
                WorkPackage(
                    index=next_index + len(workpackages),
                    title=title,
                    description=topic,
                    owner_name="",
                    reviewer_name="",
                )
            )
            if len(workpackages) >= self.discussion_config.max_followup_items:
                break
        return workpackages

    def _followup_title_for_topic(self, topic: str, *, source_kind: str) -> str:
        lowered = topic.lower()
        if any(token in topic for token in ["证据", "文献", "引用", "输入"]) or any(token in lowered for token in ["evidence", "citation", "literature", "input"]):
            return self._tr("证据补强", "Evidence Gap Closure")
        if any(token in topic for token in ["偏离", "偏题", "流程", "复核", "协调"]) or any(token in lowered for token in ["off-topic", "scope", "workflow", "reviewer", "coordination"]):
            return self._tr("任务收束", "Scope Correction")
        if source_kind == "conflict":
            return self._tr("争议收束", "Conflict Closure")
        return self._tr("关键未决收束", "Open-Question Closure")

    def _followup_priority_score(self, topic: str, *, source_kind: str) -> int:
        lowered = topic.lower()
        score = 6 if source_kind == "conflict" else 3
        if any(token in topic for token in ["证据", "引用", "无效", "缺失", "输入", "复核", "偏离", "偏题", "阻塞"]) or any(
            token in lowered for token in ["evidence", "citation", "invalid", "missing", "input", "review", "off-topic", "blocked"]
        ):
            score += 6
        if any(token in topic for token in ["数学", "公式", "benchmark", "实验", "效率"]) or any(
            token in lowered for token in ["math", "equation", "benchmark", "experiment", "efficiency"]
        ):
            score += 2
        if len(topic) > 120:
            score -= 1
        return score

    def _has_remaining_followups(self, state: DiscussionState, topic_attempts: dict[str, int]) -> bool:
        for topic in [*state.conflicts, *state.open_questions]:
            normalized = self._normalize_followup_topic(topic)
            if normalized and topic_attempts.get(self._topic_key(normalized), 0) < self.discussion_config.max_followup_attempts:
                return True
        return False

    def _apply_followup_resolution(self, state: DiscussionState, topic: str, resolution_text: str) -> None:
        normalized_topic = self._normalize_followup_topic(topic)
        if not normalized_topic:
            return
        if self._resolution_reached(resolution_text):
            self._remove_matching(state.conflicts, [normalized_topic, topic])
            self._remove_matching(state.open_questions, [normalized_topic, topic])
            self._merge_unique(
                state.stable_consensus,
                [self._tr(f"围绕“{normalized_topic}”已完成追加深挖并形成阶段共识。", f'Additional follow-up discussion on "{normalized_topic}" reached a provisional consensus.')],
                limit=self.context_config.max_history_items,
            )
        else:
            self._merge_unique(
                state.action_items,
                [self._tr(f"围绕“{normalized_topic}”仍需后续验证或实验。", f'Further validation or experiments are still needed for "{normalized_topic}".')],
                limit=self.context_config.max_history_items,
            )
            if normalized_topic not in state.open_questions and normalized_topic not in state.conflicts:
                self._merge_unique(state.open_questions, [normalized_topic], limit=self.context_config.max_history_items)

    def _resolution_reached(self, resolution_text: str) -> bool:
        lowered = resolution_text.lower()
        if any(token in resolution_text for token in ["仍未达成共识", "仍未解决", "保留争议", "需要更多工作"]) or any(
            token in lowered for token in ["still unresolved", "no consensus yet", "needs more work", "conflict remains"]
        ):
            return False
        return any(token in resolution_text for token in ["达成阶段共识", "可以暂时收束", "已形成共识", "暂时接受"]) or any(
            token in lowered for token in ["provisional consensus", "can be closed for now", "consensus reached", "temporarily accepted"]
        )

    def _select_recent_entries(self, state: DiscussionState, current_index: int | None) -> list[StructuredLogEntry]:
        if not state.log_entries:
            return []
        if current_index is None:
            return state.log_entries[-self.context_config.max_history_items:]
        matching = [entry for entry in state.log_entries if entry.workpackage_index in {0, current_index, current_index - 1}]
        return (matching or state.log_entries)[-self.context_config.max_history_items:]

    def _render_recent_entries(self, entries: list[StructuredLogEntry]) -> str:
        if not entries:
            return self._tr("- 暂无增量。", "- No recent updates.")
        return "\n".join(
            f"- {self._entry_display_prefix(entry)} | {entry.speaker} | {entry.headline} | {entry.summary}"
            for entry in entries
        )

    def _render_workflow_tasks(self, tasks: list[WorkflowTask]) -> str:
        if not tasks:
            return self._tr("- 暂无任务。", "- No workflow tasks.")
        return "\n".join(
            f"- {self._task_display_prefix(task)} | {task.title} | status={task.status} | owner={task.owner_name or 'TBD'} | reviewer={task.reviewer_name or 'None'}"
            for task in tasks
        )

    def _render_generated_artifacts(self, state: DiscussionState) -> str:
        if not state.generated_artifacts:
            return self._tr("- 暂无产物。", "- No generated artifacts.")
        return "\n".join(
            f"- {artifact.artifact_type} | {artifact.title} | {artifact.path}"
            for artifact in state.generated_artifacts[-6:]
        )

    def _render_indexed_lines(self, items: list[str], *, prefix: str, limit: int) -> str:
        trimmed = [item for item in items if item.strip()][-limit:]
        if not trimmed:
            return self._tr("- 暂无。", "- None.")
        return "\n".join(
            f"- {self._state_item_label(prefix, index)}: {item}"
            for index, item in enumerate(trimmed, start=1)
        )

    def _extract_workpackages(self, text: str) -> list[WorkPackage]:
        packages: list[WorkPackage] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^\s*(\d+)[\.)\u3001-]\s*(.+)$", line)
            if not match:
                continue
            index = int(match.group(1))
            body = match.group(2).strip()
            parts = [part.strip() for part in body.split("|") if part.strip()]
            if not parts:
                continue
            title = self._extract_assignment_part(parts[0]) or parts[0]
            owner_name = self._extract_assignment(
                parts,
                aliases=("Owner", "Primary", "Lead", "主责", "负责人"),
                fallback_index=1,
                allow_provider_guess=True,
            )
            reviewer_name = self._extract_assignment(
                parts,
                aliases=("Reviewer", "Review", "Audit", "复核", "审核"),
                fallback_index=2,
                allow_provider_guess=True,
            )
            description = self._extract_assignment(
                parts,
                aliases=("Description", "Why", "Reason", "Rationale", "Notes", "说明", "原因"),
                fallback_index=3,
                allow_provider_guess=False,
            )
            packages.append(WorkPackage(index=index, title=title, description=description, owner_name=owner_name, reviewer_name=reviewer_name))
        return packages

    def _extract_assignment(
        self,
        parts: list[str],
        *,
        aliases: tuple[str, ...],
        fallback_index: int | None,
        allow_provider_guess: bool,
    ) -> str:
        for part in parts[1:]:
            value = self._extract_assignment_part(part, aliases)
            if not value:
                continue
            if allow_provider_guess:
                provider_name = self._find_provider_in_text(value)
                if provider_name:
                    return provider_name
            return value

        if fallback_index is not None and len(parts) > fallback_index:
            fallback_value = self._extract_assignment_part(parts[fallback_index], aliases)
            if fallback_value:
                if allow_provider_guess:
                    provider_name = self._find_provider_in_text(fallback_value)
                    if provider_name:
                        return provider_name
                return fallback_value

        if allow_provider_guess:
            for part in parts[1:]:
                provider_name = self._find_provider_in_text(part)
                if provider_name:
                    return provider_name
        return ""

    def _extract_assignment_part(self, part: str, aliases: tuple[str, ...] = ()) -> str:
        cleaned = part.strip().strip("|")
        if not cleaned:
            return ""
        if aliases:
            for alias in aliases:
                pattern = rf"^{re.escape(alias)}\s*[:\uFF1A=-]\s*(.+)$"
                match = re.match(pattern, cleaned, re.I)
                if match:
                    return match.group(1).strip()
        generic = re.match(r"^[A-Za-z\u4e00-\u9fff\s/_-]{1,18}\s*[:\uFF1A=-]\s*(.+)$", cleaned)
        if generic:
            return generic.group(1).strip()
        return cleaned

    def _find_provider_in_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        providers = sorted(self.providers_by_name.values(), key=lambda provider: len(provider.name), reverse=True)
        for provider in providers:
            provider_name = provider.name.strip()
            if not provider_name:
                continue
            provider_lower = provider_name.lower()
            if provider_lower in lowered or lowered in provider_lower:
                return provider_name
        return ""

    def _extract_named_value(self, text: str, field_names: str | tuple[str, ...]) -> str:
        aliases = (field_names,) if isinstance(field_names, str) else field_names
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for alias in aliases:
                match = re.match(rf"^{re.escape(alias)}\s*[:\uFF1A=-]\s*(.+)$", line, re.I)
                if match:
                    return match.group(1).strip()
        return ""

    def _normalize_followup_topic(self, topic: str) -> str:
        normalized = self._collapse_whitespace(topic)
        normalized = re.sub(r"^(?:\u4e89\u8bae\u6df1\u6316|\u672a\u51b3\u95ee\u9898\u6df1\u6316|\u6df1\u6316\u95ee\u9898)\s*[:\uFF1A]\s*", "", normalized)
        return normalized.strip('\u201c\u201d" ')

    def _normalize_section_name(self, header: str) -> str | None:
        normalized = re.sub(r"\s+", " ", header.strip().lower())
        aliases = {
            "judgment": "judgment",
            "verdict": "verdict",
            "reasons": "reasons",
            "support": "support",
            "synthesis": "synthesis",
            "corrections": "corrections",
            "risk": "risk",
            "gap": "gap",
            "open gap": "open gap",
            "handoff": "handoff",
            "evidence": "evidence",
            "evidence check": "evidence check",
            "residual risk": "residual risk",
            "coordination decision": "coordination decision",
            "need more work?": "need more work",
            "need more work": "need more work",
        }
        return aliases.get(normalized)

    def _clean_candidate_line(self, raw_line: str) -> str:
        cleaned = raw_line.strip()
        if not cleaned:
            return ""
        if re.fullmatch(r"\[[^\]]+\]", cleaned):
            return ""
        cleaned = cleaned.replace("**", "").replace("`", "")
        cleaned = re.sub(r"^[-*\u2022]+\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[\.)\u3001-]\s*", "", cleaned)
        cleaned = self._collapse_whitespace(cleaned)
        lowered = cleaned.lower().rstrip(":")
        if lowered in {
            "judgment",
            "verdict",
            "reasons",
            "support",
            "synthesis",
            "corrections",
            "risk",
            "gap",
            "open gap",
            "handoff",
            "evidence",
            "evidence check",
            "residual risk",
            "coordination decision",
            "need more work",
        }:
            return ""
        if len(cleaned) < 6:
            return ""
        return cleaned

    def _extract_structured_sections(self, text: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            header_match = re.match(r"^\[([^\]]+)\]\s*$", stripped)
            if header_match:
                current = self._normalize_section_name(header_match.group(1))
                continue
            cleaned = self._clean_candidate_line(stripped)
            if not cleaned or current is None:
                continue
            bucket = sections.setdefault(current, [])
            if cleaned not in bucket:
                bucket.append(cleaned)
        return sections

    def _render_fallback_assignments(self) -> str:
        if self.output_language == "zh":
            return (
                "研究目标：围绕用户任务完成问题定义、证据梳理、方法论证和结果整合。\n"
                "领域判断：以用户问题和附件为准，保持通用学术研究框架。\n"
                "1. 问题界定 | 主责: Qwen3-Max | 复核: DeepSeek | 说明: 先澄清研究问题、边界和资料缺口。\n"
                "2. 证据梳理 | 主责: MiniMax | 复核: Qwen-Math | 说明: 提炼已有事实、数据和关键论据。\n"
                "3. 方法论证 | 主责: Qwen-Math | 复核: Qwen3-Max | 说明: 设计分析框架、验证路径和评估标准。\n"
                "4. 风险复核 | 主责: DeepSeek | 复核: MiniMax | 说明: 查找漏洞、反例和潜在幻觉。\n"
                "派工原则：按专长分工，所有关键结论都要经过至少一次交叉复核。"
            )
        return (
            "Research Goal: define the problem, organize evidence, test methodology, and integrate results around the user's task.\n"
            "Domain: stay within a general academic research framework and infer the field from the user request and attachments.\n"
            "1. Problem framing | Owner: Qwen3-Max | Reviewer: DeepSeek | Description: clarify the research question, boundaries, and information gaps first.\n"
            "2. Evidence mapping | Owner: MiniMax | Reviewer: Qwen-Math | Description: extract the key facts, data points, and supporting arguments from the materials.\n"
            "3. Method reasoning | Owner: Qwen-Math | Reviewer: Qwen3-Max | Description: design the analytical framework, validation path, and execution logic.\n"
            "4. Risk review | Owner: DeepSeek | Reviewer: MiniMax | Description: search for loopholes, counterexamples, hallucinations, and residual uncertainty.\n"
            "Assignment Principle: divide work by specialty and require at least one cross-check for every important conclusion."
        )

    def _fallback_workpackages(self) -> list[WorkPackage]:
        if self.output_language != "zh":
            return [
                WorkPackage(1, "Problem framing", "Clarify the task goal, boundary conditions, and information gaps", "Qwen3-Max", "DeepSeek"),
                WorkPackage(2, "Evidence mapping", "Extract key facts, data points, and arguments from the attachments", "MiniMax", "Qwen-Math"),
                WorkPackage(3, "Method reasoning", "Propose the analysis framework, validation method, and execution path", "Qwen-Math", "Qwen3-Max"),
                WorkPackage(4, "Risk review", "Identify hallucinations, loopholes, counterexamples, and residual uncertainty", "DeepSeek", "MiniMax"),
            ]
        return [
            WorkPackage(1, "问题界定", "明确任务目标、边界条件和资料缺口", "Qwen3-Max", "DeepSeek"),
            WorkPackage(2, "证据梳理", "提取附件中的关键事实、数据和论据", "MiniMax", "Qwen-Math"),
            WorkPackage(3, "方法论证", "提出分析框架、验证方式和执行路径", "Qwen-Math", "Qwen3-Max"),
            WorkPackage(4, "风险复核", "识别幻觉、漏洞、反例和剩余不确定性", "DeepSeek", "MiniMax"),
        ]

    def _fallback_log_line(self, message: DiscussionMessage, workpackage: str) -> str:
        return (
            f"{self._tr('轮次', 'Round')} {message.round_index}: {workpackage} | "
            f"{self._tr('记录对象', 'Logged role')}: {message.speaker} ({message.stage}) | "
            f"{self._tr('摘要', 'Summary')}: {self._collapse_whitespace(self._truncate_text(message.content, 140))}"
        )

    def _entry_display_prefix(self, entry: StructuredLogEntry) -> str:
        title = entry.workpackage_title or ""
        if any(token in title for token in ["证据补强", "任务收束", "争议收束", "关键未决收束"]):
            return self._tr(f"收束轮 {entry.workpackage_index}", f"Closure Round {entry.workpackage_index}")
        return self._tr(f"主讨论第 {entry.workpackage_index} 轮", f"Main Round {entry.workpackage_index}")

    def _task_display_prefix(self, task: WorkflowTask) -> str:
        if task.source_kind == "followup":
            return self._tr(f"收束轮 {task.round_index}", f"Closure Round {task.round_index}")
        return self._tr(f"主讨论第 {task.round_index} 轮", f"Main Round {task.round_index}")

    def _report_state_sections(self, state: DiscussionState) -> list[tuple[str, str]]:
        limit = min(8, self.context_config.max_history_items)
        sections: list[tuple[str, str]] = []
        if self.report_options.include_consensus:
            sections.append(
                (
                    self._tr("已固化共识", "Stable Consensus"),
                    self._render_indexed_lines(state.stable_consensus, prefix="K", limit=limit),
                )
            )
        sections.append(
            (
                self._tr("关键争议", "Key Conflicts"),
                self._render_indexed_lines(state.conflicts, prefix="C", limit=limit),
            )
        )
        if self.report_options.include_open_questions:
            sections.append(
                (
                    self._tr("未决问题", "Open Questions"),
                    self._render_indexed_lines(state.open_questions, prefix="Q", limit=limit),
                )
            )
        if self.report_options.include_action_items:
            sections.append(
                (
                    self._tr("后续动作", "Next Actions"),
                    self._render_indexed_lines(state.action_items, prefix="A", limit=limit),
                )
            )
        return sections

    def _build_fallback_report(self, state: DiscussionState) -> str:
        sections = [
            ("任务概述" if self.output_language == "zh" else "Task Overview", state.goal or state.topic or ("暂无" if self.output_language == "zh" else "None"))
        ]
        sections.extend(self._report_state_sections(state))
        lines = ["# 研究报告" if self.output_language == "zh" else "# Research Report", ""]
        for title, body in sections:
            lines.extend([f"## {title}", "", body, ""])
        return "\n".join(lines).strip()

    def _provider_key(self, provider: ProviderConfig) -> str:
        return self._provider_key_from_text(f"{provider.name} {provider.model} {provider.base_url}")

    def _provider_key_from_text(self, text: str) -> str:
        lowered = text.lower()
        if "kimi" in lowered or "moonshot" in lowered:
            return "kimi"
        if "glm" in lowered or "bigmodel" in lowered:
            return "glm"
        if "qwen" in lowered or "dashscope" in lowered:
            return "qwen"
        if "deepseek" in lowered:
            return "deepseek"
        if "minimax" in lowered:
            return "minimax"
        if "gpt" in lowered or "openai" in lowered:
            return "gpt"
        return "generic"

    def _prompt_budget(self, provider: ProviderConfig) -> int:
        return PROVIDER_PROMPT_BUDGETS.get(self._provider_key(provider), 8000)

    def _evidence_budget(self, provider: ProviderConfig | None) -> int:
        if provider is None:
            return 2200
        return 1200 if self._provider_key(provider) == "qwen" else 2600

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _topic_key(self, text: str) -> str:
        return self._collapse_whitespace(text).lower()

    def _artifact_stem(self, text: str) -> str:
        normalized = self._topic_key(text)
        stem = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        return stem or "research"

    def _extract_terms(self, text: str) -> list[str]:
        terms = re.findall(r"[a-z0-9_\-]{3,}|[\u4e00-\u9fff]{2,}", text.lower())
        return terms[:20]

    def _response_needs_repair(
        self,
        content: str,
        required_sections: list[str],
        *,
        min_chars: int = 96,
        allowed_evidence_ids: set[str] | None = None,
        reject_missing_review_input: bool = False,
    ) -> bool:
        if not content or content.startswith("[Call Failed]"):
            return False
        if len(self._collapse_whitespace(content)) < min_chars:
            return True
        if content.strip().lower() in {"analysis", "review", "log", "ok"}:
            return True
        if any(section not in content for section in required_sections):
            return True
        if reject_missing_review_input and self._claims_missing_review_input(content):
            return True
        return bool(self._invalid_evidence_ids(content, allowed_evidence_ids))

    def _claims_missing_review_input(self, content: str) -> bool:
        if not content:
            return False
        lowered = content.lower()
        zh_markers = [
            "输入缺失",
            "缺少输入",
            "未收到待复核",
            "未提供待复核",
            "无法执行实质复核",
            "未收到待主持复核",
            "当前未收到待复核",
            "缺少待复核的具体专家发言",
        ]
        en_markers = [
            "input missing",
            "missing input",
            "insufficient input",
            "not enough input",
            "did not receive the statement under review",
            "did not provide the statement under review",
            "cannot perform substantive review",
        ]
        return any(marker in content for marker in zh_markers) or any(marker in lowered for marker in en_markers)

    def _extract_evidence_ids(self, text: str) -> list[str]:
        matches = re.findall(r"(?i)\b(?:evd|evidence(?:\s*id)?|e)\s*[-_: ]*\[?0*(\d{1,4})\]?\b", text or "")
        ordered: list[str] = []
        seen: set[str] = set()
        for match in matches:
            normalized = f"E{int(match)}"
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _invalid_evidence_ids(self, text: str, allowed_evidence_ids: set[str] | None) -> list[str]:
        if not allowed_evidence_ids:
            return []
        return [evidence_id for evidence_id in self._extract_evidence_ids(text) if evidence_id not in allowed_evidence_ids]

    def _sanitize_document(self, text: str) -> str:
        today_cn = datetime.now().strftime("%Y年%m月%d日")
        text = text.replace("2023年X月X日", today_cn).replace("20XX年X月X日", today_cn)
        lines: list[str] = []
        previous = None
        for line in text.splitlines():
            normalized = line.strip()
            if normalized and previous == normalized:
                continue
            lines.append(line)
            previous = normalized if normalized else previous
        sanitized = "\n".join(lines)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def _is_failed_message(self, message: DiscussionMessage) -> bool:
        return message.content.startswith("[Call Failed]")

    def _extract_json_object(self, text: str) -> str | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match is None:
            return None
        return match.group(0)

    def _coerce_str_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            items = [value.strip()]
        else:
            items = []
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped[:4]

    def _candidate_lines(self, text: str) -> list[str]:
        ordered: list[str] = []
        sections = self._extract_structured_sections(text)
        for key in [
            "judgment",
            "verdict",
            "reasons",
            "support",
            "synthesis",
            "corrections",
            "risk",
            "residual risk",
            "gap",
            "open gap",
            "coordination decision",
            "need more work",
            "handoff",
            "evidence check",
        ]:
            for item in sections.get(key, []):
                if item not in ordered:
                    ordered.append(item)
        if ordered:
            return ordered[:6]

        lines: list[str] = []
        for raw_line in text.splitlines():
            cleaned = self._clean_candidate_line(raw_line)
            if not cleaned or cleaned in lines:
                continue
            lines.append(cleaned)
        return lines[:6]

    def _merge_unique(self, target: list[str], incoming: list[str], *, limit: int) -> None:
        for item in incoming:
            if not item.strip():
                continue
            if item not in target:
                target.append(item)
        if len(target) > limit:
            del target[:-limit]

    def _remove_matching(self, target: list[str], removals: list[str]) -> None:
        if not removals:
            return
        kept: list[str] = []
        for existing in target:
            lowered = existing.lower()
            if any(removal.lower() in lowered or lowered in removal.lower() for removal in removals):
                continue
            kept.append(existing)
        target[:] = kept

    def _summarize_snippet(self, snippet: AttachmentSnippet) -> str:
        collapsed = self._collapse_whitespace(snippet.content)
        return self._truncate_text(collapsed, 220)

    def _stage_label(self, stage: str) -> str:
        mapping = {
            "analysis": self._tr("分析日志", "Analysis log"),
            "review": self._tr("复核日志", "Review log"),
            "assignment": self._tr("派工日志", "Assignment log"),
            "coordination": self._tr("主持安排日志", "Coordination log"),
            "literature_review": self._tr("文献综述日志", "Literature review log"),
            "literature_analysis": self._tr("文献分析日志", "Literature analysis log"),
            "synthesis": self._tr("综合日志", "Synthesis log"),
            "coordination_review": self._tr("主持复核日志", "Host review log"),
            "followup_resolution": self._tr("深挖收束日志", "Follow-up resolution log"),
            "log": self._tr("状态日志", "State log"),
        }
        return mapping.get(stage, stage or "Log")

    def _build_cancelled_result(self, result: DiscussionResult, state: DiscussionState, message: str) -> DiscussionResult:
        result.cancelled = True
        result.meeting_state = state
        result.final_summary = self._build_cancelled_report(state, message)
        result.meeting_minutes = self._build_cancelled_minutes(state)
        self.state_manager.update_summary(state, result.final_summary)
        return result

    def _build_cancelled_report(self, state: DiscussionState, heading: str) -> str:
        sections = [("状态" if self.output_language == "zh" else "Status", heading)]
        sections.extend(self._report_state_sections(state))
        lines = ["# 研究报告" if self.output_language == "zh" else "# Research Report", ""]
        for title, body in sections:
            lines.extend([f"## {title}", "", body, ""])
        return "\n".join(lines).strip()

    def _build_cancelled_minutes(self, state: DiscussionState) -> str:
        limit = min(6, self.context_config.max_history_items)
        if self.output_language == "zh":
            return (
                "## 会议状态\n\n"
                "讨论已被手动停止。\n\n"
                "## 最近检查点\n\n"
                f"{self._build_checkpoint_timeline(state)}\n\n"
                "## 未决问题\n\n"
                f"{self._render_indexed_lines(state.open_questions, prefix='Q', limit=limit)}"
            )
        return (
            "## Meeting Status\n\n"
            "The discussion was stopped manually.\n\n"
            "## Latest Checkpoints\n\n"
            f"{self._build_checkpoint_timeline(state)}\n\n"
            "## Open Questions\n\n"
            f"{self._render_indexed_lines(state.open_questions, prefix='Q', limit=limit)}"
        )

    def _build_fallback_minutes(self, state: DiscussionState, summary_source: str = "") -> str:
        limit = min(6, self.context_config.max_history_items)
        stage_lines = "\n".join(
            f"- {record.stage_label} | status={record.status}"
            for record in state.workflow_stage_records[-8:]
        ) or self._tr("- 暂无已记录 workflow 阶段。", "- No workflow stages were recorded.")
        task_lines = "\n".join(
            f"- {task.title} | status={task.status} | owner={task.owner_name or self._tr('待定', 'TBD')}"
            for task in state.workflow_tasks[-8:]
        ) or self._tr("- 暂无已记录 workflow 任务。", "- No workflow tasks were recorded.")
        summary_block = summary_source or state.goal or state.topic or self._tr("暂无阶段总结。", "No structured summary is available yet.")
        if self.output_language == "zh":
            return (
                "## 会议状态\n\n"
                "会议已完成，以下纪要基于结构化状态自动生成。\n\n"
                "## 阶段总结\n\n"
                f"{summary_block}\n\n"
                "## Workflow 阶段轨迹\n\n"
                f"{stage_lines}\n\n"
                "## Workflow 任务\n\n"
                f"{task_lines}\n\n"
                "## 最近检查点\n\n"
                f"{self._build_checkpoint_timeline(state)}\n\n"
                "## 共识\n\n"
                f"{self._render_indexed_lines(state.stable_consensus, prefix='K', limit=limit)}\n\n"
                "## 未决问题\n\n"
                f"{self._render_indexed_lines(state.open_questions, prefix='Q', limit=limit)}\n\n"
                "## 后续动作\n\n"
                f"{self._render_indexed_lines(state.action_items, prefix='A', limit=limit)}"
            )
        return (
            "## Meeting Status\n\n"
            "The workflow completed and these minutes were generated from the structured state.\n\n"
            "## Stage Summary\n\n"
            f"{summary_block}\n\n"
            "## Workflow Stage Trace\n\n"
            f"{stage_lines}\n\n"
            "## Workflow Tasks\n\n"
            f"{task_lines}\n\n"
            "## Latest Checkpoints\n\n"
            f"{self._build_checkpoint_timeline(state)}\n\n"
            "## Consensus\n\n"
            f"{self._render_indexed_lines(state.stable_consensus, prefix='K', limit=limit)}\n\n"
            "## Open Questions\n\n"
            f"{self._render_indexed_lines(state.open_questions, prefix='Q', limit=limit)}\n\n"
            "## Next Actions\n\n"
            f"{self._render_indexed_lines(state.action_items, prefix='A', limit=limit)}"
        )
