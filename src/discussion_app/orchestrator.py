from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .attachments import (
    build_attachment_index,
    render_attachment_snippets,
    render_literature_review_context,
    select_attachment_snippets,
    select_literature_review_snippets,
    split_literature_review_packets,
    summarize_snippet_coverage,
)
from .language import detect_primary_language, language_name
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
    MeetingCheckpoint,
    MeetingState,
    ProviderConfig,
    ReaderReference,
    StructuredLogEntry,
)


MAX_WORKPACKAGES = 8
MAX_STATE_ITEMS = 12
MAX_EVIDENCE_CARDS = 24
MAX_LOG_ENTRIES = 40
MAX_CHECKPOINTS = 8
MAX_FOLLOWUP_ITEMS = 3
MAX_FOLLOWUP_ATTEMPTS = 2
MAX_LITERATURE_REVIEW_BATCHES = 3

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
6. Keep the whole output within 320 words.
"""

HOST_COORDINATION_PROMPT = f"""你是这个模拟学术研讨团队的主持人，只负责统筹规划和协作节奏，不直接给出研究结论。

{ACADEMIC_COLLABORATION_PRINCIPLE}

你会拿到当前会议状态、派工结果和会议规则。请输出一个简洁的执行计划，要求：
1. 明确子问题顺序和切换条件。
2. 提醒各执行角色如何互相校验、避免幻觉和跳步。
3. 提醒统稿人应记录哪些证据、争议和未决问题。
4. 如果主流程结束后仍存在争议，明确要求进入深挖阶段。
5. 全文控制在 220 字以内。
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
4. 全文控制在 360 字以内。
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
3. 全文控制在 240 字以内。
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
3. 全文控制在 320 字以内。
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
3. 全文控制在 260 字以内。
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
3. 全文控制在 220 字以内。
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
3. 全文控制在 220 字以内。
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

请基于会议状态、检查点、证据账本和文献综述输出 Markdown 报告，至少包含：
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

请基于会议状态、检查点和日志账本输出 Markdown 会议纪要，至少包含：
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
    def __init__(self, providers: list[ProviderConfig]) -> None:
        enabled = [provider for provider in providers if provider.enabled and provider.api_key]
        self.lead_provider = next((provider for provider in enabled if provider.duty == LEAD_DUTY), None)
        self.host_provider = next((provider for provider in enabled if provider.duty == HOST_DUTY), None)
        self.literature_provider = next((provider for provider in enabled if provider.duty == LITERATURE_DUTY), None)
        self.report_provider = next((provider for provider in enabled if provider.duty == REPORT_DUTY), None)
        self.expert_providers = [provider for provider in enabled if provider.duty == EXPERT_DUTY]
        self.enabled_providers = enabled
        self.providers_by_name = {provider.name: provider for provider in enabled}
        self.attachment_index: list[AttachmentSnippet] = []
        self.reader_references: list[ReaderReference] = []
        self.cached_pdf_reader_context = ""
        self.latest_result: DiscussionResult | None = None
        self.latest_state: MeetingState | None = None
        self.output_language = "en"

        if self.report_provider is None:
            self.report_provider = next((provider for provider in enabled if provider is not self.lead_provider), None)

    def run_discussion(
        self,
        user_request: str,
        attachments: list[AttachmentPayload],
        rounds: int = 0,
        generate_literature_review: bool = False,
        on_message: Callable[[DiscussionMessage], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> DiscussionResult:
        del rounds

        self.output_language = detect_primary_language(user_request)
        result = DiscussionResult()
        self.latest_result = result
        state = self._initialize_state(user_request)
        self.latest_state = state
        result.meeting_state = state
        self.attachment_index = build_attachment_index(attachments)
        self.reader_references = load_pdf_reader_references(attachments)
        self.cached_pdf_reader_context = render_cached_pdf_reader_context(attachments, max_chars=9000)
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

        successful_messages: list[DiscussionMessage] = []
        log_messages: list[DiscussionMessage] = []

        if should_cancel is not None and should_cancel():
            return self._build_cancelled_result(result, state, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

        assignments_text = self._render_fallback_assignments()
        workpackages = self._fallback_workpackages()
        team_roster = self._build_team_roster()
        literature_review_text = ""

        if generate_literature_review:
            if self.literature_provider is not None and attachments:
                if on_status is not None:
                    on_status(self._tr("综述专家正在生成文献综述", "Literature reviewer is generating the literature review"))
                literature_message = self._generate_literature_review(user_request)
                if not self._is_failed_message(literature_message):
                    result.literature_review = literature_message.content
                    literature_review_text = literature_message.content
                self._push_message(result, successful_messages, on_message, literature_message)
            elif on_status is not None:
                on_status(self._tr("已启用文献综述，但未找到可用的综述专家或参考附件，已跳过。", "Literature review is enabled, but no usable literature reviewer or reference attachment was found. Skipping."))

        if self.lead_provider is not None:
            if on_status is not None:
                on_status(self._tr("总负责人正在根据团队专长拆解任务", "Lead is decomposing the task based on team specialties"))
            lead_message = self._lead_assign(user_request, team_roster, literature_review_text)
            assignments_text = lead_message.content
            parsed = self._extract_workpackages(lead_message.content)
            if parsed:
                workpackages = parsed[:MAX_WORKPACKAGES]
            self._update_state_from_assignment(state, lead_message.content, workpackages)
            self._push_message(result, successful_messages, on_message, lead_message)

        if self.host_provider is not None:
            if should_cancel is not None and should_cancel():
                return self._build_cancelled_result(result, state, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))
            if on_status is not None:
                on_status(self._tr("主持人正在准备协作安排", "Host is preparing the coordination plan"))
            host_message = self._host_coordinate(user_request, assignments_text, team_roster, state, literature_review_text)
            self._update_state_from_coordination(state, host_message.content)
            self._push_message(result, successful_messages, on_message, host_message)

        kickoff_source = successful_messages[-1] if successful_messages else None
        kickoff_snippets = self._select_relevant_snippets(user_request, self.report_provider)
        kickoff_message, kickoff_entry = self._build_log_entry(
            user_request=user_request,
            state=state,
            source_message=kickoff_source,
            workpackage_title="项目启动",
            index=0,
            relevant_snippets=kickoff_snippets,
            fallback_text="已创建项目日志，等待各角色按专长执行子问题。",
        )
        self._record_log(result, successful_messages, on_message, log_messages, state, kickoff_message, kickoff_entry)
        self._create_checkpoint(state, label="初始化", workpackage_index=0)

        if not self.expert_providers:
            result.final_summary = self._generate_report(user_request, team_roster, state, literature_review_text)
            result.meeting_minutes = self._generate_meeting_minutes(
                user_request=user_request,
                team_roster=team_roster,
                state=state,
                literature_review_text=literature_review_text,
                final_report=result.final_summary,
                cancelled=False,
            )
            return result

        for workpackage in workpackages:
            if should_cancel is not None and should_cancel():
                return self._build_cancelled_result(result, state, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

            owner = self._resolve_owner(workpackage.owner_name, fallback_index=workpackage.index - 1)
            reviewer = self._resolve_reviewer(workpackage.reviewer_name, owner)
            state.current_stage = f"Task {workpackage.index}: {workpackage.title}"
            state.current_question = workpackage.display_text

            if on_status is not None:
                    on_status(self._tr(f"任务 {workpackage.index} 已启动：{workpackage.display_text}", f"Task {workpackage.index} started: {workpackage.display_text}"))

            primary_snippets = self._select_relevant_snippets(
                f"{user_request}\n{workpackage.display_text}\n{owner.specialty}\n{' '.join(state.open_questions[-3:])}",
                owner,
            )
            primary_message = self._run_primary_assignment(
                provider=owner,
                user_request=user_request,
                assignments_text=assignments_text,
                team_roster=team_roster,
                literature_review_text=literature_review_text,
                workpackage=workpackage,
                state=state,
                relevant_snippets=primary_snippets,
                attachments=attachments,
            )
            self._push_message(result, successful_messages, on_message, primary_message)

            primary_log_message, primary_entry = self._build_log_entry(
                user_request=user_request,
                state=state,
                source_message=primary_message,
                workpackage_title=workpackage.display_text,
                index=workpackage.index,
                relevant_snippets=primary_snippets,
                fallback_text=self._fallback_log_line(primary_message, workpackage.display_text),
            )
            self._record_log(result, successful_messages, on_message, log_messages, state, primary_log_message, primary_entry)

            if reviewer is not None:
                if should_cancel is not None and should_cancel():
                    return self._build_cancelled_result(result, state, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

                reviewer_snippets = self._select_relevant_snippets(
                    f"{user_request}\n{workpackage.display_text}\n{reviewer.specialty}\n{primary_message.content}",
                    reviewer,
                )
                review_message = self._run_review_assignment(
                    provider=reviewer,
                    user_request=user_request,
                    assignments_text=assignments_text,
                    team_roster=team_roster,
                    literature_review_text=literature_review_text,
                    workpackage=workpackage,
                    previous_message=primary_message,
                    state=state,
                    relevant_snippets=reviewer_snippets,
                )
                self._push_message(result, successful_messages, on_message, review_message)

                review_log_message, review_entry = self._build_log_entry(
                    user_request=user_request,
                    state=state,
                    source_message=review_message,
                    workpackage_title=workpackage.display_text,
                    index=workpackage.index,
                    relevant_snippets=reviewer_snippets,
                    fallback_text=self._fallback_log_line(review_message, workpackage.display_text),
                )
                self._record_log(result, successful_messages, on_message, log_messages, state, review_log_message, review_entry)

            checkpoint = self._create_checkpoint(state, label=workpackage.title, workpackage_index=workpackage.index)
            if on_status is not None:
                    on_status(self._tr(f"{self._checkpoint_label(checkpoint.checkpoint_id)} 已记录：{checkpoint.label}", f"{self._checkpoint_label(checkpoint.checkpoint_id)} recorded: {checkpoint.label}"))

        self._run_consensus_followups(
            result=result,
            state=state,
            user_request=user_request,
            attachments=attachments,
            assignments_text=assignments_text,
            team_roster=team_roster,
            literature_review_text=literature_review_text,
            successful_messages=successful_messages,
            log_messages=log_messages,
            on_message=on_message,
            on_status=on_status,
            should_cancel=should_cancel,
        )

        if should_cancel is not None and should_cancel():
            return self._build_cancelled_result(result, state, self._tr("讨论已被手动停止。", "Discussion was stopped manually."))

        if on_status is not None:
            on_status(self._tr("统稿人正在根据会议状态生成研究报告", "Reporter is synthesizing the research report from the meeting state"))
        result.final_summary = self._generate_report(user_request, team_roster, state, literature_review_text)

        if on_status is not None:
            on_status(self._tr("统稿人正在根据会议状态撰写会议纪要", "Reporter is drafting the meeting minutes from the meeting state"))
        result.meeting_minutes = self._generate_meeting_minutes(
            user_request=user_request,
            team_roster=team_roster,
            state=state,
            literature_review_text=literature_review_text,
            final_report=result.final_summary,
            cancelled=False,
        )
        return result

    def _initialize_state(self, user_request: str) -> MeetingState:
        return MeetingState(
            topic=self._collapse_whitespace(self._truncate_text(user_request, 160)),
            goal=self._collapse_whitespace(self._truncate_text(user_request, 220)),
            rules=self._localized_meeting_rules(),
            current_stage="Initialization",
            current_question="Waiting for the lead to decompose the task",
        )

    def _tr(self, zh_text: str, en_text: str) -> str:
        del zh_text
        return en_text

    def _localized_meeting_rules(self) -> list[str]:
        return [
            "Important claims must be tied to evidence, theoretical support, or explicitly marked as pending verification.",
            "Consensus, conflicts, hypotheses, and open questions must be recorded separately.",
            "Each role should focus only on its assigned subproblem and avoid replaying long history.",
            "If hallucinations, vague definitions, or missing evidence appear, they must be logged explicitly.",
            "If open questions or conflicts remain after the main flow, follow-up discussion passes are mandatory.",
        ]

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

    def _lead_assign(
        self,
        user_request: str,
        team_roster: str,
        literature_review_text: str,
    ) -> DiscussionMessage:
        assert self.lead_provider is not None
        snippets = self._select_relevant_snippets(user_request, self.lead_provider, max_snippets=4)
        reader_context = self._build_reader_context(user_request, self.lead_provider, max_chars=1800, max_items=4)
        reader_attachments = self._build_reader_attachments(user_request, self.lead_provider, max_items=2)
        prompt = (
            f"User task:\n{user_request}\n\n"
            f"Team roster and specialties:\n{team_roster}\n\n"
            f"Relevant attachment snippets:\n{render_attachment_snippets(snippets, max_chars=2200) or 'No attachment snippets.'}\n\n"
            f"PDF reader retrieval:\n{reader_context}\n\n"
            f"Literature review context:\n{self._build_literature_context(literature_review_text, 1400)}\n\n"
            "Generate the delegation plan using the fixed English schema labels."
        )
        content = self._chat(
            provider=self.lead_provider,
            system_prompt=LEAD_ASSIGNMENT_PROMPT,
            user_prompt=prompt,
            max_tokens=480,
            attachments=reader_attachments,
            max_continuations=1,
        )
        return DiscussionMessage(
            speaker=self.lead_provider.name,
            role="assistant",
            content=content,
            round_index=0,
            model_name=self.lead_provider.model,
            duty=LEAD_DUTY,
            stage="assignment",
        )

    def _host_coordinate(
        self,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        state: MeetingState,
        literature_review_text: str,
    ) -> DiscussionMessage:
        assert self.host_provider is not None
        host_query = f"{user_request}\n{assignments_text}\n{state.current_question}"
        reader_context = self._build_reader_context(host_query, self.host_provider, max_chars=1600, max_items=4)
        reader_attachments = self._build_reader_attachments(host_query, self.host_provider, max_items=2)
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
            provider=self.host_provider,
            system_prompt=HOST_COORDINATION_PROMPT,
            user_prompt=prompt,
            max_tokens=320,
            attachments=reader_attachments,
            max_continuations=0,
        )
        return DiscussionMessage(
            speaker=self.host_provider.name,
            role="assistant",
            content=content,
            round_index=0,
            model_name=self.host_provider.model,
            duty=HOST_DUTY,
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
        state: MeetingState,
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
        state: MeetingState,
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
        state: MeetingState,
        relevant_snippets: list[AttachmentSnippet],
        attachments: list[AttachmentPayload],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{provider.specialty}\n{state.current_question}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1800, max_items=5)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"主责角色：{provider.name}\n"
            f"你的专长：{provider.specialty or '未填写'}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='expert')}\n\n"
            f"相关证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=self._evidence_budget(provider)) or '暂无可检索证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1200)}\n\n"
            "请只处理当前子问题。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=EXPERT_ANALYSIS_PROMPT,
            user_prompt=prompt,
            max_tokens=620,
            attachments=self._merge_chat_attachments(attachments, reader_attachments),
            max_continuations=1,
            required_sections=["[Judgment]", "[Reasons]", "[Evidence]", "[Risk]", "[Handoff]"],
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
        state: MeetingState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{provider.specialty}\n{previous_message.content}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1600, max_items=4)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"复核角色：{provider.name}\n"
            f"你的专长：{provider.specialty or '未填写'}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='reviewer')}\n\n"
            f"相关证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=self._evidence_budget(provider)) or '暂无可检索证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1000)}\n\n"
            f"待复核发言：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 1200)}\n\n"
            "请基于你的专长进行复核。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=EXPERT_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=360,
            attachments=reader_attachments,
            max_continuations=0,
            required_sections=["[Verdict]", "[Corrections]", "[Evidence Check]", "[Residual Risk]"],
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
        state: MeetingState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{provider.specialty}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=2200, max_items=6)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=3)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='expert')}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1500)}\n\n"
            f"相关文献证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=2200) or '暂无可检索证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            "请从文献支持角度回答当前子问题。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=LITERATURE_ANALYSIS_PROMPT,
            user_prompt=prompt,
            max_tokens=420,
            attachments=reader_attachments,
            max_continuations=1,
            required_sections=["[Judgment]", "[Support]", "[Gap]", "[Risk]", "[Handoff]"],
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
        state: MeetingState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{previous_message.content}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=2200, max_items=6)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=3)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='reviewer')}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1500)}\n\n"
            f"相关文献证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=2200) or '暂无可检索证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"待复核发言：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 1200)}\n\n"
            "请从文献支持和相关工作角度复核。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=EXPERT_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=320,
            attachments=reader_attachments,
            max_continuations=0,
            required_sections=["[Verdict]", "[Corrections]", "[Evidence Check]", "[Residual Risk]"],
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
        state: MeetingState,
        relevant_snippets: list[AttachmentSnippet],
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{state.current_question}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1500, max_items=4)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='report')}\n\n"
            f"相关证据片段：\n{render_attachment_snippets(relevant_snippets, max_chars=1800) or '暂无可检索证据片段。'}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"文献综述参考：\n{self._build_literature_context(literature_review_text, 1000)}\n\n"
            "请整合已有观点，给出当前子问题的结构化综合判断。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=REPORT_SYNTHESIS_PROMPT,
            user_prompt=prompt,
            max_tokens=360,
            attachments=reader_attachments,
            max_continuations=0,
            required_sections=["[Judgment]", "[Synthesis]", "[Open Gap]", "[Handoff]"],
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
        state: MeetingState,
    ) -> DiscussionMessage:
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='report')}\n\n"
            f"待复核整合稿：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 1200)}\n\n"
            "请判断当前整合稿是否足够支撑收束。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=HOST_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=280,
            max_continuations=0,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
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
        state: MeetingState,
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
            max_tokens=280,
            attachments=reader_attachments,
            max_continuations=0,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
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
        state: MeetingState,
    ) -> DiscussionMessage:
        reader_query = f"{user_request}\n{workpackage.display_text}\n{previous_message.content}"
        reader_context = self._build_reader_context(reader_query, provider, max_chars=1400, max_items=4)
        reader_attachments = self._build_reader_attachments(reader_query, provider, max_items=2)
        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"总负责派工：\n{assignments_text}\n\n"
            f"当前子问题：任务 {workpackage.index} - {workpackage.display_text}\n\n"
            f"当前会议状态快照：\n{self._build_state_snapshot(state, workpackage=workpackage, mode='host')}\n\n"
            f"PDF reader 索引检索：\n{reader_context}\n\n"
            f"待主持复核发言：\n[{previous_message.speaker}]\n{self._truncate_text(previous_message.content, 1200)}\n\n"
            "请判断该子问题是否可以暂时收束。"
        )
        content = self._chat_with_sections(
            provider=provider,
            system_prompt=HOST_REVIEW_PROMPT,
            user_prompt=prompt,
            max_tokens=280,
            attachments=reader_attachments,
            max_continuations=0,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
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
        state: MeetingState,
        source_message: DiscussionMessage | None,
        workpackage_title: str,
        index: int,
        relevant_snippets: list[AttachmentSnippet],
        fallback_text: str,
    ) -> tuple[DiscussionMessage, StructuredLogEntry]:
        if source_message is None or self.report_provider is None or self._is_failed_message(source_message):
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
            f"PDF reader 索引检索：\n{self._build_reader_context(f'{user_request}\\n{workpackage_title}\\n{source_message.content}', self.report_provider, max_chars=1400, max_items=4)}\n\n"
            f"最新讨论内容：\n[{source_message.speaker} | {source_message.stage}]\n{self._truncate_text(source_message.content, 1200)}\n\n"
            "请输出状态补丁 JSON。"
        )
        content = self._chat(
            provider=self.report_provider,
            system_prompt=REPORT_LOG_PROMPT,
            user_prompt=prompt,
            max_tokens=360,
            attachments=self._build_reader_attachments(f"{user_request}\n{workpackage_title}\n{source_message.content}", self.report_provider, max_items=2),
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
        state: MeetingState,
        literature_review_text: str,
    ) -> str:
        report_provider = self.report_provider or self.host_provider or self.lead_provider
        if report_provider is None:
            return self._build_fallback_report(state)

        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"会议状态总览：\n{self._build_state_snapshot(state, workpackage=None, mode='report')}\n\n"
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
            max_tokens=1500,
            attachments=self._build_reader_attachments(user_request + "\n" + state.current_question, report_provider, max_items=2),
            max_continuations=1,
        )
        return self._sanitize_document(content)

    def _generate_meeting_minutes(
        self,
        *,
        user_request: str,
        team_roster: str,
        state: MeetingState,
        literature_review_text: str,
        final_report: str,
        cancelled: bool,
    ) -> str:
        minutes_provider = self.report_provider or self.host_provider or self.lead_provider
        if minutes_provider is None:
            return self._build_cancelled_minutes(state) if cancelled else final_report

        prompt = (
            f"用户任务：\n{user_request}\n\n"
            f"团队成员与专长：\n{team_roster}\n\n"
            f"会议状态总览：\n{self._build_state_snapshot(state, workpackage=None, mode='minutes')}\n\n"
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
            max_tokens=1200,
            attachments=self._build_reader_attachments(user_request + "\n" + state.current_question, minutes_provider, max_items=2),
            max_continuations=1,
        )
        return self._sanitize_document(content)

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
    ) -> str:
        content = self._chat(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            attachments=attachments,
            max_continuations=max_continuations,
        )
        if not self._response_needs_repair(content, required_sections):
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
        if not self._response_needs_repair(repaired, required_sections):
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
    ) -> str:
        return self._chat_with_sections(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            required_sections=required_sections,
            attachments=attachments,
            max_continuations=max_continuations,
        )

    def _generate_literature_review(self, user_request: str) -> DiscussionMessage:
        assert self.literature_provider is not None
        snippets = select_literature_review_snippets(
            self.attachment_index,
            max_snippets=16,
            max_chars=15000,
        )
        cached_context = self._truncate_text(self.cached_pdf_reader_context, 8500)
        if not snippets and cached_context:
            return DiscussionMessage(
                speaker=self.literature_provider.name,
                role="assistant",
                content=f"## {self._tr('PDF Reader 缓存摘要', 'Cached PDF Reader Digest')}\n\n{cached_context}",
                round_index=0,
                model_name=self.literature_provider.model,
                duty=LITERATURE_DUTY,
                stage="literature_review",
            )
        if not snippets:
            return DiscussionMessage(
                speaker=self.literature_provider.name,
                role="assistant",
                content=self._tr(
                    "[调用失败] 没有可用于文献综述的文本附件。",
                    "[Call Failed] No text attachment was available for literature review.",
                ),
                round_index=0,
                model_name=self.literature_provider.model,
                duty=LITERATURE_DUTY,
                stage="literature_review",
            )

        packets = split_literature_review_packets(
            snippets,
            max_chars_per_packet=4200,
            max_packets=MAX_LITERATURE_REVIEW_BATCHES,
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
                provider=self.literature_provider,
                system_prompt=LITERATURE_PACKET_NOTES_PROMPT,
                user_prompt=note_prompt,
                max_tokens=850,
                required_sections=required_packet_sections,
                max_continuations=1,
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
            provider=self.literature_provider,
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
        )
        return DiscussionMessage(
            speaker=self.literature_provider.name,
            role="assistant",
            content=content,
            round_index=0,
            model_name=self.literature_provider.model,
            duty=LITERATURE_DUTY,
            stage="literature_review",
        )

    def _run_consensus_followups(
        self,
        *,
        result: DiscussionResult,
        state: MeetingState,
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
    ) -> None:
        topic_attempts: dict[str, int] = {}

        for pass_index in range(1, MAX_FOLLOWUP_ATTEMPTS + 1):
            followups = self._build_followup_workpackages(state, topic_attempts)
            if not followups:
                return

            if on_status is not None:
                on_status(self._tr(f"主持人正在组织第 {pass_index} 轮未决问题深挖", f"Host is organizing unresolved-issue pass {pass_index}"))

            for workpackage in followups:
                if should_cancel is not None and should_cancel():
                    return

                topic_key = self._topic_key(workpackage.display_text)
                topic_attempts[topic_key] = topic_attempts.get(topic_key, 0) + 1
                owner, reviewer = self._resolve_followup_pair(workpackage.display_text)
                state.current_stage = f"未决深挖 {pass_index}: {workpackage.title}"
                state.current_question = workpackage.display_text

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

                checkpoint = self._create_checkpoint(state, label=workpackage.title, workpackage_index=workpackage.index)
                if on_status is not None:
                    on_status(self._tr(f"未决问题 {self._checkpoint_label(checkpoint.checkpoint_id)} 已更新：{checkpoint.label}", f"Unresolved-issue {self._checkpoint_label(checkpoint.checkpoint_id)} updated: {checkpoint.label}"))

            if not self._has_remaining_followups(state, topic_attempts):
                return

    def _run_host_resolution(
        self,
        *,
        user_request: str,
        assignments_text: str,
        team_roster: str,
        workpackage: WorkPackage,
        state: MeetingState,
        primary_message: DiscussionMessage,
        review_message: DiscussionMessage | None,
    ) -> DiscussionMessage:
        assert self.host_provider is not None
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
            provider=self.host_provider,
            system_prompt=FOLLOWUP_HOST_PROMPT,
            user_prompt=prompt,
            max_tokens=320,
            max_continuations=0,
            required_sections=["[Verdict]", "[Coordination Decision]", "[Need More Work?]", "[Residual Risk]"],
        )
        return DiscussionMessage(
            speaker=self.host_provider.name,
            role="assistant",
            content=content,
            round_index=workpackage.index,
            model_name=self.host_provider.model,
            duty=self.host_provider.duty,
            stage="followup_resolution",
        )

    def _update_state_from_assignment(self, state: MeetingState, assignment_text: str, workpackages: list[WorkPackage]) -> None:
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

    def _update_state_from_coordination(self, state: MeetingState, coordination_text: str) -> None:
        state.coordination_summary = self._truncate_text(coordination_text, 800)

    def _record_log(
        self,
        result: DiscussionResult,
        successful_messages: list[DiscussionMessage],
        on_message: Callable[[DiscussionMessage], None] | None,
        log_messages: list[DiscussionMessage],
        state: MeetingState,
        log_message: DiscussionMessage,
        entry: StructuredLogEntry,
    ) -> None:
        self._apply_log_entry(state, entry)
        self._push_message(result, successful_messages, on_message, log_message)
        log_messages.append(log_message)

    def _apply_log_entry(self, state: MeetingState, entry: StructuredLogEntry) -> None:
        state.log_entries.append(entry)
        if len(state.log_entries) > MAX_LOG_ENTRIES:
            state.log_entries = state.log_entries[-MAX_LOG_ENTRIES:]

        self._merge_unique(state.stable_consensus, entry.consensus_add, limit=MAX_STATE_ITEMS)
        self._merge_unique(state.conflicts, entry.conflicts_add, limit=MAX_STATE_ITEMS)
        self._merge_unique(state.open_questions, entry.open_questions_add, limit=MAX_STATE_ITEMS)
        self._merge_unique(state.rejected_lines, entry.rejected_add, limit=MAX_STATE_ITEMS)
        self._merge_unique(state.action_items, entry.action_items_add, limit=MAX_STATE_ITEMS)
        self._remove_matching(state.conflicts, entry.resolved_conflicts)
        self._remove_matching(state.open_questions, entry.resolved_questions)

        evidence_by_id = {card.evidence_id: card for card in state.evidence_cards}
        for card in entry.evidence_add:
            if card.evidence_id in evidence_by_id:
                continue
            state.evidence_cards.append(card)
        if len(state.evidence_cards) > MAX_EVIDENCE_CARDS:
            state.evidence_cards = state.evidence_cards[-MAX_EVIDENCE_CARDS:]

    def _create_checkpoint(self, state: MeetingState, *, label: str, workpackage_index: int) -> MeetingCheckpoint:
        recent_entries = [entry for entry in state.log_entries if entry.workpackage_index == workpackage_index][-3:]
        summary_parts = [self._tr(f"完成阶段：{label}", f"Completed stage: {label}")]
        if recent_entries:
            summary_parts.append(self._tr("；", "; ").join(entry.headline for entry in recent_entries if entry.headline))
        elif state.stable_consensus:
            summary_parts.append(self._tr("近期共识：", "Recent consensus: ") + self._tr("；", "; ").join(state.stable_consensus[-2:]))

        checkpoint = MeetingCheckpoint(
            checkpoint_id=f"CP{len(state.checkpoints) + 1}",
            label=label,
            workpackage_index=workpackage_index,
            summary=self._truncate_text(" ".join(summary_parts), 220),
            consensus=state.stable_consensus[-4:],
            conflicts=state.conflicts[-3:],
            open_questions=state.open_questions[-3:],
            action_items=state.action_items[-4:],
        )
        state.checkpoints.append(checkpoint)
        if len(state.checkpoints) > MAX_CHECKPOINTS:
            state.checkpoints = state.checkpoints[-MAX_CHECKPOINTS:]
        return checkpoint

    def _build_state_snapshot(
        self,
        state: MeetingState,
        *,
        workpackage: WorkPackage | None = None,
        workpackage_index: int | None = None,
        mode: str,
    ) -> str:
        current_index = workpackage.index if workpackage is not None else workpackage_index
        checkpoint = state.checkpoints[-1] if state.checkpoints else None
        relevant_entries = self._select_recent_entries(state, current_index)

        parts = [
            f"{self._tr('主题', 'Topic')}: {state.topic or self._tr('未设定', 'Not set')}",
            f"{self._tr('领域', 'Domain')}: {state.domain or self._tr('待讨论判断', 'To be determined during discussion')}",
            f"{self._tr('研究目标', 'Research Goal')}: {state.goal or self._tr('未提炼', 'Not distilled yet')}",
            f"{self._tr('当前阶段', 'Current Stage')}: {state.current_stage or self._tr('未开始', 'Not started')}",
            f"{self._tr('当前子问题', 'Current Workpackage')}: {workpackage.display_text if workpackage is not None else state.current_question or self._tr('未设定', 'Not set')}",
            f"{self._tr('会议规则', 'Meeting Rules')}:",
            self._render_indexed_lines(state.rules, prefix="R", limit=5),
        ]

        if state.assignment_summary:
            parts.append(f"{self._tr('派工摘要', 'Assignment Summary')}:\n{self._truncate_text(state.assignment_summary, 320)}")
        if mode in {"host", "report", "minutes"} and state.coordination_summary:
            parts.append(f"{self._tr('主持安排摘要', 'Coordination Summary')}:\n{self._truncate_text(state.coordination_summary, 260)}")
        if checkpoint is not None:
            parts.append(f"{self._tr('最近检查点', 'Latest Checkpoint')} {checkpoint.checkpoint_id}: {checkpoint.summary}")

        parts.append(f"{self._tr('稳定共识', 'Stable Consensus')}:")
        parts.append(self._render_indexed_lines(state.stable_consensus, prefix="K", limit=6))
        parts.append(f"{self._tr('当前争议', 'Active Conflicts')}:")
        parts.append(self._render_indexed_lines(state.conflicts, prefix="C", limit=5))
        parts.append(f"{self._tr('未决问题', 'Open Questions')}:")
        parts.append(self._render_indexed_lines(state.open_questions, prefix="Q", limit=5))

        if mode in {"expert", "reviewer", "logger", "report", "minutes", "host"}:
            parts.append(f"{self._tr('近期增量', 'Recent Updates')}:")
            parts.append(self._render_recent_entries(relevant_entries))

        if mode in {"report", "minutes", "host"}:
            parts.append(f"{self._tr('行动项', 'Action Items')}:")
            parts.append(self._render_indexed_lines(state.action_items, prefix="A", limit=6))

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
        ordered: list[ProviderConfig] = []
        for provider in [self.lead_provider, self.host_provider, self.literature_provider, *self.expert_providers, self.report_provider]:
            if provider is not None and provider not in ordered:
                ordered.append(provider)
        return "\n".join(
            f"- {provider.name} | {provider.duty} | {self._tr('专长', 'Specialty')}: {provider.specialty or self._tr('未填写', 'Not set')}"
            for provider in ordered
        ) or self._tr("- 暂无有效角色", "- No active roles")

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

    def _build_checkpoint_timeline(self, state: MeetingState) -> str:
        if not state.checkpoints:
            return self._tr("- 暂无检查点。", "- No checkpoints yet.")
        return "\n".join(
            f"- {self._checkpoint_label(checkpoint.checkpoint_id)} | {self._tr('任务', 'Task')} {checkpoint.workpackage_index} | {checkpoint.label} | {checkpoint.summary}"
            for checkpoint in state.checkpoints[-MAX_CHECKPOINTS:]
        )

    def _build_evidence_ledger(self, state: MeetingState) -> str:
        if not state.evidence_cards:
            return self._tr("- 暂无已固化证据引用。", "- No evidence references have been consolidated yet.")
        return "\n".join(
            f"- {card.display_label or self._fallback_evidence_label(card.evidence_id, card.source)} | {self._tr('任务', 'Task')} {card.workpackage_index} | {card.summary}"
            for card in state.evidence_cards[-MAX_EVIDENCE_CARDS:]
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

    def _build_followup_workpackages(self, state: MeetingState, topic_attempts: dict[str, int]) -> list[WorkPackage]:
        candidates: list[tuple[str, str]] = []
        for item in state.conflicts[-MAX_STATE_ITEMS:]:
            normalized = self._normalize_followup_topic(item)
            if normalized:
                candidates.append(("\u4e89\u8bae\u6df1\u6316", normalized))
        for item in state.open_questions[-MAX_STATE_ITEMS:]:
            normalized = self._normalize_followup_topic(item)
            if normalized:
                candidates.append(("\u672a\u51b3\u95ee\u9898\u6df1\u6316", normalized))

        workpackages: list[WorkPackage] = []
        base_index = 100 + len(state.checkpoints) * 10
        seen_topics: set[str] = set()
        for title, topic in candidates:
            key = self._topic_key(topic)
            if not key or key in seen_topics:
                continue
            if topic_attempts.get(key, 0) >= MAX_FOLLOWUP_ATTEMPTS:
                continue
            seen_topics.add(key)
            workpackages.append(
                WorkPackage(
                    index=base_index + len(workpackages) + 1,
                    title=title,
                    description=topic,
                    owner_name="",
                    reviewer_name="",
                )
            )
            if len(workpackages) >= MAX_FOLLOWUP_ITEMS:
                break
        return workpackages

    def _has_remaining_followups(self, state: MeetingState, topic_attempts: dict[str, int]) -> bool:
        for topic in [*state.conflicts, *state.open_questions]:
            normalized = self._normalize_followup_topic(topic)
            if normalized and topic_attempts.get(self._topic_key(normalized), 0) < MAX_FOLLOWUP_ATTEMPTS:
                return True
        return False

    def _apply_followup_resolution(self, state: MeetingState, topic: str, resolution_text: str) -> None:
        normalized_topic = self._normalize_followup_topic(topic)
        if not normalized_topic:
            return
        if self._resolution_reached(resolution_text):
            self._remove_matching(state.conflicts, [normalized_topic, topic])
            self._remove_matching(state.open_questions, [normalized_topic, topic])
            self._merge_unique(
                state.stable_consensus,
                [self._tr(f"围绕“{normalized_topic}”已完成追加深挖并形成阶段共识。", f'Additional follow-up discussion on "{normalized_topic}" reached a provisional consensus.')],
                limit=MAX_STATE_ITEMS,
            )
        else:
            self._merge_unique(
                state.action_items,
                [self._tr(f"围绕“{normalized_topic}”仍需后续验证或实验。", f'Further validation or experiments are still needed for "{normalized_topic}".')],
                limit=MAX_STATE_ITEMS,
            )
            if normalized_topic not in state.open_questions and normalized_topic not in state.conflicts:
                self._merge_unique(state.open_questions, [normalized_topic], limit=MAX_STATE_ITEMS)

    def _resolution_reached(self, resolution_text: str) -> bool:
        lowered = resolution_text.lower()
        if any(token in resolution_text for token in ["仍未达成共识", "仍未解决", "保留争议", "需要更多工作"]) or any(
            token in lowered for token in ["still unresolved", "no consensus yet", "needs more work", "conflict remains"]
        ):
            return False
        return any(token in resolution_text for token in ["达成阶段共识", "可以暂时收束", "已形成共识", "暂时接受"]) or any(
            token in lowered for token in ["provisional consensus", "can be closed for now", "consensus reached", "temporarily accepted"]
        )

    def _select_recent_entries(self, state: MeetingState, current_index: int | None) -> list[StructuredLogEntry]:
        if not state.log_entries:
            return []
        if current_index is None:
            return state.log_entries[-4:]
        matching = [entry for entry in state.log_entries if entry.workpackage_index in {0, current_index, current_index - 1}]
        return (matching or state.log_entries)[-4:]

    def _render_recent_entries(self, entries: list[StructuredLogEntry]) -> str:
        if not entries:
            return self._tr("- 暂无增量。", "- No recent updates.")
        return "\n".join(
            f"- {self._tr('任务', 'Task')} {entry.workpackage_index} | {entry.speaker} | {entry.headline} | {entry.summary}"
            for entry in entries
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
            f"{self._tr('任务', 'Task')} {message.round_index}: {workpackage} | "
            f"{self._tr('记录对象', 'Logged role')}: {message.speaker} ({message.stage}) | "
            f"{self._tr('摘要', 'Summary')}: {self._collapse_whitespace(self._truncate_text(message.content, 140))}"
        )

    def _build_fallback_report(self, state: MeetingState) -> str:
        if self.output_language == "zh":
            return (
                "# 研究报告\n\n"
                f"## 任务概述\n\n{state.goal or state.topic or '暂无'}\n\n"
                "## 已固化共识\n\n"
                f"{self._render_indexed_lines(state.stable_consensus, prefix='K', limit=8)}\n\n"
                "## 关键争议\n\n"
                f"{self._render_indexed_lines(state.conflicts, prefix='C', limit=6)}\n\n"
                "## 后续动作\n\n"
                f"{self._render_indexed_lines(state.action_items, prefix='A', limit=6)}"
            )
        return (
            "# Research Report\n\n"
            f"## Task Overview\n\n{state.goal or state.topic or 'None'}\n\n"
            "## Stable Consensus\n\n"
            f"{self._render_indexed_lines(state.stable_consensus, prefix='K', limit=8)}\n\n"
            "## Key Conflicts\n\n"
            f"{self._render_indexed_lines(state.conflicts, prefix='C', limit=6)}\n\n"
            "## Next Actions\n\n"
            f"{self._render_indexed_lines(state.action_items, prefix='A', limit=6)}"
        )

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

    def _extract_terms(self, text: str) -> list[str]:
        terms = re.findall(r"[a-z0-9_\-]{3,}|[\u4e00-\u9fff]{2,}", text.lower())
        return terms[:20]

    def _response_needs_repair(self, content: str, required_sections: list[str]) -> bool:
        if not content or content.startswith("[Call Failed]"):
            return False
        if len(self._collapse_whitespace(content)) < 48:
            return True
        if content.strip().lower() in {"analysis", "review", "log", "ok"}:
            return True
        return any(section not in content for section in required_sections)

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

    def _build_cancelled_result(self, result: DiscussionResult, state: MeetingState, message: str) -> DiscussionResult:
        result.cancelled = True
        result.meeting_state = state
        result.final_summary = self._build_cancelled_report(state, message)
        result.meeting_minutes = self._build_cancelled_minutes(state)
        return result

    def _build_cancelled_report(self, state: MeetingState, heading: str) -> str:
        if self.output_language == "zh":
            return (
                "# 研究报告\n\n"
                f"## 状态\n\n{heading}\n\n"
                "## 已固化共识\n\n"
                f"{self._render_indexed_lines(state.stable_consensus, prefix='K', limit=6)}\n\n"
                "## 当前争议\n\n"
                f"{self._render_indexed_lines(state.conflicts, prefix='C', limit=6)}\n\n"
                "## 下一步动作\n\n"
                f"{self._render_indexed_lines(state.action_items, prefix='A', limit=6)}"
            )
        return (
            "# Research Report\n\n"
            f"## Status\n\n{heading}\n\n"
            "## Stable Consensus\n\n"
            f"{self._render_indexed_lines(state.stable_consensus, prefix='K', limit=6)}\n\n"
            "## Active Conflicts\n\n"
            f"{self._render_indexed_lines(state.conflicts, prefix='C', limit=6)}\n\n"
            "## Next Actions\n\n"
            f"{self._render_indexed_lines(state.action_items, prefix='A', limit=6)}"
        )

    def _build_cancelled_minutes(self, state: MeetingState) -> str:
        if self.output_language == "zh":
            return (
                "## 会议状态\n\n"
                "讨论已被手动停止。\n\n"
                "## 最近检查点\n\n"
                f"{self._build_checkpoint_timeline(state)}\n\n"
                "## 未决问题\n\n"
                f"{self._render_indexed_lines(state.open_questions, prefix='Q', limit=6)}"
            )
        return (
            "## Meeting Status\n\n"
            "The discussion was stopped manually.\n\n"
            "## Latest Checkpoints\n\n"
            f"{self._build_checkpoint_timeline(state)}\n\n"
            "## Open Questions\n\n"
            f"{self._render_indexed_lines(state.open_questions, prefix='Q', limit=6)}"
        )
