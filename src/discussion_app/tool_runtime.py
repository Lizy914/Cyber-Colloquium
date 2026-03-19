from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .roles import MODERATOR_ROLE_KEY, NOTETAKER_ROLE_KEY, RESEARCH_ANALYST_ROLE_KEY, REVIEWER_ROLE_KEY


TOOL_STATUS_SUCCESS = "success"
TOOL_STATUS_DENIED = "denied"
TOOL_STATUS_FAILED = "failed"

PYTHON_EXECUTION_TOOL_KEY = "python_execution"
LATEX_GENERATION_TOOL_KEY = "latex_generation"
BIBTEX_GENERATION_TOOL_KEY = "bibtex_generation"
FIGURE_GENERATION_TOOL_KEY = "figure_generation"
ARTIFACT_EXPORT_TOOL_KEY = "artifact_export"
MOCK_ARTIFACT_EXPORT_TOOL_KEY = "mock_artifact_export"


@dataclass(frozen=True)
class ToolSpec:
    key: str
    name: str
    description: str
    capability: str
    outputs_artifacts: bool = False


@dataclass(frozen=True)
class ToolArtifact:
    artifact_type: str
    title: str
    content: str = ""
    path_hint: str = ""
    mime_type: str = "text/plain"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionRequest:
    tool_key: str
    role_key: str
    payload: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    project_id: str = ""
    user_request: str = ""
    working_directory: str = ""


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_key: str
    status: str
    message: str = ""
    output_text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[ToolArtifact, ...] = ()
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == TOOL_STATUS_SUCCESS


class ResearchTool(Protocol):
    spec: ToolSpec

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        ...


class ToolRegistry:
    def __init__(self, tools: list[ResearchTool] | None = None) -> None:
        self._tools: dict[str, ResearchTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ResearchTool) -> None:
        self._tools[tool.spec.key] = tool

    def get(self, key: str) -> ResearchTool | None:
        return self._tools.get(key)

    def require(self, key: str) -> ResearchTool:
        tool = self.get(key)
        if tool is None:
            raise KeyError(f"Tool '{key}' is not registered.")
        return tool

    def tools(self) -> list[ResearchTool]:
        return list(self._tools.values())


@dataclass(frozen=True)
class ToolPermissionPolicy:
    permissions: dict[str, tuple[str, ...]]

    def allowed_tool_keys(self, role_key: str) -> tuple[str, ...]:
        return self.permissions.get(role_key, ())

    def can_use(self, role_key: str, tool_key: str) -> bool:
        return tool_key in self.allowed_tool_keys(role_key)


class ToolRuntime:
    def __init__(self, registry: ToolRegistry | None = None, permission_policy: ToolPermissionPolicy | None = None) -> None:
        self.registry = registry or default_tool_registry()
        self.permission_policy = permission_policy or default_tool_permission_policy()

    def available_tools_for_role(self, role_key: str) -> list[ToolSpec]:
        allowed = set(self.permission_policy.allowed_tool_keys(role_key))
        return [tool.spec for tool in self.registry.tools() if tool.spec.key in allowed]

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        tool = self.registry.get(request.tool_key)
        if tool is None:
            return ToolExecutionResult(
                tool_key=request.tool_key,
                status=TOOL_STATUS_FAILED,
                message="The requested tool is not registered.",
                error_message=f"Unknown tool: {request.tool_key}",
            )
        if not self.permission_policy.can_use(request.role_key, request.tool_key):
            return ToolExecutionResult(
                tool_key=request.tool_key,
                status=TOOL_STATUS_DENIED,
                message="The current role is not allowed to use this tool.",
                error_message=f"Role '{request.role_key}' cannot use tool '{request.tool_key}'.",
            )
        try:
            return tool.execute(request)
        except Exception as exc:
            return ToolExecutionResult(
                tool_key=request.tool_key,
                status=TOOL_STATUS_FAILED,
                message="The tool execution failed.",
                error_message=str(exc),
            )


@dataclass(frozen=True)
class MockArtifactExportTool:
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            key=MOCK_ARTIFACT_EXPORT_TOOL_KEY,
            name="Mock Artifact Export Tool",
            description="Create a mock markdown artifact without invoking any external runtime.",
            capability=ARTIFACT_EXPORT_TOOL_KEY,
            outputs_artifacts=True,
        )
    )

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        title = str(request.payload.get("title") or "Mock Artifact").strip() or "Mock Artifact"
        body = str(request.payload.get("body") or request.user_request or "").strip()
        artifact_type = str(request.payload.get("artifact_type") or "mock_export").strip() or "mock_export"
        path_hint = str(request.payload.get("path_hint") or f"artifacts/{_slugify(title)}.md").strip()
        content = f"# {title}\n\n{body}".strip()
        artifact = ToolArtifact(
            artifact_type=artifact_type,
            title=title,
            content=content,
            path_hint=path_hint,
            mime_type="text/markdown",
            metadata={"generator": self.spec.key},
        )
        return ToolExecutionResult(
            tool_key=self.spec.key,
            status=TOOL_STATUS_SUCCESS,
            message="Mock artifact export completed.",
            output_text=content,
            data={"artifact_count": 1, "capability": self.spec.capability},
            artifacts=(artifact,),
        )


@dataclass(frozen=True)
class PythonArtifactTool:
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            key=PYTHON_EXECUTION_TOOL_KEY,
            name="Python Draft Tool",
            description="Create a local Python analysis draft from the structured discussion state.",
            capability=PYTHON_EXECUTION_TOOL_KEY,
            outputs_artifacts=True,
        )
    )

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        title = str(request.payload.get("title") or "analysis_draft").strip() or "analysis_draft"
        topic = str(request.payload.get("topic") or request.user_request or "Research task").strip()
        summary = str(request.payload.get("summary") or "").strip()
        consensus = _coerce_lines(request.payload.get("consensus"))
        open_questions = _coerce_lines(request.payload.get("open_questions"))
        action_items = _coerce_lines(request.payload.get("action_items"))
        source_names = _coerce_lines(request.payload.get("source_names"))
        file_stem = _slugify(title)
        path_hint = str(request.payload.get("path_hint") or f"generated_artifacts/{file_stem}.py").strip()
        code = _render_python_artifact(
            topic=topic,
            summary=summary,
            consensus=consensus,
            open_questions=open_questions,
            action_items=action_items,
            source_names=source_names,
        )
        artifact = ToolArtifact(
            artifact_type="python_script",
            title=title,
            content=code,
            path_hint=path_hint,
            mime_type="text/x-python",
            metadata={"generator": self.spec.key},
        )
        return ToolExecutionResult(
            tool_key=self.spec.key,
            status=TOOL_STATUS_SUCCESS,
            message="Python draft artifact generated.",
            output_text=code,
            data={"artifact_count": 1, "capability": self.spec.capability},
            artifacts=(artifact,),
        )


@dataclass(frozen=True)
class LatexGenerationTool:
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            key=LATEX_GENERATION_TOOL_KEY,
            name="LaTeX Draft Tool",
            description="Create a LaTeX report draft from the structured discussion output.",
            capability=LATEX_GENERATION_TOOL_KEY,
            outputs_artifacts=True,
        )
    )

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        title = str(request.payload.get("title") or "research_report_draft").strip() or "research_report_draft"
        topic = str(request.payload.get("topic") or request.user_request or "Research task").strip()
        summary = str(request.payload.get("summary") or "").strip()
        minutes = str(request.payload.get("minutes") or "").strip()
        consensus = _coerce_lines(request.payload.get("consensus"))
        open_questions = _coerce_lines(request.payload.get("open_questions"))
        action_items = _coerce_lines(request.payload.get("action_items"))
        evidence_labels = _coerce_lines(request.payload.get("evidence_labels"))
        bibtex_keys = _coerce_lines(request.payload.get("bibtex_keys"))
        bibliography_basename = str(request.payload.get("bibliography_basename") or "references").strip() or "references"
        file_stem = _slugify(title)
        path_hint = str(request.payload.get("path_hint") or f"generated_artifacts/{file_stem}.tex").strip()
        latex = _render_latex_artifact(
            topic=topic,
            summary=summary,
            minutes=minutes,
            consensus=consensus,
            open_questions=open_questions,
            action_items=action_items,
            evidence_labels=evidence_labels,
            bibtex_keys=bibtex_keys,
            bibliography_basename=bibliography_basename,
        )
        artifact = ToolArtifact(
            artifact_type="latex_document",
            title=title,
            content=latex,
            path_hint=path_hint,
            mime_type="text/x-tex",
            metadata={"generator": self.spec.key},
        )
        return ToolExecutionResult(
            tool_key=self.spec.key,
            status=TOOL_STATUS_SUCCESS,
            message="LaTeX draft artifact generated.",
            output_text=latex,
            data={"artifact_count": 1, "capability": self.spec.capability},
            artifacts=(artifact,),
        )


@dataclass(frozen=True)
class BibtexGenerationTool:
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            key=BIBTEX_GENERATION_TOOL_KEY,
            name="BibTeX Library Tool",
            description="Create a BibTeX library artifact from discovered paper metadata.",
            capability=BIBTEX_GENERATION_TOOL_KEY,
            outputs_artifacts=True,
        )
    )

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        title = str(request.payload.get("title") or "references").strip() or "references"
        entries = _coerce_lines(request.payload.get("bibtex_entries"))
        if not entries:
            return ToolExecutionResult(
                tool_key=self.spec.key,
                status=TOOL_STATUS_FAILED,
                message="No BibTeX entries were supplied.",
                error_message="Missing bibtex_entries payload.",
            )
        file_stem = _slugify(title)
        path_hint = str(request.payload.get("path_hint") or f"generated_artifacts/{file_stem}.bib").strip()
        content = "\n".join(entry.rstrip() for entry in entries if entry.strip()).strip() + "\n"
        artifact = ToolArtifact(
            artifact_type="bibtex_library",
            title=title,
            content=content,
            path_hint=path_hint,
            mime_type="text/x-bibtex",
            metadata={"generator": self.spec.key},
        )
        return ToolExecutionResult(
            tool_key=self.spec.key,
            status=TOOL_STATUS_SUCCESS,
            message="BibTeX library artifact generated.",
            output_text=content,
            data={"artifact_count": 1, "capability": self.spec.capability},
            artifacts=(artifact,),
        )


def default_tool_permission_policy() -> ToolPermissionPolicy:
    return ToolPermissionPolicy(
        permissions={
            MODERATOR_ROLE_KEY: (ARTIFACT_EXPORT_TOOL_KEY,),
            NOTETAKER_ROLE_KEY: (ARTIFACT_EXPORT_TOOL_KEY, LATEX_GENERATION_TOOL_KEY, BIBTEX_GENERATION_TOOL_KEY, MOCK_ARTIFACT_EXPORT_TOOL_KEY),
            RESEARCH_ANALYST_ROLE_KEY: (PYTHON_EXECUTION_TOOL_KEY, FIGURE_GENERATION_TOOL_KEY, MOCK_ARTIFACT_EXPORT_TOOL_KEY),
            REVIEWER_ROLE_KEY: (ARTIFACT_EXPORT_TOOL_KEY, MOCK_ARTIFACT_EXPORT_TOOL_KEY),
        }
    )


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(tools=[MockArtifactExportTool(), PythonArtifactTool(), LatexGenerationTool(), BibtexGenerationTool()])


def default_tool_runtime() -> ToolRuntime:
    return ToolRuntime(registry=default_tool_registry(), permission_policy=default_tool_permission_policy())


def _slugify(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or "artifact"


def _coerce_lines(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]


def _render_python_artifact(
    *,
    topic: str,
    summary: str,
    consensus: list[str],
    open_questions: list[str],
    action_items: list[str],
    source_names: list[str],
) -> str:
    summary_block = summary or "Fill in the discussion summary here."
    consensus_block = "\n".join(f"    {json.dumps(item, ensure_ascii=False)}," for item in consensus) or f"    {json.dumps('No consensus captured yet.', ensure_ascii=False)},"
    open_questions_block = "\n".join(f"    {json.dumps(item, ensure_ascii=False)}," for item in open_questions) or f"    {json.dumps('No open questions recorded.', ensure_ascii=False)},"
    action_items_block = "\n".join(f"    {json.dumps(item, ensure_ascii=False)}," for item in action_items) or f"    {json.dumps('No action items recorded.', ensure_ascii=False)},"
    sources_block = "\n".join(f"    {json.dumps(item, ensure_ascii=False)}," for item in source_names) or f"    {json.dumps('No source attachments recorded.', ensure_ascii=False)},"
    topic_literal = json.dumps(topic, ensure_ascii=False)
    summary_literal = json.dumps(summary_block, ensure_ascii=False)
    return f'''# Python analysis scaffold generated by Cyber Colloquium.
# Task: {topic}

from __future__ import annotations

import json
import os
from pathlib import Path


CONSENSUS = [
{consensus_block}
]

OPEN_QUESTIONS = [
{open_questions_block}
]

ACTION_ITEMS = [
{action_items_block}
]

SOURCE_FILES = [
{sources_block}
]


def load_inputs() -> dict[str, object]:
    """Load mapped workspace inputs when available, otherwise fall back to metadata only."""
    manifest_path = os.environ.get("CYBER_COLLOQUIUM_INPUT_MANIFEST", "").strip()
    input_dir = os.environ.get("CYBER_COLLOQUIUM_INPUT_DIR", "").strip()
    run_mode = os.environ.get("CYBER_COLLOQUIUM_RUN_MODE", "manual").strip() or "manual"
    manifest: dict[str, object] = {{}}
    mapped_inputs: list[dict[str, object]] = []
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            mapped_inputs = list(manifest.get("mapped_inputs", [])) if isinstance(manifest, dict) else []
        except Exception:
            manifest = {{}}
            mapped_inputs = []
    return {{
        "sources": SOURCE_FILES,
        "mapped_inputs": mapped_inputs,
        "input_dir": input_dir,
        "input_manifest": manifest,
        "run_mode": run_mode,
        "consensus": CONSENSUS,
        "open_questions": OPEN_QUESTIONS,
        "action_items": ACTION_ITEMS,
    }}


def run_analysis(inputs: dict[str, object]) -> dict[str, object]:
    """Implement the computational workflow for this research task."""
    return {{
        "topic": {topic_literal},
        "summary": {summary_literal},
        "inputs": inputs,
        "results": [],
    }}


def render_summary(results: dict[str, object]) -> str:
    """Convert analysis outputs into a concise text report."""
    return "\\n".join(
        [
            f"Topic: {{results['topic']}}",
            f"Run mode: {{results['inputs'].get('run_mode', 'manual')}}",
            f"Tracked sources: {{len(SOURCE_FILES)}}",
            f"Consensus items: {{len(CONSENSUS)}}",
            f"Open questions: {{len(OPEN_QUESTIONS)}}",
            "TODO: replace this scaffold with domain-specific reporting.",
        ]
    )


def main() -> None:
    inputs = load_inputs()
    results = run_analysis(inputs)
    print(render_summary(results))


if __name__ == "__main__":
    main()
'''


def _render_latex_artifact(
    *,
    topic: str,
    summary: str,
    minutes: str,
    consensus: list[str],
    open_questions: list[str],
    action_items: list[str],
    evidence_labels: list[str],
    bibtex_keys: list[str],
    bibliography_basename: str,
) -> str:
    consensus_block = _latex_itemize(consensus or ["No consensus captured yet."])
    open_questions_block = _latex_itemize(open_questions or ["No open questions recorded."])
    action_items_block = _latex_itemize(action_items or ["No action items recorded."])
    evidence_block = _latex_itemize(evidence_labels or ["No evidence labels recorded."])
    bibliography_block = _latex_itemize([f"\\cite{{{key}}}" for key in bibtex_keys] or ["No external references were linked in this draft."])
    summary_block = _latex_escape(summary or "Fill in the final report here.")
    minutes_block = _latex_escape(minutes or "Add detailed meeting minutes here.")
    bibliography_footer = (
        f"\\bibliographystyle{{plain}}\n\\bibliography{{{_latex_escape(bibliography_basename)}}}\n" if bibtex_keys else ""
    )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\title{{{_latex_escape(topic)}}}
\author{{Cyber Colloquium}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Research Goal}}
{_latex_escape(topic)}

\section{{Executive Summary}}
{summary_block}

\section{{Consensus}}
{consensus_block}

\section{{Open Questions}}
{open_questions_block}

\section{{Action Items}}
{action_items_block}

\section{{Evidence Anchors}}
{evidence_block}

\section{{Reference Handles}}
{bibliography_block}

\section{{Meeting Notes Extract}}
{minutes_block}

\section{{Next Steps}}
Replace this draft with a polished manuscript structure, then expand the methods, experiments, and appendix sections as needed.

{bibliography_footer}
\end{{document}}
"""


def _latex_itemize(items: list[str]) -> str:
    body = "\n".join(f"    \\item {_latex_escape(item)}" for item in items)
    return "\\begin{itemize}\n" + body + "\n\\end{itemize}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    return escaped.replace("\n", "\n\n")
