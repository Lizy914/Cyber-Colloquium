from __future__ import annotations

import sys
import types
import unittest

from src.discussion_app.models import EXPERT_DUTY, REPORT_DUTY, ProviderConfig
from src.discussion_app.roles import MODERATOR_ROLE_KEY, NOTETAKER_ROLE_KEY, RESEARCH_ANALYST_ROLE_KEY
from src.discussion_app.tool_runtime import (
    BIBTEX_GENERATION_TOOL_KEY,
    LATEX_GENERATION_TOOL_KEY,
    MOCK_ARTIFACT_EXPORT_TOOL_KEY,
    PYTHON_EXECUTION_TOOL_KEY,
    TOOL_STATUS_DENIED,
    TOOL_STATUS_FAILED,
    TOOL_STATUS_SUCCESS,
    ToolExecutionRequest,
    default_tool_runtime,
)
from src.discussion_app.workflow_config import workflow_config_from_dict


_pypdf_stub = types.ModuleType("pypdf")


class _PdfReader:
    def __init__(self, *args, **kwargs) -> None:
        self.pages = []


_pypdf_stub.PdfReader = _PdfReader
sys.modules.setdefault("pypdf", _pypdf_stub)

from src.discussion_app.orchestrator import DiscussionOrchestrator


def _provider(name: str, duty: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model="mock-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        duty=duty,
        specialty=f"{name} specialty",
    )


class ToolRuntimeTests(unittest.TestCase):
    def test_mock_tool_executes_for_allowed_role(self) -> None:
        runtime = default_tool_runtime()

        result = runtime.execute(
            ToolExecutionRequest(
                tool_key=MOCK_ARTIFACT_EXPORT_TOOL_KEY,
                role_key=NOTETAKER_ROLE_KEY,
                user_request="Export the current meeting minutes.",
                payload={
                    "title": "Meeting Notes Draft",
                    "body": "Structured notes go here.",
                    "path_hint": "artifacts/meeting_notes_draft.md",
                    "artifact_type": "meeting_minutes",
                },
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, TOOL_STATUS_SUCCESS)
        self.assertEqual(result.tool_key, MOCK_ARTIFACT_EXPORT_TOOL_KEY)
        self.assertIn("# Meeting Notes Draft", result.output_text)
        self.assertEqual(result.artifacts[0].artifact_type, "meeting_minutes")
        self.assertEqual(result.artifacts[0].path_hint, "artifacts/meeting_notes_draft.md")

    def test_tool_runtime_denies_disallowed_role(self) -> None:
        runtime = default_tool_runtime()

        result = runtime.execute(
            ToolExecutionRequest(
                tool_key=MOCK_ARTIFACT_EXPORT_TOOL_KEY,
                role_key=MODERATOR_ROLE_KEY,
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, TOOL_STATUS_DENIED)
        self.assertIn("cannot use tool", result.error_message)

    def test_runtime_exposes_future_permissions_but_only_lists_registered_tools(self) -> None:
        runtime = default_tool_runtime()

        allowed = runtime.permission_policy.allowed_tool_keys(RESEARCH_ANALYST_ROLE_KEY)
        available = [spec.key for spec in runtime.available_tools_for_role(RESEARCH_ANALYST_ROLE_KEY)]

        self.assertIn(PYTHON_EXECUTION_TOOL_KEY, allowed)
        self.assertEqual(available, [MOCK_ARTIFACT_EXPORT_TOOL_KEY, PYTHON_EXECUTION_TOOL_KEY])

    def test_python_tool_generates_script_artifact(self) -> None:
        runtime = default_tool_runtime()

        result = runtime.execute(
            ToolExecutionRequest(
                tool_key=PYTHON_EXECUTION_TOOL_KEY,
                role_key=RESEARCH_ANALYST_ROLE_KEY,
                user_request="Draft a Python scaffold for the experiment.",
                payload={"title": "experiment_scaffold", "summary": "Build the evaluation loop."},
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.artifacts[0].artifact_type, "python_script")
        self.assertTrue(result.artifacts[0].path_hint.endswith(".py"))
        self.assertIn("def run_analysis", result.output_text)

    def test_latex_tool_generates_tex_artifact(self) -> None:
        runtime = default_tool_runtime()

        result = runtime.execute(
            ToolExecutionRequest(
                tool_key=LATEX_GENERATION_TOOL_KEY,
                role_key=NOTETAKER_ROLE_KEY,
                user_request="Draft the paper skeleton.",
                payload={"title": "paper_draft", "summary": "A concise summary."},
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.artifacts[0].artifact_type, "latex_document")
        self.assertTrue(result.artifacts[0].path_hint.endswith(".tex"))
        self.assertIn("\\documentclass", result.output_text)

    def test_bibtex_tool_generates_bib_artifact(self) -> None:
        runtime = default_tool_runtime()

        result = runtime.execute(
            ToolExecutionRequest(
                tool_key=BIBTEX_GENERATION_TOOL_KEY,
                role_key=NOTETAKER_ROLE_KEY,
                user_request="Create a BibTeX library for the discovered papers.",
                payload={
                    "title": "references",
                    "bibtex_entries": [
                        "@article{smith2024paper,\n  title = {Paper Title}\n}",
                    ],
                },
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.artifacts[0].artifact_type, "bibtex_library")
        self.assertTrue(result.artifacts[0].path_hint.endswith(".bib"))
        self.assertIn("@article{smith2024paper", result.output_text)

    def test_unknown_tool_returns_failed_result(self) -> None:
        runtime = default_tool_runtime()

        result = runtime.execute(
            ToolExecutionRequest(
                tool_key="missing_tool",
                role_key=NOTETAKER_ROLE_KEY,
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, TOOL_STATUS_FAILED)
        self.assertIn("Unknown tool", result.error_message)

    def test_orchestrator_exposes_default_tool_runtime_as_extension_point(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Analyst", EXPERT_DUTY),
                _provider("Reporter", REPORT_DUTY),
            ],
            workflow_config=workflow_config_from_dict({}),
        )

        analyst_tools = [spec.key for spec in orchestrator.tool_runtime.available_tools_for_role(orchestrator.team.primary_member_for_duty(EXPERT_DUTY).role_key)]
        reporter_tools = [spec.key for spec in orchestrator.tool_runtime.available_tools_for_role(orchestrator.team.primary_member_for_duty(REPORT_DUTY).role_key)]

        self.assertEqual(analyst_tools, [MOCK_ARTIFACT_EXPORT_TOOL_KEY, PYTHON_EXECUTION_TOOL_KEY])
        self.assertEqual(reporter_tools, [MOCK_ARTIFACT_EXPORT_TOOL_KEY, LATEX_GENERATION_TOOL_KEY, BIBTEX_GENERATION_TOOL_KEY])


if __name__ == "__main__":
    unittest.main()
