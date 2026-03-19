from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.discussion_app.attachments import build_attachment_index
from src.discussion_app.models import (
    AttachmentPayload,
    DiscussionMessage,
    DiscussionResult,
    EXPERT_DUTY,
    EvidenceCard,
    REPORT_DUTY,
    ProviderConfig,
)
from src.discussion_app.state import DiscussionState
from src.discussion_app.workflow import WorkflowRuntimeContext
from src.discussion_app.workflow_config import workflow_config_from_dict


_pypdf_stub = types.ModuleType("pypdf")


class _PdfReader:
    def __init__(self, *args, **kwargs) -> None:
        self.pages = []


_pypdf_stub.PdfReader = _PdfReader
sys.modules.setdefault("pypdf", _pypdf_stub)

import src.discussion_app.orchestrator as orchestrator_module
from src.discussion_app.arxiv_client import ArxivPaper
from src.discussion_app.orchestrator import DiscussionOrchestrator, WorkPackage
from src.discussion_app.tool_runtime import ToolExecutionResult


def _provider(name: str, duty: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model="mock-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        duty=duty,
        specialty=f"{name} specialty",
    )


def _runtime_context(orchestrator: DiscussionOrchestrator) -> WorkflowRuntimeContext:
    project = orchestrator._initialize_project("Validate the runtime path.", [])
    result = DiscussionResult()
    result.research_project = project
    result.meeting_state = project.discussion_state
    return WorkflowRuntimeContext(
        user_request="Validate the runtime path.",
        attachments=[],
        max_rounds=1,
        generate_literature_review=False,
        local_execution_authorized=True,
        result=result,
        project=project,
        state=project.discussion_state,
        state_manager=orchestrator.state_manager,
    )


class OrchestratorExecutionTests(unittest.TestCase):
    def test_expand_literature_search_passes_artifact_stem_to_bibtex_tool(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [],
            workflow_config=workflow_config_from_dict(
                {
                    "tooling": {
                        "enable_arxiv_discovery": True,
                        "download_arxiv_pdfs": False,
                        "enable_bibtex_artifact": True,
                    }
                }
            ),
        )
        context = _runtime_context(orchestrator)
        captured: dict[str, object] = {}
        paper = ArxivPaper(
            paper_id="2401.12345v1",
            title="A Test Paper",
            abstract="A short abstract.",
            authors=("Alice", "Bob"),
            categories=("cs.AI",),
            published_at="2024-01-08T12:00:00Z",
            updated_at="2024-01-10T12:00:00Z",
            entry_url="https://arxiv.org/abs/2401.12345v1",
            pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
        )

        def fake_bibtex_tool(runtime_context, bib_entries, *, artifact_stem):
            captured["context"] = runtime_context
            captured["entries"] = list(bib_entries)
            captured["artifact_stem"] = artifact_stem
            return ToolExecutionResult(tool_key="bibtex_generation", status="failed", message="skip write in test")

        with mock.patch.object(
            orchestrator,
            "_plan_discussion_guided_search_queries",
            return_value=(["runtime path validation"], "fill evidence gaps", "Lead"),
        ):
            with mock.patch.object(orchestrator_module, "search_arxiv", return_value=[paper]):
                with mock.patch.object(orchestrator, "_execute_bibtex_artifact_tool", side_effect=fake_bibtex_tool):
                    note = orchestrator._workflow_stage_expand_literature_search(context, None)

        self.assertIn("added 1 paper", note.lower())
        self.assertEqual(len(context.state.literature_library), 1)
        self.assertEqual(captured["context"], context)
        self.assertTrue(captured["entries"])
        self.assertEqual(captured["artifact_stem"], "validate_the_runtime_path")

    def test_expand_literature_search_uses_discussion_guided_queries(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [],
            workflow_config=workflow_config_from_dict(
                {
                    "tooling": {
                        "enable_arxiv_discovery": True,
                        "download_arxiv_pdfs": False,
                    }
                }
            ),
        )
        context = _runtime_context(orchestrator)
        papers = [
            ArxivPaper(
                paper_id="2401.12345v1",
                title="Targeted Paper One",
                abstract="A short abstract.",
                authors=("Alice",),
                categories=("cs.CV",),
                published_at="2024-01-08T12:00:00Z",
                updated_at="2024-01-10T12:00:00Z",
                entry_url="https://arxiv.org/abs/2401.12345v1",
                pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
            ),
            ArxivPaper(
                paper_id="2402.54321v1",
                title="Targeted Paper Two",
                abstract="Another abstract.",
                authors=("Bob",),
                categories=("eess.IV",),
                published_at="2024-02-08T12:00:00Z",
                updated_at="2024-02-10T12:00:00Z",
                entry_url="https://arxiv.org/abs/2402.54321v1",
                pdf_url="https://arxiv.org/pdf/2402.54321v1.pdf",
            ),
        ]

        with mock.patch.object(
            orchestrator,
            "_plan_discussion_guided_search_queries",
            return_value=(["mamba computational imaging", "physics informed reconstruction"], "fill evidence gaps", "Lead"),
        ):
            with mock.patch.object(orchestrator_module, "search_arxiv", side_effect=[[papers[0]], [papers[1]]]):
                note = orchestrator._workflow_stage_expand_literature_search(context, None)

        self.assertIn("added 2 paper", note.lower())
        self.assertEqual([paper.paper_id for paper in context.state.literature_library], ["2401.12345v1", "2402.54321v1"])

    def test_response_needs_repair_rejects_invalid_evidence_ids(self) -> None:
        orchestrator = DiscussionOrchestrator([])
        content = (
            "[Judgment]\n接受。\n"
            "[Reasons]\n- 基本成立。\n"
            "[Evidence]\n- 引用了 E99。\n"
            "[Risk]\n- 仍需补充实验。\n"
            "[Handoff]\n- 请复核。"
        )

        self.assertTrue(
            orchestrator._response_needs_repair(
                content,
                ["[Judgment]", "[Reasons]", "[Evidence]", "[Risk]", "[Handoff]"],
                min_chars=30,
                allowed_evidence_ids={"E1", "E2"},
            )
        )

    def test_chat_with_sections_repairs_false_missing_review_input_claim(self) -> None:
        provider = _provider("Reviewer", EXPERT_DUTY)
        orchestrator = DiscussionOrchestrator([provider])
        missing_input_reply = (
            "[Verdict]\n输入缺失，无法执行复核。\n"
            "[Corrections]\n- 缺少待复核发言。\n"
            "[Evidence Check]\n- 未收到 Evidence ID。\n"
            "[Residual Risk]\n- 当前无法判断。"
        )
        repaired_reply = (
            "[Verdict]\n暂不接受上一条发言。\n"
            "[Corrections]\n- 关键结论缺少直接证据支撑。\n"
            "[Evidence Check]\n- E1 可以支撑背景，但核心主张仍缺补证。\n"
            "[Residual Risk]\n- 若不补证，结论会继续漂移。"
        )

        with mock.patch.object(orchestrator, "_chat", side_effect=[missing_input_reply, repaired_reply]):
            content = orchestrator._chat_with_sections(
                provider=provider,
                system_prompt="review system",
                user_prompt="review user prompt",
                max_tokens=300,
                required_sections=["[Verdict]", "[Corrections]", "[Evidence Check]", "[Residual Risk]"],
                min_chars=40,
                allowed_evidence_ids={"E1"},
                reject_missing_review_input=True,
            )

        self.assertEqual(content, repaired_reply)

    def test_augment_review_snippets_reuses_state_evidence_cards(self) -> None:
        provider = _provider("Reviewer", EXPERT_DUTY)
        orchestrator = DiscussionOrchestrator([provider])
        attachment = AttachmentPayload(
            path=Path("paper.txt"),
            display_name="paper.txt",
            kind="text",
            content=(
                "Mamba long-context stability depends on state drift control. "
                "Selective state updates can forget details when the sequence is extremely long. "
            )
            * 8,
        )
        orchestrator.attachment_index = build_attachment_index([attachment], chunk_chars=120, overlap_chars=0)
        anchor = orchestrator.attachment_index[0]
        state = DiscussionState(
            user_question="分析 Mamba 的长上下文故障模式",
            current_question="长上下文与可扩展性挑战的故障模式分析",
            evidence_cards=[
                EvidenceCard(
                    evidence_id=anchor.evidence_id,
                    summary="状态漂移与长上下文遗忘风险",
                    source="paper.txt | chunk 1",
                    attachment_name="paper.txt",
                    workpackage_index=5,
                )
            ],
        )
        workpackage = WorkPackage(
            index=5,
            title="长上下文与可扩展性挑战的故障模式分析",
            description="检查状态漂移、遗忘模式和稳定性边界",
            owner_name="Expert 4",
            reviewer_name="Reviewer",
        )
        previous_message = DiscussionMessage(
            speaker="Expert 4",
            role="assistant",
            content=f"请重点复核 {anchor.evidence_id} 所对应的状态漂移证据。",
            round_index=5,
            duty=EXPERT_DUTY,
            stage="analysis",
        )

        snippets = orchestrator._augment_review_snippets(
            [],
            state=state,
            workpackage=workpackage,
            previous_message=previous_message,
            provider=provider,
        )

        self.assertTrue(snippets)
        self.assertEqual(snippets[0].evidence_id, anchor.evidence_id)

    def test_run_reviewer_frontloads_previous_message_and_evidence_catalog(self) -> None:
        provider = _provider("Reviewer", EXPERT_DUTY)
        orchestrator = DiscussionOrchestrator([provider])
        attachment = AttachmentPayload(
            path=Path("paper.txt"),
            display_name="paper.txt",
            kind="text",
            content=("Mamba selective state space model long-context evidence. " * 20).strip(),
        )
        orchestrator.attachment_index = build_attachment_index([attachment], chunk_chars=120, overlap_chars=0)
        anchor = orchestrator.attachment_index[0]
        state = DiscussionState(
            user_question="分析 Mamba 的理论边界",
            current_question="长上下文与可扩展性挑战的故障模式分析",
            current_stage="Run Reviewer Pass",
            evidence_cards=[
                EvidenceCard(
                    evidence_id=anchor.evidence_id,
                    summary="长上下文证据锚点",
                    source="paper.txt | chunk 1",
                    attachment_name="paper.txt",
                    workpackage_index=5,
                )
            ],
        )
        workpackage = WorkPackage(
            index=5,
            title="长上下文与可扩展性挑战的故障模式分析",
            description="检查状态漂移、遗忘模式和稳定性边界",
            owner_name="Expert 4",
            reviewer_name="Reviewer",
        )
        previous_message = DiscussionMessage(
            speaker="Expert 4",
            role="assistant",
            content="上一条专家发言明确讨论了长上下文状态漂移和遗忘模式。",
            round_index=5,
            duty=EXPERT_DUTY,
            stage="analysis",
        )
        captured: dict[str, object] = {}

        def fake_chat_with_sections(**kwargs):
            captured.update(kwargs)
            return (
                "[Verdict]\n接受。\n"
                "[Corrections]\n- 无。\n"
                "[Evidence Check]\n- 证据足够。\n"
                "[Residual Risk]\n- 仍需补充更长序列实验。"
            )

        with mock.patch.object(orchestrator, "_chat_with_sections", side_effect=fake_chat_with_sections):
            orchestrator._run_reviewer(
                provider=provider,
                user_request="分析 Mamba 的理论边界",
                assignments_text="任务 5 关注长上下文与故障模式。",
                team_roster="- Reviewer | Expert reviewer",
                literature_review_text="已有综述指出长序列稳定性仍有空白。",
                workpackage=workpackage,
                previous_message=previous_message,
                state=state,
                relevant_snippets=[],
            )

        prompt = str(captured["user_prompt"])
        self.assertIn("待复核发言", prompt)
        self.assertIn(anchor.evidence_id, prompt)
        self.assertLess(prompt.index("待复核发言"), prompt.index("团队成员简表"))
        self.assertTrue(bool(captured["reject_missing_review_input"]))

    def test_attachment_index_refresh_preserves_existing_evidence_ids(self) -> None:
        base_attachment = AttachmentPayload(
            path=Path("paper1.txt"),
            display_name="paper1.txt",
            kind="text",
            content=("alpha " * 40).strip(),
        )
        new_attachment = AttachmentPayload(
            path=Path("paper2.txt"),
            display_name="paper2.txt",
            kind="text",
            content=("beta " * 40).strip(),
        )

        initial = build_attachment_index([base_attachment], chunk_chars=60, overlap_chars=0)
        refreshed = build_attachment_index([base_attachment, new_attachment], chunk_chars=60, overlap_chars=0, existing_snippets=initial)

        self.assertGreater(len(initial), 1)
        self.assertEqual(
            [snippet.evidence_id for snippet in refreshed if snippet.attachment_name == "paper1.txt"],
            [snippet.evidence_id for snippet in initial],
        )

    def test_python_smoke_test_runs_inside_isolated_workspace(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Analyst", EXPERT_DUTY),
                _provider("Reporter", REPORT_DUTY),
            ],
            workflow_config=workflow_config_from_dict(
                {
                    "tooling": {
                        "python_execution_timeout_seconds": 33,
                        "python_workspace_input_limit_mb": 1,
                    }
                }
            ),
        )
        context = _runtime_context(orchestrator)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_script = temp_path / "analysis_draft.py"
            source_script.write_text("print('hello from isolated workspace')\n", encoding="utf-8")
            mapped_source = temp_path / "input.csv"
            mapped_source.write_text("col\n1\n", encoding="utf-8")
            context.attachments = [
                AttachmentPayload(path=mapped_source, display_name="input.csv", kind="text", content="col\n1\n")
            ]
            calls: list[tuple[list[str], Path, int]] = []
            envs: list[dict[str, str] | None] = []

            def fake_run(
                command: list[str],
                *,
                cwd: Path,
                timeout_seconds: int = 20,
                env: dict[str, str] | None = None,
            ) -> dict[str, str | int]:
                calls.append((list(command), Path(cwd), timeout_seconds))
                envs.append(env)
                return {"returncode": 0, "stdout": "ok", "stderr": ""}

            with mock.patch.object(orchestrator_module, "PYTHON_EXECUTION_RUNS_DIR", temp_path / "execution_runs"):
                with mock.patch.object(orchestrator, "_run_subprocess", side_effect=fake_run):
                        outputs, passed = orchestrator._run_python_smoke_test(context, source_script)

            self.assertEqual(len(calls), 2)
            self.assertTrue(passed)
            experiment_run = context.state.experiment_runs[-1]
            workspace = Path(experiment_run.working_directory)
            self.assertTrue(workspace.exists())
            self.assertNotEqual(workspace, source_script.parent)
            self.assertTrue((workspace / source_script.name).exists())
            manifest_path = workspace / "input_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mapped_input_count"], 1)
            self.assertTrue((workspace / "inputs" / "input.csv").exists())
            self.assertEqual(experiment_run.script_path, str(workspace / source_script.name))
            self.assertEqual(experiment_run.interpreter_path, sys.executable)
            self.assertEqual(experiment_run.run_mode, "smoke")
            self.assertTrue(all(cwd == workspace for _, cwd, _ in calls))
            self.assertEqual(calls[0][0], [sys.executable, "-m", "py_compile", source_script.name])
            self.assertEqual(calls[1][0], [sys.executable, source_script.name])
            self.assertTrue(all(timeout == 33 for _, _, timeout in calls))
            self.assertEqual(envs[0]["CYBER_COLLOQUIUM_INPUT_MANIFEST"], str(manifest_path))
            self.assertEqual(envs[0]["CYBER_COLLOQUIUM_INPUT_DIR"], str(workspace / "inputs"))
            self.assertEqual(envs[0]["CYBER_COLLOQUIUM_RUN_MODE"], "smoke")
            self.assertEqual(Path(outputs[0]).parent, workspace)
            self.assertEqual(Path(outputs[1]).parent, workspace)

    def test_python_full_run_uses_current_interpreter_and_full_mode(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Analyst", EXPERT_DUTY),
                _provider("Reporter", REPORT_DUTY),
            ],
            workflow_config=workflow_config_from_dict(
                {
                    "tooling": {
                        "python_full_execution_timeout_seconds": 120,
                        "python_workspace_input_limit_mb": 1,
                    }
                }
            ),
        )
        context = _runtime_context(orchestrator)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_script = temp_path / "analysis_draft.py"
            source_script.write_text("print('full run')\n", encoding="utf-8")
            envs: list[dict[str, str] | None] = []
            calls: list[tuple[list[str], Path, int]] = []

            def fake_run(
                command: list[str],
                *,
                cwd: Path,
                timeout_seconds: int = 20,
                env: dict[str, str] | None = None,
            ) -> dict[str, str | int]:
                calls.append((list(command), Path(cwd), timeout_seconds))
                envs.append(env)
                return {"returncode": 0, "stdout": "ok", "stderr": ""}

            with mock.patch.object(orchestrator_module, "PYTHON_EXECUTION_RUNS_DIR", temp_path / "execution_runs"):
                with mock.patch.object(orchestrator, "_run_subprocess", side_effect=fake_run):
                    outputs, passed = orchestrator._run_python_full_execution(context, source_script)

            self.assertTrue(passed)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(timeout == 120 for _, _, timeout in calls))
            self.assertEqual(envs[0]["CYBER_COLLOQUIUM_RUN_MODE"], "full")
            self.assertEqual(envs[0]["CYBER_COLLOQUIUM_FULL_RUN"], "1")
            self.assertEqual(envs[0]["CYBER_COLLOQUIUM_SMOKE_TEST"], "0")
            self.assertEqual(context.state.experiment_runs[-1].run_mode, "full")
            self.assertEqual(context.state.experiment_runs[-1].interpreter_path, sys.executable)
            self.assertTrue(any(str(path).endswith("_full_run_log.txt") for path in outputs))

    def test_latex_compile_uses_tectonic_and_separate_build_directory(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Analyst", EXPERT_DUTY),
                _provider("Reporter", REPORT_DUTY),
            ],
            workflow_config=workflow_config_from_dict({}),
        )
        context = _runtime_context(orchestrator)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "generated_artifacts"
            source_dir.mkdir(parents=True, exist_ok=True)
            tex_path = source_dir / "paper_draft.tex"
            tex_path.write_text("\\documentclass{article}\\begin{document}Hello\\end{document}\n", encoding="utf-8")
            bib_path = source_dir / "paper_draft_references.bib"
            bib_path.write_text("@article{ref,title={Reference}}\n", encoding="utf-8")
            orchestrator.state_manager.record_artifact(
                context.state,
                artifact_type="bibtex_library",
                title="references",
                path=str(bib_path),
                preview="refs",
            )
            calls: list[tuple[list[str], Path, int]] = []

            def fake_run(command: list[str], *, cwd: Path, timeout_seconds: int = 20) -> dict[str, str | int]:
                calls.append((list(command), Path(cwd), timeout_seconds))
                outdir = Path(command[command.index("--outdir") + 1])
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / f"{tex_path.stem}.pdf").write_bytes(b"%PDF-1.4\n")
                (outdir / f"{tex_path.stem}.log").write_text("Tectonic engine log", encoding="utf-8")
                return {"returncode": 0, "stdout": "build ok", "stderr": ""}

            with mock.patch.object(orchestrator_module, "LATEX_BUILD_RUNS_DIR", temp_path / "latex_builds"):
                with mock.patch("src.discussion_app.orchestrator.shutil.which", side_effect=lambda name: "C:/tectonic/tectonic.exe" if name == "tectonic" else None):
                    with mock.patch.object(orchestrator, "_run_subprocess", side_effect=fake_run):
                        outputs = orchestrator._compile_latex_artifact(context, tex_path)

            self.assertEqual(len(calls), 1)
            command, cwd, timeout_seconds = calls[0]
            self.assertEqual(command[0], "C:/tectonic/tectonic.exe")
            self.assertIn("--outdir", command)
            self.assertIn("--untrusted", command)
            self.assertEqual(cwd, tex_path.parent)
            self.assertEqual(timeout_seconds, 90)
            build_dir = Path(command[command.index("--outdir") + 1])
            self.assertTrue(build_dir.exists())
            self.assertNotEqual(build_dir, tex_path.parent)
            self.assertEqual({path.suffix for path in outputs}, {".txt", ".pdf"})
            self.assertTrue(any(artifact.metadata.get("compiler") == "tectonic" for artifact in context.state.generated_artifacts))
            self.assertTrue(context.state.approval_records[-1].scope.endswith(":tectonic"))


if __name__ == "__main__":
    unittest.main()
