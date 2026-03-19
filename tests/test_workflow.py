from __future__ import annotations

import sys
import types
import unittest

from src.discussion_app.workflow_config import workflow_config_from_dict


_pypdf_stub = types.ModuleType("pypdf")


class _PdfReader:
    def __init__(self, *args, **kwargs) -> None:
        self.pages = []


_pypdf_stub.PdfReader = _PdfReader
sys.modules.setdefault("pypdf", _pypdf_stub)

from src.discussion_app.orchestrator import DiscussionOrchestrator


class WorkflowExecutionTests(unittest.TestCase):
    def test_orchestrator_routes_stage_execution_through_workflow_template(self) -> None:
        config = workflow_config_from_dict(
            {
                "workflow_template": {
                    "stages": [
                        {"key": "discover_literature", "enabled": True},
                        {"key": "run_team_discussion", "enabled": True},
                        {"key": "ingest_source_material", "enabled": True},
                        {"key": "run_reviewer_pass", "enabled": False},
                        {"key": "update_structured_state", "enabled": True},
                        {"key": "run_experiment_cycle", "enabled": True},
                        {"key": "generate_meeting_notes", "enabled": True},
                        {"key": "generate_research_report", "enabled": True},
                        {"key": "compile_latex_artifacts", "enabled": True},
                    ]
                }
            }
        )
        orchestrator = DiscussionOrchestrator([], workflow_config=config)
        stage_order: list[str] = []

        def _bind_stage(stage_key: str):
            def _handler(self, context, stage):
                del stage
                stage_order.append(stage_key)
                if stage_key == "generate_meeting_notes":
                    context.result.meeting_minutes = "minutes"
                if stage_key == "generate_research_report":
                    context.result.final_summary = "report"
                return f"{stage_key} complete"

            return types.MethodType(_handler, orchestrator)

        for stage_key in [
            "discover_literature",
            "run_team_discussion",
            "ingest_source_material",
            "expand_literature_search",
            "run_reviewer_pass",
            "update_structured_state",
            "run_experiment_cycle",
            "generate_meeting_notes",
            "generate_research_report",
            "compile_latex_artifacts",
        ]:
            setattr(orchestrator, f"_workflow_stage_{stage_key}", _bind_stage(stage_key))

        result = orchestrator.run_discussion("Question", [], rounds=1)

        self.assertIsNotNone(result.meeting_state)
        self.assertEqual(
            stage_order,
            [
                "discover_literature",
                "run_team_discussion",
                "ingest_source_material",
                "expand_literature_search",
                "update_structured_state",
                "run_experiment_cycle",
                "generate_meeting_notes",
                "generate_research_report",
                "compile_latex_artifacts",
            ],
        )
        self.assertEqual(
            [record.stage_key for record in result.meeting_state.workflow_stage_records],
            stage_order,
        )
        self.assertTrue(all(record.status == "completed" for record in result.meeting_state.workflow_stage_records))
        self.assertEqual(result.meeting_state.current_stage, "Compile LaTeX Artifacts")

    def test_stop_request_skips_optional_tool_stage_but_keeps_final_exports(self) -> None:
        orchestrator = DiscussionOrchestrator([], workflow_config=workflow_config_from_dict({}))
        stage_order: list[str] = []

        def _bind_stage(stage_key: str):
            def _handler(self, context, stage):
                del stage
                stage_order.append(stage_key)
                if stage_key == "generate_meeting_notes":
                    context.result.meeting_minutes = "minutes"
                if stage_key == "generate_research_report":
                    context.result.final_summary = "report"
                return f"{stage_key} complete"

            return types.MethodType(_handler, orchestrator)

        for stage_key in [
            "discover_literature",
            "ingest_source_material",
            "run_team_discussion",
            "expand_literature_search",
            "run_reviewer_pass",
            "update_structured_state",
            "run_experiment_cycle",
            "generate_meeting_notes",
            "generate_research_report",
            "compile_latex_artifacts",
        ]:
            setattr(orchestrator, f"_workflow_stage_{stage_key}", _bind_stage(stage_key))

        result = orchestrator.run_discussion(
            "Question",
            [],
            rounds=1,
            should_cancel=lambda: len(stage_order) >= 6,
        )

        self.assertEqual(
            stage_order,
            [
                "discover_literature",
                "ingest_source_material",
                "run_team_discussion",
                "expand_literature_search",
                "run_reviewer_pass",
                "update_structured_state",
                "generate_meeting_notes",
                "generate_research_report",
            ],
        )
        self.assertEqual(result.meeting_state.workflow_stage_records[6].stage_key, "run_experiment_cycle")
        self.assertEqual(result.meeting_state.workflow_stage_records[6].status, "cancelled")
        self.assertEqual(result.meeting_state.workflow_stage_records[7].stage_key, "generate_meeting_notes")
        self.assertEqual(result.final_summary, "report")
        self.assertEqual(result.meeting_minutes, "minutes")

    def test_real_workflow_produces_state_backed_outputs_without_providers(self) -> None:
        orchestrator = DiscussionOrchestrator([], workflow_config=workflow_config_from_dict({}))

        result = orchestrator.run_discussion(
            "Investigate robustness gaps in the proposed method.",
            [],
            rounds=2,
            generate_literature_review=False,
        )

        self.assertFalse(result.cancelled)
        self.assertIsNotNone(result.research_project)
        self.assertIsNotNone(result.meeting_state)
        self.assertIn("# Research Report", result.final_summary)
        self.assertIn("## Meeting Status", result.meeting_minutes)
        self.assertIn("## Workflow Stage Trace", result.meeting_minutes)
        self.assertEqual(
            [record.stage_key for record in result.meeting_state.workflow_stage_records],
            [stage.key for stage in orchestrator.workflow_config.workflow_template.enabled_stages()],
        )
        self.assertTrue(all(record.status == "completed" for record in result.meeting_state.workflow_stage_records))
        self.assertEqual(result.meeting_state.current_stage, "Compile LaTeX Artifacts")
        self.assertEqual(result.meeting_state.summary, result.final_summary)


if __name__ == "__main__":
    unittest.main()
