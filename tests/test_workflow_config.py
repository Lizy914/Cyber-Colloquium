from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.discussion_app.models import EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, REPORT_DUTY
from src.discussion_app.workflow_config import load_workflow_config, workflow_config_from_dict


class WorkflowConfigLoadingTests(unittest.TestCase):
    def test_missing_file_creates_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow_config.json"

            config = load_workflow_config(path)

            self.assertTrue(path.exists())
            self.assertEqual(config.discussion.max_rounds, 8)
            self.assertEqual(config.routing.default_roles[0], LEAD_DUTY)
            self.assertTrue(config.notes.include_role_labels)
            self.assertFalse(config.tooling.enable_arxiv_discovery)
            self.assertTrue(config.tooling.download_arxiv_pdfs)
            self.assertEqual(config.tooling.arxiv_max_results, 3)
            self.assertFalse(config.tooling.enable_python_artifact)
            self.assertFalse(config.tooling.enable_latex_artifact)
            self.assertFalse(config.tooling.enable_bibtex_artifact)
            self.assertFalse(config.tooling.enable_python_execution_test)
            self.assertFalse(config.tooling.enable_python_full_execution)
            self.assertEqual(config.tooling.python_execution_timeout_seconds, 20)
            self.assertEqual(config.tooling.python_full_execution_timeout_seconds, 300)
            self.assertEqual(config.tooling.python_workspace_input_limit_mb, 64)
            self.assertFalse(config.tooling.enable_latex_compile)
            self.assertTrue(config.is_role_enabled(HOST_DUTY))
            self.assertEqual(config.team_template.name, "Cyber Colloquium Basic Team")
            self.assertEqual(config.role_config(LEAD_DUTY).role_key, "moderator")

    def test_invalid_values_fall_back_and_partial_team_roles_merge(self) -> None:
        payload = {
            "discussion": {
                "max_rounds": -1,
                "enable_reviewer_role": "no",
            },
            "routing": {
                "default_roles": [REPORT_DUTY, "Invalid Duty", LEAD_DUTY],
                "parallel_roles": [EXPERT_DUTY, "Bad"],
            },
            "context": {
                "max_history_items": "7",
                "summary_slots": ["conflicts", "bad-slot", "action_items"],
            },
            "report": {
                "include_action_items": "off",
            },
            "notes": {
                "include_role_labels": 0,
            },
            "tooling": {
                "enable_arxiv_discovery": True,
                "download_arxiv_pdfs": False,
                "arxiv_max_results": 6,
                "enable_python_artifact": 1,
                "enable_latex_artifact": "yes",
                "enable_bibtex_artifact": True,
                "enable_python_execution_test": True,
                "enable_python_full_execution": True,
                "python_execution_timeout_seconds": 55,
                "python_full_execution_timeout_seconds": 240,
                "python_workspace_input_limit_mb": 96,
                "enable_latex_compile": True,
            },
            "team_roles": [
                {
                    "duty": EXPERT_DUTY,
                    "label": "Analyst",
                    "enabled": False,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "workflow_config.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            config = load_workflow_config(path)

            self.assertEqual(config.discussion.max_rounds, 8)
            self.assertFalse(config.discussion.enable_reviewer_role)
            self.assertEqual(config.routing.default_roles, [REPORT_DUTY, LEAD_DUTY])
            self.assertEqual(config.routing.parallel_roles, [EXPERT_DUTY])
            self.assertEqual(config.context.max_history_items, 7)
            self.assertEqual(config.context.summary_slots, ["conflicts", "action_items"])
            self.assertFalse(config.report.include_action_items)
            self.assertFalse(config.notes.include_role_labels)
            self.assertTrue(config.tooling.enable_arxiv_discovery)
            self.assertFalse(config.tooling.download_arxiv_pdfs)
            self.assertEqual(config.tooling.arxiv_max_results, 6)
            self.assertTrue(config.tooling.enable_python_artifact)
            self.assertTrue(config.tooling.enable_latex_artifact)
            self.assertTrue(config.tooling.enable_bibtex_artifact)
            self.assertTrue(config.tooling.enable_python_execution_test)
            self.assertTrue(config.tooling.enable_python_full_execution)
            self.assertEqual(config.tooling.python_execution_timeout_seconds, 55)
            self.assertEqual(config.tooling.python_full_execution_timeout_seconds, 240)
            self.assertEqual(config.tooling.python_workspace_input_limit_mb, 96)
            self.assertTrue(config.tooling.enable_latex_compile)
            self.assertFalse(config.role_config(EXPERT_DUTY).enabled)
            self.assertTrue(config.role_config(HOST_DUTY).enabled)
            self.assertEqual(config.role_config(EXPERT_DUTY).role_key, "research_analyst")

    def test_legacy_stage_order_inserts_discussion_guided_search_before_reviewer(self) -> None:
        config = workflow_config_from_dict(
            {
                "workflow_template": {
                    "stages": [
                        {"key": "discover_literature", "enabled": True},
                        {"key": "ingest_source_material", "enabled": True},
                        {"key": "run_team_discussion", "enabled": True},
                        {"key": "run_reviewer_pass", "enabled": True},
                        {"key": "update_structured_state", "enabled": True},
                    ]
                }
            }
        )

        self.assertEqual(
            [stage.key for stage in config.workflow_template.stages[:5]],
            [
                "discover_literature",
                "ingest_source_material",
                "run_team_discussion",
                "expand_literature_search",
                "run_reviewer_pass",
            ],
        )


if __name__ == "__main__":
    unittest.main()
