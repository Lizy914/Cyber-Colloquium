from __future__ import annotations

import sys
import types
import unittest

from src.discussion_app.models import EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, LITERATURE_DUTY, REPORT_DUTY, ProviderConfig
from src.discussion_app.state import DiscussionState, WorkflowTask
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


class OrchestratorConfigTests(unittest.TestCase):
    def test_reviewer_role_can_be_disabled_from_workflow_config(self) -> None:
        config = workflow_config_from_dict(
            {
                "discussion": {
                    "enable_reviewer_role": False,
                }
            }
        )
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Lead", LEAD_DUTY),
                _provider("Expert A", EXPERT_DUTY),
                _provider("Expert B", EXPERT_DUTY),
            ],
            workflow_config=config,
        )

        reviewer = orchestrator._resolve_reviewer("Expert B", orchestrator.expert_providers[0])

        self.assertIsNone(reviewer)

    def test_team_role_order_and_filters_follow_workflow_config(self) -> None:
        config = workflow_config_from_dict(
            {
                "routing": {
                    "default_roles": [REPORT_DUTY, LEAD_DUTY, EXPERT_DUTY],
                },
                "team_roles": [
                    {
                        "duty": LITERATURE_DUTY,
                        "enabled": False,
                    }
                ],
            }
        )
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Lead", LEAD_DUTY),
                _provider("Host", HOST_DUTY),
                _provider("Reviewer", LITERATURE_DUTY),
                _provider("Reporter", REPORT_DUTY),
                _provider("Expert A", EXPERT_DUTY),
            ],
            workflow_config=config,
        )

        roster = orchestrator._build_team_roster().splitlines()

        self.assertTrue(roster[0].startswith("Team template:"))
        self.assertTrue(roster[1].startswith("- Reporter"))
        self.assertTrue(roster[2].startswith("- Lead"))
        self.assertNotIn("Reviewer", "\n".join(roster))
        self.assertTrue(all(provider.duty != LITERATURE_DUTY for provider in orchestrator.enabled_providers))

    def test_state_snapshot_respects_summary_slots(self) -> None:
        config = workflow_config_from_dict(
            {
                "context": {
                    "summary_slots": ["consensus"],
                }
            }
        )
        orchestrator = DiscussionOrchestrator([], workflow_config=config)
        state = DiscussionState(
            topic="Topic",
            user_question="Question",
            goal="Goal",
            current_stage="Stage",
            current_question="Question",
            rules=["Rule 1"],
            consensus_points=["Consensus item"],
            risks_or_disagreements=["Conflict item"],
            open_questions=["Open question item"],
            action_items=["Action item"],
        )

        snapshot = orchestrator._build_state_snapshot(state, mode="report")

        self.assertIn("Stable Consensus", snapshot)
        self.assertNotIn("Active Conflicts", snapshot)
        self.assertNotIn("Open Questions", snapshot)
        self.assertNotIn("Action Items", snapshot)

    def test_host_duty_is_preferred_for_host_actions(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Lead", LEAD_DUTY),
                _provider("Host", HOST_DUTY),
            ],
            workflow_config=workflow_config_from_dict({}),
        )

        selected = orchestrator._team_member_for_action("coordinate_workflow", fallback_duty=HOST_DUTY)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.provider.name, "Host")

    def test_checkpoint_frequency_is_config_driven(self) -> None:
        config = workflow_config_from_dict(
            {
                "discussion": {
                    "checkpoint_every_n_rounds": 2,
                }
            }
        )
        orchestrator = DiscussionOrchestrator([], workflow_config=config)
        project = orchestrator._initialize_project("Question", [])
        state = project.discussion_state

        first = orchestrator._maybe_create_checkpoint(
            state,
            label="Task 1",
            workpackage_index=1,
            completed_rounds=1,
        )
        second = orchestrator._maybe_create_checkpoint(
            state,
            label="Task 2",
            workpackage_index=2,
            completed_rounds=2,
        )

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(state.checkpoints), 1)

    def test_followup_workpackages_use_sequential_rounds_and_prioritize_blockers(self) -> None:
        orchestrator = DiscussionOrchestrator(
            [
                _provider("Lead", LEAD_DUTY),
                _provider("Host", HOST_DUTY),
                _provider("Reviewer", LITERATURE_DUTY),
                _provider("Expert A", EXPERT_DUTY),
            ],
            workflow_config=workflow_config_from_dict({}),
        )
        state = DiscussionState(
            current_round=5,
            workflow_tasks=[
                WorkflowTask(task_id="task_1", title="Main 1", round_index=1, source_kind="assignment"),
                WorkflowTask(task_id="task_5", title="Main 5", round_index=5, source_kind="assignment"),
            ],
            risks_or_disagreements=[
                "证据引用异常：当前 Evidence ID 不存在，导致 reviewer 无法继续。",
                "某专家发言偏离任务边界，需要收束。",
            ],
            open_questions=[
                "还缺少 benchmark 量化对比。"
            ],
        )

        followups = orchestrator._build_followup_workpackages(state, {})

        self.assertEqual([item.index for item in followups], [6, 7])
        self.assertEqual(followups[0].title, "Evidence Gap Closure")
        self.assertEqual(followups[1].title, "Scope Correction")


if __name__ == "__main__":
    unittest.main()
