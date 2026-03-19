from __future__ import annotations

import unittest

from src.discussion_app.meeting_minutes import _render_state_sections, _render_transcript
from src.discussion_app.models import DiscussionMessage
from src.discussion_app.state import ApprovalRecord, Checkpoint, DiscussionState, ExperimentRunRecord, PaperRecord, WorkflowStageRecord, WorkflowTask


class MeetingMinutesConfigTests(unittest.TestCase):
    def test_report_state_sections_can_hide_optional_blocks(self) -> None:
        state = DiscussionState(
            topic="Topic",
            user_question="Question",
            goal="Goal",
            current_round=1,
            current_stage="Complete",
            consensus_points=["Consensus 1"],
            risks_or_disagreements=["Conflict 1"],
            open_questions=["Question 1"],
            action_items=["Action 1"],
            checkpoints=[Checkpoint(checkpoint_id="CP1", label="Init", workpackage_index=0, round_index=1, summary="Ready")],
            literature_library=[PaperRecord(paper_id="2401.12345", title="Paper A", authors=["Alice"])],
            experiment_runs=[
                ExperimentRunRecord(
                    run_id="run_1",
                    script_path="draft.py",
                    working_directory="generated_artifacts/execution_runs/project_1/run_1",
                    status="passed",
                    created_at="2026-03-16",
                )
            ],
            approval_records=[ApprovalRecord(approval_id="approval_1", approval_type="local_execution", scope="python_execution:draft.py", granted=True, created_at="2026-03-16")],
            workflow_tasks=[
                WorkflowTask(task_id="task_1", title="Main task", round_index=1, source_kind="assignment"),
                WorkflowTask(task_id="task_6", title="Closure task", round_index=6, source_kind="followup"),
            ],
        )

        rendered = _render_state_sections(
            state,
            include_consensus=False,
            include_open_questions=False,
            include_action_items=False,
        )

        self.assertNotIn("Stable Consensus", rendered)
        self.assertIn("Active Conflicts", rendered)
        self.assertNotIn("Open Questions", rendered)
        self.assertNotIn("Action Items", rendered)
        self.assertIn("Literature Library", rendered)
        self.assertIn("Experiment Runs", rendered)
        self.assertIn("cwd=generated_artifacts/execution_runs/project_1/run_1", rendered)
        self.assertIn("Main Round 1", rendered)
        self.assertIn("Closure Round 6", rendered)

    def test_transcript_can_omit_role_labels(self) -> None:
        transcript = _render_transcript(
            [
                DiscussionMessage(
                    speaker="Analyst",
                    role="assistant",
                    content="A useful finding.",
                    round_index=1,
                    duty="Expert",
                    stage="analysis",
                )
            ],
            include_role_labels=False,
        )

        self.assertIn("### Analyst | Round 1 | analysis", transcript)
        self.assertNotIn("### Analyst | Expert | Round 1 | analysis", transcript)

    def test_workflow_stage_notes_are_rendered_for_debugging(self) -> None:
        state = DiscussionState(
            workflow_stage_records=[
                WorkflowStageRecord(
                    stage_key="discover_literature",
                    stage_label="Discover arXiv Literature",
                    status="completed",
                    notes="arXiv discovery returned no results.",
                )
            ]
        )

        rendered = _render_state_sections(state)

        self.assertIn("notes=arXiv discovery returned no results.", rendered)


if __name__ == "__main__":
    unittest.main()
