from __future__ import annotations

import unittest

from src.discussion_app.models import EvidenceCard, StructuredLogEntry
from src.discussion_app.state import ExperimentRunRecord, PaperRecord, ProjectStateManager, TASK_STATUS_COMPLETED, TASK_STATUS_IN_PROGRESS


class StateLayerTests(unittest.TestCase):
    def test_project_state_manager_tracks_project_tasks_logs_and_artifacts(self) -> None:
        manager = ProjectStateManager()
        project = manager.start_project(
            topic="Topic",
            user_question="What is the result?",
            uploaded_sources=["paper.pdf", "notes.md"],
            language="en",
            rules=["Rule 1"],
            current_stage="Initialization",
            current_question="Waiting",
        )
        state = project.discussion_state

        task = manager.ensure_task(
            state,
            task_id="task_1",
            title="Problem framing",
            description="Clarify the question",
            owner_name="Analyst",
            reviewer_name="Reviewer",
            round_index=1,
            source_kind="assignment",
        )
        manager.begin_task(
            state,
            task_id=task.task_id,
            stage_label="Task 1: Problem framing",
            question=task.display_text,
            round_index=1,
        )
        manager.apply_log_entry(
            state,
            entry=StructuredLogEntry(
                workpackage_index=1,
                workpackage_title="Problem framing",
                speaker="Analyst",
                stage="analysis",
                headline="Initial framing",
                summary="Clarified scope.",
                consensus_add=["Scope is clear"],
                conflicts_add=["Need more evidence"],
                open_questions_add=["Which dataset applies?"],
                action_items_add=["Collect evidence"],
                evidence_add=[EvidenceCard(evidence_id="E1", summary="Snippet", source="paper.pdf")],
            ),
            max_history_items=12,
            max_log_entries=40,
            max_evidence_cards=24,
        )
        manager.complete_task(state, task_id=task.task_id, notes="Done")
        checkpoint = manager.create_checkpoint(
            state,
            label="Problem framing",
            workpackage_index=1,
            max_checkpoints=8,
        )
        artifact = manager.record_artifact(
            state,
            artifact_type="research_report",
            title="Research Report",
            path="meeting_minutes/report.md",
            preview="Final summary",
        )
        manager.record_paper(
            state,
            PaperRecord(
                paper_id="2401.12345",
                title="A Test Paper",
                authors=["Alice", "Bob"],
                local_pdf_path="arxiv_library/project/pdfs/2401.12345.pdf",
                bibtex_key="alice2024test",
            ),
        )
        manager.record_experiment_run(
            state,
            ExperimentRunRecord(
                run_id="run_1",
                script_path="generated_artifacts/test.py",
                working_directory="generated_artifacts/execution_runs/project_1/run_1",
                command=["python", "generated_artifacts/test.py"],
                compile_returncode=0,
                runtime_returncode=0,
                log_path="generated_artifacts/test_run_log.txt",
                status="passed",
                authorized=True,
                created_at="2026-03-16 10:00:00",
            ),
        )
        manager.record_approval(
            state,
            approval_type="local_execution",
            scope="python_execution:test.py",
            granted=True,
            details="User approved the run.",
        )
        manager.update_summary(state, "Final summary")

        self.assertEqual(project.user_question, "What is the result?")
        self.assertEqual(state.uploaded_sources, ["paper.pdf", "notes.md"])
        self.assertEqual(state.workflow_tasks[0].status, TASK_STATUS_COMPLETED)
        self.assertEqual(state.current_round, 1)
        self.assertIn("Scope is clear", state.consensus_points)
        self.assertIn("Need more evidence", state.risks_or_disagreements)
        self.assertEqual(state.workflow_tasks[0].notes, "Done")
        self.assertEqual(checkpoint.related_task_ids, ["task_1"])
        self.assertEqual(artifact.path, "meeting_minutes/report.md")
        self.assertEqual(state.literature_library[0].paper_id, "2401.12345")
        self.assertEqual(state.experiment_runs[0].status, "passed")
        self.assertEqual(state.experiment_runs[0].working_directory, "generated_artifacts/execution_runs/project_1/run_1")
        self.assertTrue(state.approval_records[0].granted)
        self.assertEqual(state.summary, "Final summary")

    def test_begin_task_marks_state_as_in_progress(self) -> None:
        manager = ProjectStateManager()
        project = manager.start_project(
            topic="Topic",
            user_question="Question",
            uploaded_sources=[],
            language="en",
            rules=[],
            current_stage="Initialization",
            current_question="Waiting",
        )
        state = project.discussion_state
        manager.ensure_task(
            state,
            task_id="task_2",
            title="Analysis",
            description="Analyze",
            owner_name="Analyst",
            reviewer_name="",
            round_index=2,
            source_kind="assignment",
        )

        task = manager.begin_task(
            state,
            task_id="task_2",
            stage_label="Task 2: Analysis",
            question="Analyze",
            round_index=2,
        )

        self.assertIsNotNone(task)
        self.assertEqual(task.status, TASK_STATUS_IN_PROGRESS)
        self.assertEqual(state.current_stage, "Task 2: Analysis")
        self.assertEqual(state.current_round, 2)


if __name__ == "__main__":
    unittest.main()
