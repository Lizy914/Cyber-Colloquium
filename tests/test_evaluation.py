from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


_pypdf_stub = types.ModuleType("pypdf")


class _PdfReader:
    def __init__(self, *args, **kwargs) -> None:
        self.pages = []


_pypdf_stub.PdfReader = _PdfReader
sys.modules.setdefault("pypdf", _pypdf_stub)

from src.discussion_app.evaluation import BenchmarkTask, WorkflowEvaluationRunner, discover_benchmark_tasks, load_benchmark_task
from src.discussion_app.models import DiscussionMessage, DiscussionResult
from src.discussion_app.state import Checkpoint, DiscussionState, WorkflowStageRecord, WorkflowTask
from src.discussion_app.workflow_config import workflow_config_from_dict


def _fake_executor(task, workflow_config, providers, attachments, user_request, generate_literature_review):  # noqa: ANN001
    del workflow_config, providers, attachments, user_request, generate_literature_review
    state = DiscussionState(
        topic=task.inputs.topic,
        user_question=task.inputs.user_question or task.inputs.topic,
        summary="Structured benchmark summary",
        current_stage="Generate Research Report",
        current_round=2,
        consensus_points=["Consensus item"],
        open_questions=["Open question item"],
        risks_or_disagreements=["Critical disagreement"],
        action_items=["Action item"],
        workflow_tasks=[
            WorkflowTask(
                task_id="task_1",
                title="Problem framing",
                owner_name="Lead",
                reviewer_name="Reviewer",
                round_index=1,
                status="completed",
            )
        ],
        workflow_stage_records=[
            WorkflowStageRecord(stage_key="ingest_source_material", stage_label="Ingest Source Material", status="completed"),
            WorkflowStageRecord(stage_key="generate_research_report", stage_label="Generate Research Report", status="completed"),
        ],
        checkpoints=[
            Checkpoint(
                checkpoint_id="CP1",
                label="Checkpoint",
                workpackage_index=1,
                round_index=1,
                summary="Checkpoint summary",
            )
        ],
    )
    return DiscussionResult(
        messages=[
            DiscussionMessage(
                speaker="Lead",
                role="assistant",
                content="Assignment message",
                round_index=0,
                duty="Lead",
                stage="assignment",
            ),
            DiscussionMessage(
                speaker="Reviewer",
                role="assistant",
                content="Critical issue found in the argument.",
                round_index=1,
                duty="Literature Reviewer",
                stage="review",
            ),
        ],
        literature_review="Literature review output",
        final_summary="# Research Report\n\n## Findings\n\nA structured result.",
        meeting_minutes="# Meeting Minutes\n\n## Decisions\n\nA structured note set.",
        meeting_state=state,
    )


class EvaluationHarnessTests(unittest.TestCase):
    def test_load_and_discover_benchmark_tasks_by_split(self) -> None:
        payload = {
            "schema_version": 1,
            "task_id": "cc_eval_001",
            "task_type": "open_research_discussion",
            "title": "Benchmark title",
            "inputs": {
                "source_type": "topic_only",
                "topic": "Topic",
                "pdf_paths": [],
                "seed_summary": None,
                "user_question": "Question",
            },
            "expected_outputs": {
                "require_summary": False,
                "require_meeting_notes": True,
                "require_research_report": True,
            },
            "scoring": {
                "required_slots": ["consensus_points", "action_items"],
                "reviewer_must_raise_critique": True,
                "must_include_action_items": True,
            },
            "metadata": {
                "difficulty": "easy",
                "domain": "general_research",
                "split": "dev",
                "source_origin": "handcrafted",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir) / "tasks" / "dev"
            task_dir.mkdir(parents=True, exist_ok=True)
            task_path = task_dir / "cc_eval_001.json"
            task_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            task = load_benchmark_task(task_path)
            discovered = discover_benchmark_tasks(Path(temp_dir) / "tasks", split="dev")

            self.assertEqual(task.task_id, "cc_eval_001")
            self.assertEqual(task.metadata.split, "dev")
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].title, "Benchmark title")

    def test_runner_persists_outputs_trace_and_suite_summary(self) -> None:
        task = BenchmarkTask(
            task_id="cc_eval_002",
            task_type="critique_heavy_discussion",
            title="Critique-heavy task",
            inputs=load_benchmark_task(Path("benchmarks/tasks/dev/cc_bench_101.json")).inputs,
            expected_outputs=load_benchmark_task(Path("benchmarks/tasks/dev/cc_bench_101.json")).expected_outputs,
            scoring=load_benchmark_task(Path("benchmarks/tasks/dev/cc_bench_101.json")).scoring,
            metadata=load_benchmark_task(Path("benchmarks/tasks/dev/cc_bench_101.json")).metadata,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = WorkflowEvaluationRunner(
                workflow_config=workflow_config_from_dict({}),
                providers=[],
                output_root=Path(temp_dir),
                policy_version="policy_test",
                executor=_fake_executor,
            )

            run_result = runner.run_task(task)
            suite_result = runner.run_suite([task])

            self.assertTrue(run_result.success)
            self.assertGreater(run_result.completion_score, 0.0)
            self.assertEqual(run_result.checkpoint_count, 1)
            self.assertEqual(run_result.api_calls, 2)
            self.assertEqual(run_result.benchmark_level, task.metadata.benchmark_level)
            self.assertEqual(run_result.benchmark_category, task.metadata.category or task.task_type)
            self.assertIsInstance(run_result.objective_breakdown, dict)
            self.assertTrue(Path(run_result.meeting_notes_path).exists())
            self.assertTrue(Path(run_result.research_report_path).exists())
            self.assertTrue(Path(run_result.execution_trace_path).exists())
            self.assertTrue(Path(run_result.workflow_graph_path).exists())
            self.assertTrue(Path(run_result.workflow_mermaid_path).exists())
            self.assertTrue(Path(run_result.policy_snapshot_path).exists())
            self.assertTrue(Path(run_result.result_path).exists())
            self.assertEqual(len(run_result.role_execution_trace), 2)
            self.assertEqual(suite_result.task_count, 1)
            self.assertEqual(suite_result.success_count, 1)
            self.assertGreaterEqual(suite_result.average_objective_loss, -1.0)
            self.assertTrue(Path(suite_result.results_path).exists())

    def test_runner_marks_missing_required_outputs_as_failure(self) -> None:
        task = BenchmarkTask(
            task_id="cc_eval_003",
            task_type="open_research_discussion",
            title="Missing outputs task",
            inputs=load_benchmark_task(Path("benchmarks/tasks/train/cc_bench_001.json")).inputs,
            expected_outputs=load_benchmark_task(Path("benchmarks/tasks/train/cc_bench_001.json")).expected_outputs,
            scoring=load_benchmark_task(Path("benchmarks/tasks/train/cc_bench_001.json")).scoring,
            metadata=load_benchmark_task(Path("benchmarks/tasks/train/cc_bench_001.json")).metadata,
        )

        def _empty_executor(task, workflow_config, providers, attachments, user_request, generate_literature_review):  # noqa: ANN001
            del task, workflow_config, providers, attachments, user_request, generate_literature_review
            return DiscussionResult()

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = WorkflowEvaluationRunner(
                workflow_config=workflow_config_from_dict({}),
                providers=[],
                output_root=Path(temp_dir),
                policy_version="policy_empty",
                executor=_empty_executor,
            )

            run_result = runner.run_task(task)

            self.assertFalse(run_result.success)
            self.assertEqual(run_result.failure_reason, "Required outputs were not generated.")
            self.assertEqual(run_result.output_presence["meeting_notes"], False)
            self.assertEqual(run_result.output_presence["research_report"], False)


if __name__ == "__main__":
    unittest.main()
