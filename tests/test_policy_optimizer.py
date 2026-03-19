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

from src.discussion_app.evaluation import load_benchmark_task
from src.discussion_app.models import DiscussionMessage, DiscussionResult
from src.discussion_app.policy_optimizer import WorkflowPolicyOptimizer
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
            )
        ],
        final_summary="# Research Report\n\n## Findings\n\nA structured result.",
        meeting_minutes="# Meeting Minutes\n\n## Decisions\n\nA structured note set.",
        meeting_state=state,
    )


class PolicyOptimizerTests(unittest.TestCase):
    def test_optimizer_runs_candidates_and_exports_training_corpus(self) -> None:
        task = load_benchmark_task(Path("benchmarks/tasks/train/cc_bench_001.json"))

        with tempfile.TemporaryDirectory() as temp_dir:
            optimizer = WorkflowPolicyOptimizer(
                base_config=workflow_config_from_dict({}),
                providers=[],
                output_root=Path(temp_dir),
                executor=_fake_executor,
            )

            result = optimizer.run(
                [task],
                split="train",
                samples=2,
                seed=11,
                include_base_candidate=True,
            )

            self.assertEqual(result.task_count, 1)
            self.assertEqual(result.candidate_count, 3)
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.training_corpus_path).exists())
            self.assertTrue(Path(result.best_config_path).exists())

            summary_payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            corpus_lines = Path(result.training_corpus_path).read_text(encoding="utf-8").splitlines()

            self.assertEqual(summary_payload["candidate_count"], 3)
            self.assertEqual(len(corpus_lines), 3)
            self.assertIn("best_candidate_id", summary_payload)
            self.assertTrue(any("policy_snapshot" in line for line in corpus_lines))


if __name__ == "__main__":
    unittest.main()
