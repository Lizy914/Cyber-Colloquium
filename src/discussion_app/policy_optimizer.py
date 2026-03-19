from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

from .evaluation import (
    BENCHMARK_TASKS_DIR,
    BenchmarkTask,
    WorkflowEvaluationRunner,
    WorkflowObjectiveWeights,
    discover_benchmark_tasks,
)
from .models import ProviderConfig
from .workflow_config import WorkflowConfig, load_workflow_config, save_workflow_config
from .workflow_graph import build_workflow_graph, workflow_policy_snapshot


POLICY_SEARCHES_DIR = Path("benchmarks") / "searches"


@dataclass(frozen=True)
class WorkflowPolicySearchSpace:
    max_rounds_options: tuple[int, ...] = (4, 6, 8)
    checkpoint_every_options: tuple[int, ...] = (1, 2)
    reviewer_enabled_options: tuple[bool, ...] = (True, False)
    max_history_options: tuple[int, ...] = (8, 12, 16)
    max_evidence_options: tuple[int, ...] = (12, 24, 32)
    max_log_entries_options: tuple[int, ...] = (24, 40, 60)
    max_followup_items_options: tuple[int, ...] = (2, 3, 4)
    max_followup_attempts_options: tuple[int, ...] = (1, 2, 3)
    summary_slot_sets: tuple[tuple[str, ...], ...] = (
        ("consensus", "conflicts", "open_questions", "action_items"),
        ("consensus", "conflicts", "open_questions", "recent_updates", "action_items"),
        ("consensus", "open_questions", "action_items"),
    )


@dataclass(frozen=True)
class PolicyCandidateResult:
    candidate_id: str
    policy_version: str
    config_path: str
    suite_results_path: str
    average_objective_loss: float
    average_overall_score: float
    success_rate: float
    parameter_snapshot: dict[str, object]


@dataclass(frozen=True)
class PolicyOptimizationResult:
    search_id: str
    benchmark_split: str
    task_count: int
    candidate_count: int
    best_candidate_id: str
    best_policy_version: str
    best_config_path: str
    summary_path: str
    training_corpus_path: str
    candidates: list[PolicyCandidateResult] = field(default_factory=list)

class WorkflowPolicyOptimizer:
    def __init__(
        self,
        *,
        base_config: WorkflowConfig | None = None,
        providers: list[ProviderConfig] | None = None,
        output_root: Path = POLICY_SEARCHES_DIR,
        objective_weights: WorkflowObjectiveWeights | None = None,
        search_space: WorkflowPolicySearchSpace | None = None,
        executor=None,
    ) -> None:
        self.base_config = base_config or load_workflow_config()
        self.providers = providers or []
        self.output_root = output_root
        self.objective_weights = objective_weights or WorkflowObjectiveWeights()
        self.search_space = search_space or WorkflowPolicySearchSpace()
        self.executor = executor

    def run(
        self,
        tasks: list[BenchmarkTask],
        *,
        split: str,
        samples: int = 6,
        seed: int = 7,
        include_base_candidate: bool = True,
    ) -> PolicyOptimizationResult:
        if not tasks:
            raise ValueError("At least one benchmark task is required for policy optimization.")

        rng = random.Random(seed)
        search_id = f"policy_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        search_dir = self.output_root / search_id
        search_dir.mkdir(parents=True, exist_ok=True)
        training_corpus_path = search_dir / "policy_training_corpus.jsonl"

        candidates: list[WorkflowConfig] = []
        if include_base_candidate:
            candidates.append(self.base_config)
        for _ in range(max(0, samples)):
            candidates.append(self._sample_config(rng))

        candidate_results: list[PolicyCandidateResult] = []
        with training_corpus_path.open("w", encoding="utf-8") as corpus_handle:
            for index, config in enumerate(candidates, start=1):
                candidate_id = f"candidate_{index:02d}"
                policy_version = self._policy_version(candidate_id, config)
                candidate_dir = search_dir / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                config_path = candidate_dir / "workflow_config.json"
                save_workflow_config(config, config_path)

                runner = WorkflowEvaluationRunner(
                    workflow_config=config,
                    providers=self.providers,
                    policy_version=policy_version,
                    output_root=candidate_dir / "runs",
                    objective_weights=self.objective_weights,
                    executor=self.executor,
                )
                suite_result = runner.run_suite(tasks)
                parameter_snapshot = workflow_policy_snapshot(config)
                candidate_result = PolicyCandidateResult(
                    candidate_id=candidate_id,
                    policy_version=policy_version,
                    config_path=str(config_path),
                    suite_results_path=suite_result.results_path,
                    average_objective_loss=suite_result.average_objective_loss,
                    average_overall_score=suite_result.average_overall_score,
                    success_rate=suite_result.success_rate,
                    parameter_snapshot=parameter_snapshot,
                )
                candidate_results.append(candidate_result)

                corpus_handle.write(
                    json.dumps(
                        {
                            "search_id": search_id,
                            "candidate_id": candidate_id,
                            "policy_version": policy_version,
                            "benchmark_split": split,
                            "task_count": len(tasks),
                            "config_snapshot": asdict(config),
                            "policy_snapshot": parameter_snapshot,
                            "workflow_graph": build_workflow_graph(config.workflow_template).to_dict(),
                            "objective_weights": asdict(self.objective_weights),
                            "metrics": {
                                "average_objective_loss": suite_result.average_objective_loss,
                                "average_overall_score": suite_result.average_overall_score,
                                "success_rate": suite_result.success_rate,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        best_candidate = min(
            candidate_results,
            key=lambda item: (item.average_objective_loss, -item.average_overall_score, -item.success_rate),
        )
        result = PolicyOptimizationResult(
            search_id=search_id,
            benchmark_split=split,
            task_count=len(tasks),
            candidate_count=len(candidate_results),
            best_candidate_id=best_candidate.candidate_id,
            best_policy_version=best_candidate.policy_version,
            best_config_path=best_candidate.config_path,
            summary_path=str(search_dir / "policy_search_summary.json"),
            training_corpus_path=str(training_corpus_path),
            candidates=candidate_results,
        )
        Path(result.summary_path).write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _sample_config(self, rng: random.Random) -> WorkflowConfig:
        summary_slots = list(rng.choice(list(self.search_space.summary_slot_sets)))
        return replace(
            self.base_config,
            discussion=replace(
                self.base_config.discussion,
                max_rounds=rng.choice(list(self.search_space.max_rounds_options)),
                checkpoint_every_n_rounds=rng.choice(list(self.search_space.checkpoint_every_options)),
                enable_reviewer_role=rng.choice(list(self.search_space.reviewer_enabled_options)),
                max_followup_items=rng.choice(list(self.search_space.max_followup_items_options)),
                max_followup_attempts=rng.choice(list(self.search_space.max_followup_attempts_options)),
            ),
            context=replace(
                self.base_config.context,
                max_history_items=rng.choice(list(self.search_space.max_history_options)),
                summary_slots=summary_slots,
                max_evidence_cards=rng.choice(list(self.search_space.max_evidence_options)),
                max_log_entries=rng.choice(list(self.search_space.max_log_entries_options)),
            ),
        )

    def _policy_version(self, candidate_id: str, config: WorkflowConfig) -> str:
        reviewer_flag = "rv1" if config.discussion.enable_reviewer_role else "rv0"
        return (
            f"{candidate_id}_"
            f"r{config.discussion.max_rounds}_"
            f"ck{config.discussion.checkpoint_every_n_rounds}_"
            f"{reviewer_flag}_"
            f"ctx{config.context.max_history_items}_"
            f"ev{config.context.max_evidence_cards}_"
            f"fu{config.discussion.max_followup_items}x{config.discussion.max_followup_attempts}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run random-search workflow policy optimization on the benchmark suite.")
    parser.add_argument("--tasks-root", type=Path, default=BENCHMARK_TASKS_DIR, help="Directory containing benchmark tasks.")
    parser.add_argument("--split", default="train", help="Benchmark split to optimize on.")
    parser.add_argument("--workflow-config", type=Path, default=None, help="Optional workflow config path.")
    parser.add_argument("--output-root", type=Path, default=POLICY_SEARCHES_DIR, help="Directory used to persist search outputs.")
    parser.add_argument("--samples", type=int, default=6, help="Number of random policy samples to evaluate in addition to the base config.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for policy sampling.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of tasks to run from the split.")
    parser.add_argument("--skip-base-config", action="store_true", help="Skip evaluating the current workflow config as the baseline candidate.")
    args = parser.parse_args(argv)

    workflow_config = load_workflow_config(args.workflow_config) if args.workflow_config is not None else load_workflow_config()
    tasks = discover_benchmark_tasks(args.tasks_root, split=args.split or None)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        print("No benchmark tasks were found for policy optimization.")
        return 1

    optimizer = WorkflowPolicyOptimizer(
        base_config=workflow_config,
        output_root=args.output_root,
    )
    result = optimizer.run(
        tasks,
        split=args.split or "all",
        samples=args.samples,
        seed=args.seed,
        include_base_candidate=not args.skip_base_config,
    )
    print(f"Policy search summary saved to: {result.summary_path}")
    print(f"Best candidate: {result.best_candidate_id} | policy_version={result.best_policy_version}")
    print(f"Best config path: {result.best_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
