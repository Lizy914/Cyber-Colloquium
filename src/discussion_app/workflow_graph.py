from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .workflow_config import WorkflowConfig, WorkflowStageConfig, WorkflowTemplateConfig


@dataclass(frozen=True)
class WorkflowGraphNode:
    key: str
    label: str
    node_kind: str
    role: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    cost: str = "medium"
    quality_prior: str = "medium"
    trigger: str = "always"
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowGraphEdge:
    source: str
    target: str
    condition: str = "always"
    edge_kind: str = "stage_transition"


@dataclass(frozen=True)
class WorkflowGraph:
    graph_id: str
    name: str
    nodes: tuple[WorkflowGraphNode, ...]
    edges: tuple[WorkflowGraphEdge, ...]

    @property
    def node_map(self) -> dict[str, WorkflowGraphNode]:
        return {node.key: node for node in self.nodes}

    def allowed_next_nodes(self, node_key: str | None) -> list[str]:
        if node_key is None:
            return [self.nodes[0].key] if self.nodes else []
        return [edge.target for edge in self.edges if edge.source == node_key]

    def can_transition(self, source: str | None, target: str) -> bool:
        if source is None:
            return not self.nodes or self.nodes[0].key == target
        return any(edge.source == source and edge.target == target for edge in self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }


STAGE_NODE_PRIORS: dict[str, dict[str, Any]] = {
    "discover_literature": {
        "label": "Discover Literature",
        "node_kind": "input",
        "role": "Literature Reviewer",
        "inputs": ("user_question",),
        "outputs": ("literature_library", "source_documents"),
        "cost": "medium",
        "quality_prior": "medium",
        "trigger": "if_arxiv_discovery_enabled",
    },
    "ingest_source_material": {
        "label": "Ingest Source Material",
        "node_kind": "input",
        "role": "System",
        "inputs": ("source_documents", "user_question"),
        "outputs": ("source_summary", "retrieval_context"),
        "cost": "low",
        "quality_prior": "high",
        "trigger": "always",
    },
    "run_team_discussion": {
        "label": "Run Team Discussion",
        "node_kind": "discussion",
        "role": "Lead/Moderator/Expert",
        "inputs": ("retrieval_context", "workflow_state"),
        "outputs": ("discussion_messages", "workflow_tasks"),
        "cost": "high",
        "quality_prior": "high",
        "trigger": "always",
    },
    "expand_literature_search": {
        "label": "Expand Literature Search",
        "node_kind": "tool",
        "role": "Lead/Literature Reviewer",
        "inputs": ("discussion_messages", "open_questions", "literature_library"),
        "outputs": ("literature_library", "retrieval_context"),
        "cost": "medium",
        "quality_prior": "high",
        "trigger": "if_arxiv_discovery_enabled",
    },
    "run_reviewer_pass": {
        "label": "Run Reviewer Pass",
        "node_kind": "discussion",
        "role": "Reviewer",
        "inputs": ("discussion_messages", "workflow_tasks"),
        "outputs": ("review_comments", "risk_updates"),
        "cost": "medium",
        "quality_prior": "high",
        "trigger": "if_reviewer_enabled",
    },
    "update_structured_state": {
        "label": "Update Structured State",
        "node_kind": "control",
        "role": "Reporter",
        "inputs": ("discussion_messages", "review_comments"),
        "outputs": ("consensus_points", "open_questions", "checkpoints"),
        "cost": "low",
        "quality_prior": "high",
        "trigger": "always",
    },
    "run_experiment_cycle": {
        "label": "Run Experiment Cycle",
        "node_kind": "tool",
        "role": "Research Analyst",
        "inputs": ("workflow_state", "action_items"),
        "outputs": ("python_artifacts", "experiment_runs"),
        "cost": "medium",
        "quality_prior": "medium",
        "trigger": "if_python_artifact_enabled",
    },
    "generate_meeting_notes": {
        "label": "Generate Meeting Notes",
        "node_kind": "output",
        "role": "Reporter",
        "inputs": ("workflow_state",),
        "outputs": ("meeting_minutes",),
        "cost": "low",
        "quality_prior": "high",
        "trigger": "always",
    },
    "generate_research_report": {
        "label": "Generate Research Report",
        "node_kind": "output",
        "role": "Reporter",
        "inputs": ("workflow_state", "meeting_minutes"),
        "outputs": ("research_report",),
        "cost": "medium",
        "quality_prior": "high",
        "trigger": "always",
    },
    "compile_latex_artifacts": {
        "label": "Compile LaTeX Artifacts",
        "node_kind": "tool",
        "role": "Reporter",
        "inputs": ("research_report", "bibtex_artifacts"),
        "outputs": ("latex_artifacts", "compiled_pdf"),
        "cost": "medium",
        "quality_prior": "medium",
        "trigger": "if_latex_compile_enabled",
    },
}


def build_workflow_graph(template: WorkflowTemplateConfig) -> WorkflowGraph:
    enabled_stages = template.enabled_stages()
    nodes = tuple(_node_for_stage(stage) for stage in enabled_stages)
    edges = tuple(
        WorkflowGraphEdge(source=enabled_stages[index].key, target=enabled_stages[index + 1].key)
        for index in range(len(enabled_stages) - 1)
    )
    return WorkflowGraph(
        graph_id=f"workflow_graph::{template.name}",
        name=template.name,
        nodes=nodes,
        edges=edges,
    )


def workflow_policy_snapshot(config: WorkflowConfig) -> dict[str, dict[str, Any]]:
    return {
        "phi_disc": {
            "max_rounds": config.discussion.max_rounds,
            "reviewer_enabled": config.discussion.enable_reviewer_role,
            "roles_per_round": len([role for role in config.team_roles if role.enabled]),
            "moderator_first": True,
        },
        "phi_ckpt": {
            "checkpoint_every_n_rounds": config.discussion.checkpoint_every_n_rounds,
            "max_checkpoints": config.discussion.max_checkpoints,
            "max_followup_items": config.discussion.max_followup_items,
            "max_followup_attempts": config.discussion.max_followup_attempts,
        },
        "phi_ctx": {
            "max_history_items": config.context.max_history_items,
            "summary_slots": list(config.context.summary_slots),
            "max_evidence_cards": config.context.max_evidence_cards,
            "max_log_entries": config.context.max_log_entries,
        },
        "phi_out": {
            "notes_include_role_labels": config.notes.include_role_labels,
            "report_include_consensus": config.report.include_consensus,
            "report_include_open_questions": config.report.include_open_questions,
            "report_include_action_items": config.report.include_action_items,
        },
        "phi_ctrl": {
            "max_node_steps": len(config.workflow_template.enabled_stages()),
            "max_token_budget": "estimated_only",
            "python_execution_enabled": config.tooling.enable_python_execution_test,
            "latex_compile_enabled": config.tooling.enable_latex_compile,
        },
    }


def render_workflow_graph_mermaid(graph: WorkflowGraph) -> str:
    lines = ["flowchart TD"]
    for node in graph.nodes:
        lines.append(f'    {node.key}["{node.label}\\n({node.role})"]')
    for edge in graph.edges:
        if edge.condition and edge.condition != "always":
            lines.append(f"    {edge.source} -->|{edge.condition}| {edge.target}")
        else:
            lines.append(f"    {edge.source} --> {edge.target}")
    return "\n".join(lines)


def render_workflow_graph_summary(graph: WorkflowGraph) -> str:
    if not graph.nodes:
        return "Graph: 0 nodes | 0 edges"
    node_kinds = sorted({node.node_kind for node in graph.nodes})
    return (
        f"Graph: {len(graph.nodes)} nodes | {len(graph.edges)} edges | "
        f"kinds={', '.join(node_kinds)}"
    )


def _node_for_stage(stage: WorkflowStageConfig) -> WorkflowGraphNode:
    prior = STAGE_NODE_PRIORS.get(stage.key, {})
    return WorkflowGraphNode(
        key=stage.key,
        label=str(prior.get("label") or stage.label),
        node_kind=str(prior.get("node_kind") or "stage"),
        role=str(prior.get("role") or "System"),
        inputs=tuple(prior.get("inputs", ())),
        outputs=tuple(prior.get("outputs", ())),
        cost=str(prior.get("cost") or "medium"),
        quality_prior=str(prior.get("quality_prior") or "medium"),
        trigger=str(prior.get("trigger") or "always"),
        attributes={
            "stage_label": stage.label,
            "stage_description": stage.description,
        },
    )
