from __future__ import annotations

import unittest

from src.discussion_app.workflow_config import workflow_config_from_dict
from src.discussion_app.workflow_graph import (
    build_workflow_graph,
    render_workflow_graph_mermaid,
    render_workflow_graph_summary,
    workflow_policy_snapshot,
)


class WorkflowGraphTests(unittest.TestCase):
    def test_build_workflow_graph_uses_enabled_stage_order(self) -> None:
        config = workflow_config_from_dict(
            {
                "workflow_template": {
                    "stages": [
                        {"key": "discover_literature", "enabled": True},
                        {"key": "ingest_source_material", "enabled": True},
                        {"key": "run_team_discussion", "enabled": True},
                        {"key": "run_reviewer_pass", "enabled": False},
                        {"key": "update_structured_state", "enabled": True},
                    ]
                }
            }
        )

        graph = build_workflow_graph(config.workflow_template)

        self.assertEqual(
            [node.key for node in graph.nodes],
            [
                "discover_literature",
                "ingest_source_material",
                "run_team_discussion",
                "expand_literature_search",
                "update_structured_state",
                "run_experiment_cycle",
                "generate_meeting_notes",
                "generate_research_report",
                "compile_latex_artifacts",
            ],
        )
        self.assertEqual(graph.allowed_next_nodes(None), ["discover_literature"])
        self.assertTrue(graph.can_transition("run_team_discussion", "expand_literature_search"))
        self.assertFalse(graph.can_transition("ingest_source_material", "generate_research_report"))

    def test_render_graph_summary_and_policy_snapshot_are_human_readable(self) -> None:
        config = workflow_config_from_dict({})

        graph = build_workflow_graph(config.workflow_template)
        summary = render_workflow_graph_summary(graph)
        mermaid = render_workflow_graph_mermaid(graph)
        snapshot = workflow_policy_snapshot(config)

        self.assertIn("Graph:", summary)
        self.assertIn("nodes", summary)
        self.assertIn("flowchart TD", mermaid)
        self.assertEqual(set(snapshot), {"phi_disc", "phi_ckpt", "phi_ctx", "phi_out", "phi_ctrl"})


if __name__ == "__main__":
    unittest.main()
