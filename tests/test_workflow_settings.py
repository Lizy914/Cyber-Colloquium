from __future__ import annotations

import unittest

from src.discussion_app.models import EXPERT_DUTY, LITERATURE_DUTY
from src.discussion_app.workflow_config import workflow_config_from_dict
from src.discussion_app.workflow_settings import (
    WorkflowSettingsState,
    apply_workflow_settings,
    render_workflow_settings_summary,
    validate_workflow_settings,
    workflow_settings_from_config,
)


class WorkflowSettingsTests(unittest.TestCase):
    def test_workflow_settings_can_update_discussion_context_and_role_toggles(self) -> None:
        config = workflow_config_from_dict({})
        settings = WorkflowSettingsState(
            max_rounds=5,
            checkpoint_every_n_rounds=2,
            reviewer_enabled=False,
            enabled_roles={
                EXPERT_DUTY: True,
                LITERATURE_DUTY: False,
            },
            summary_slots=["consensus", "action_items"],
            arxiv_discovery_enabled=True,
            arxiv_download_enabled=False,
            arxiv_max_results=4,
            python_artifact_enabled=False,
            latex_artifact_enabled=False,
            bibtex_artifact_enabled=True,
            python_execution_test_enabled=False,
            python_full_execution_enabled=True,
            python_execution_timeout_seconds=45,
            python_full_execution_timeout_seconds=180,
            python_workspace_input_limit_mb=128,
            latex_compile_enabled=False,
        )

        updated = apply_workflow_settings(config, settings)

        self.assertEqual(updated.discussion.max_rounds, 5)
        self.assertEqual(updated.discussion.checkpoint_every_n_rounds, 2)
        self.assertFalse(updated.discussion.enable_reviewer_role)
        self.assertFalse(updated.role_config(LITERATURE_DUTY).enabled)
        self.assertTrue(updated.role_config(EXPERT_DUTY).enabled)
        self.assertEqual(updated.context.summary_slots, ["consensus", "action_items"])
        self.assertTrue(updated.tooling.enable_arxiv_discovery)
        self.assertFalse(updated.tooling.download_arxiv_pdfs)
        self.assertEqual(updated.tooling.arxiv_max_results, 4)
        self.assertTrue(updated.tooling.enable_bibtex_artifact)
        self.assertFalse(updated.tooling.enable_python_full_execution)
        self.assertEqual(updated.tooling.python_execution_timeout_seconds, 45)
        self.assertEqual(updated.tooling.python_full_execution_timeout_seconds, 180)
        self.assertEqual(updated.tooling.python_workspace_input_limit_mb, 128)
        self.assertEqual(updated.workflow_template.name, config.workflow_template.name)

    def test_validate_workflow_settings_requires_enabled_role_and_summary_slot(self) -> None:
        settings = WorkflowSettingsState(
            max_rounds=3,
            checkpoint_every_n_rounds=1,
            reviewer_enabled=True,
            enabled_roles={EXPERT_DUTY: False, LITERATURE_DUTY: False},
            summary_slots=[],
            arxiv_discovery_enabled=False,
            arxiv_download_enabled=True,
            arxiv_max_results=0,
            python_artifact_enabled=False,
            latex_artifact_enabled=False,
            bibtex_artifact_enabled=False,
            python_execution_test_enabled=False,
            python_full_execution_enabled=False,
            python_execution_timeout_seconds=4,
            python_full_execution_timeout_seconds=9,
            python_workspace_input_limit_mb=0,
            latex_compile_enabled=False,
        )

        errors = validate_workflow_settings(settings, [EXPERT_DUTY, LITERATURE_DUTY])

        self.assertEqual(len(errors), 6)
        self.assertIn("arXiv 最大结果数", errors[0])
        self.assertIn("Python 执行超时", errors[1])
        self.assertIn("Python 完整运行超时", errors[2])
        self.assertIn("Python 工作目录输入上限", errors[3])
        self.assertIn("至少保留一个启用角色", errors[4])
        self.assertIn("结构化摘要槽位", errors[5])

    def test_render_workflow_settings_summary_is_human_readable(self) -> None:
        config = workflow_config_from_dict(
            {
                "discussion": {
                    "max_rounds": 6,
                    "checkpoint_every_n_rounds": 3,
                    "enable_reviewer_role": False,
                },
                "context": {
                    "summary_slots": ["conflicts", "action_items"],
                },
            }
        )

        summary = render_workflow_settings_summary(config)
        state = workflow_settings_from_config(config)

        self.assertIn("回合数: 6", summary)
        self.assertIn("检查点: 3", summary)
        self.assertIn("复核阶段: 关闭", summary)
        self.assertIn("arXiv: 关闭", summary)
        self.assertIn("冒烟测试: 关闭 (20s)", summary)
        self.assertIn("完整运行: 关闭 (300s)", summary)
        self.assertIn("Tectonic: 关闭", summary)
        self.assertIn("争议", summary)
        self.assertIn("工作流图:", summary)
        self.assertEqual(state.max_rounds, 6)
        self.assertEqual(state.summary_slots, ["conflicts", "action_items"])


if __name__ == "__main__":
    unittest.main()
