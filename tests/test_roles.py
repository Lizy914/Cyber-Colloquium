from __future__ import annotations

import unittest

from src.discussion_app.models import EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, LITERATURE_DUTY, REPORT_DUTY, ProviderConfig
from src.discussion_app.roles import MODERATOR_ROLE_KEY, NOTETAKER_ROLE_KEY, RESEARCH_ANALYST_ROLE_KEY, REVIEWER_ROLE_KEY, default_role_registry
from src.discussion_app.team import build_research_team
from src.discussion_app.workflow_config import workflow_config_from_dict


def _provider(name: str, duty: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model="mock-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        duty=duty,
        specialty=f"{name} specialty",
    )


class RoleLayerTests(unittest.TestCase):
    def test_default_role_registry_exposes_required_role_archetypes(self) -> None:
        registry = default_role_registry()

        moderator = registry.require(MODERATOR_ROLE_KEY)
        reviewer = registry.require(REVIEWER_ROLE_KEY)

        self.assertTrue(moderator.supports_action("decompose_task"))
        self.assertTrue(moderator.supports_action("coordinate_workflow"))
        self.assertTrue(reviewer.supports_action("review_literature"))
        self.assertEqual(registry.resolve_for_duty(LEAD_DUTY).key, MODERATOR_ROLE_KEY)

    def test_research_team_builds_members_from_template_and_routes_actions(self) -> None:
        config = workflow_config_from_dict({})
        team = build_research_team(
            [
                _provider("Lead", LEAD_DUTY),
                _provider("Host", HOST_DUTY),
                _provider("Analyst", EXPERT_DUTY),
                _provider("Reviewer", LITERATURE_DUTY),
                _provider("Reporter", REPORT_DUTY),
            ],
            config,
        )

        self.assertEqual(team.template.name, "Cyber Colloquium Basic Team")
        self.assertEqual(team.primary_member_for_duty(LEAD_DUTY).role_key, MODERATOR_ROLE_KEY)
        self.assertEqual(team.primary_member_for_duty(EXPERT_DUTY).role_key, RESEARCH_ANALYST_ROLE_KEY)
        self.assertEqual(team.primary_member_for_duty(LITERATURE_DUTY).role_key, REVIEWER_ROLE_KEY)
        self.assertEqual(team.primary_member_for_duty(REPORT_DUTY).role_key, NOTETAKER_ROLE_KEY)
        self.assertEqual(team.primary_member_for_action("write_minutes").provider.name, "Reporter")
        self.assertEqual(team.primary_member_for_action("coordinate_workflow").provider.name, "Lead")
        self.assertEqual(team.missing_required_duties(), [])

    def test_team_template_can_disable_role_instances(self) -> None:
        config = workflow_config_from_dict(
            {
                "team_template": {
                    "name": "Cyber Colloquium Basic Team",
                    "roles": [
                        {
                            "duty": LITERATURE_DUTY,
                            "enabled": False,
                        }
                    ],
                }
            }
        )
        team = build_research_team(
            [
                _provider("Analyst", EXPERT_DUTY),
                _provider("Reviewer", LITERATURE_DUTY),
            ],
            config,
        )

        self.assertIsNone(team.primary_member_for_duty(LITERATURE_DUTY))
        self.assertEqual(team.primary_member_for_duty(EXPERT_DUTY).provider.name, "Analyst")


if __name__ == "__main__":
    unittest.main()
