from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import EXPERT_DUTY, HOST_DUTY, LEAD_DUTY, LITERATURE_DUTY, REPORT_DUTY


MODERATOR_ROLE_KEY = "moderator"
NOTETAKER_ROLE_KEY = "notetaker"
RESEARCH_ANALYST_ROLE_KEY = "research_analyst"
REVIEWER_ROLE_KEY = "reviewer"


@dataclass(frozen=True)
class RoleOutputSchema:
    format: str
    sections: tuple[str, ...] = ()
    description: str = ""


class AgentRole(Protocol):
    key: str
    name: str
    responsibility: str
    allowed_actions: tuple[str, ...]
    output_schema: RoleOutputSchema
    compatible_duties: tuple[str, ...]

    def supports_action(self, action: str) -> bool:
        ...


@dataclass(frozen=True)
class RoleDefinition:
    key: str
    name: str
    responsibility: str
    allowed_actions: tuple[str, ...]
    output_schema: RoleOutputSchema
    compatible_duties: tuple[str, ...] = ()

    def supports_action(self, action: str) -> bool:
        return action in self.allowed_actions


class RoleRegistry:
    def __init__(self, roles: list[RoleDefinition] | None = None) -> None:
        self._roles: dict[str, RoleDefinition] = {}
        for role in roles or []:
            self.register(role)

    def register(self, role: RoleDefinition) -> None:
        self._roles[role.key] = role

    def get(self, key: str) -> RoleDefinition | None:
        return self._roles.get(key)

    def require(self, key: str) -> RoleDefinition:
        role = self.get(key)
        if role is None:
            raise KeyError(f"Role '{key}' is not registered.")
        return role

    def roles(self) -> list[RoleDefinition]:
        return list(self._roles.values())

    def resolve_for_duty(self, duty: str, *, explicit_key: str = "") -> RoleDefinition | None:
        if explicit_key:
            return self.get(explicit_key)
        return next((role for role in self._roles.values() if duty in role.compatible_duties), None)


def default_role_registry() -> RoleRegistry:
    return RoleRegistry(
        roles=[
            RoleDefinition(
                key=MODERATOR_ROLE_KEY,
                name="Moderator",
                responsibility="Guide workflow structure, decompose work, coordinate role handoffs, and decide whether follow-up discussion is needed.",
                allowed_actions=("decompose_task", "coordinate_workflow", "resolve_followup"),
                output_schema=RoleOutputSchema(
                    format="markdown",
                    sections=("Research Goal", "Domain", "Assignments", "Coordination Decision"),
                    description="Short structured markdown plans used for delegation and workflow coordination.",
                ),
                compatible_duties=(LEAD_DUTY, HOST_DUTY),
            ),
            RoleDefinition(
                key=NOTETAKER_ROLE_KEY,
                name="Notetaker",
                responsibility="Record structured discussion state and synthesize it into research reports and meeting minutes.",
                allowed_actions=("log_state", "synthesize_report", "write_minutes"),
                output_schema=RoleOutputSchema(
                    format="json_or_markdown",
                    sections=("headline", "summary", "consensus_add", "action_items_add"),
                    description="Writes JSON state patches during discussion and markdown artifacts for exports.",
                ),
                compatible_duties=(REPORT_DUTY,),
            ),
            RoleDefinition(
                key=RESEARCH_ANALYST_ROLE_KEY,
                name="Research Analyst",
                responsibility="Analyze assigned subproblems, propose judgments, cite evidence, and hand off risks for review.",
                allowed_actions=("analyze_subproblem", "synthesize_subproblem"),
                output_schema=RoleOutputSchema(
                    format="markdown",
                    sections=("Judgment", "Reasons", "Evidence", "Risk", "Handoff"),
                    description="Structured analytical markdown for subproblem execution and synthesis.",
                ),
                compatible_duties=(EXPERT_DUTY,),
            ),
            RoleDefinition(
                key=REVIEWER_ROLE_KEY,
                name="Reviewer",
                responsibility="Review literature, challenge evidence quality, and verify claims or reasoning from other roles.",
                allowed_actions=("review_subproblem", "review_literature", "inspect_sources"),
                output_schema=RoleOutputSchema(
                    format="markdown",
                    sections=("Verdict", "Corrections", "Evidence Check", "Residual Risk"),
                    description="Structured review markdown focused on source grounding and error correction.",
                ),
                compatible_duties=(LITERATURE_DUTY,),
            ),
        ]
    )
