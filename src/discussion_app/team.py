from __future__ import annotations

from dataclasses import dataclass

from .models import ProviderConfig
from .roles import RoleDefinition, RoleOutputSchema, RoleRegistry, default_role_registry
from .workflow_config import TeamRoleConfig, TeamTemplateConfig, WorkflowConfig


@dataclass(frozen=True)
class TeamMember:
    provider: ProviderConfig
    role_definition: RoleDefinition
    template_role: TeamRoleConfig
    template_name: str

    @property
    def duty(self) -> str:
        return self.provider.duty

    @property
    def role_key(self) -> str:
        return self.role_definition.key

    @property
    def role_name(self) -> str:
        return self.role_definition.name

    @property
    def responsibility(self) -> str:
        return self.role_definition.responsibility

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return self.role_definition.allowed_actions

    @property
    def output_schema(self) -> RoleOutputSchema:
        return self.role_definition.output_schema

    def supports_action(self, action: str) -> bool:
        return self.role_definition.supports_action(action)

    def roster_line(self) -> str:
        actions = ", ".join(self.allowed_actions)
        return (
            f"- {self.provider.name} | Duty: {self.duty} | Role: {self.role_name} | "
            f"Responsibility: {self.responsibility} | Allowed actions: {actions} | "
            f"Specialty: {self.provider.specialty or self.template_role.specialty_hint or 'Not set'}"
        )


@dataclass(frozen=True)
class ResearchTeam:
    template: TeamTemplateConfig
    members: tuple[TeamMember, ...]
    role_registry: RoleRegistry

    def primary_member_for_duty(self, duty: str) -> TeamMember | None:
        return next((member for member in self.members if member.duty == duty), None)

    def members_for_duty(self, duty: str) -> list[TeamMember]:
        return [member for member in self.members if member.duty == duty]

    def members_for_role(self, role_key: str) -> list[TeamMember]:
        return [member for member in self.members if member.role_key == role_key]

    def primary_member_for_action(self, action: str) -> TeamMember | None:
        return next((member for member in self.members if member.supports_action(action)), None)

    def missing_required_duties(self) -> list[str]:
        present = {member.duty for member in self.members}
        return [role.duty for role in self.template.roles if role.enabled and role.required and role.duty not in present]


def build_research_team(
    providers: list[ProviderConfig],
    workflow_config: WorkflowConfig,
    role_registry: RoleRegistry | None = None,
) -> ResearchTeam:
    registry = role_registry or default_role_registry()
    template = workflow_config.team_template
    template_roles = {role.duty: role for role in workflow_config.team_roles}

    members: list[TeamMember] = []
    for provider in providers:
        if not provider.enabled or not provider.api_key:
            continue
        template_role = template_roles.get(provider.duty)
        if template_role is None or not template_role.enabled:
            continue
        role_definition = registry.resolve_for_duty(provider.duty, explicit_key=template_role.role_key)
        if role_definition is None:
            continue
        members.append(
            TeamMember(
                provider=provider,
                role_definition=role_definition,
                template_role=template_role,
                template_name=template.name,
            )
        )

    order_lookup = {duty: index for index, duty in enumerate(workflow_config.ordered_role_duties())}
    members.sort(key=lambda member: (order_lookup.get(member.duty, len(order_lookup)), member.provider.name.lower()))
    return ResearchTeam(template=template, members=tuple(members), role_registry=registry)
