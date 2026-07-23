"""Create the complete gateway and execution-harness schema.

Revision ID: 0001_initial
Revises:
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamp(name: str, *, nullable: bool = False) -> sa.Column:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        nullable=nullable,
        server_default=sa.text("CURRENT_TIMESTAMP") if not nullable else None,
    )


def upgrade() -> None:
    op.create_table(
        "virtual_keys",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("budget", sa.Integer(), nullable=False),
        sa.Column("requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint("budget >= 0", name="ck_virtual_keys_budget"),
        sa.CheckConstraint("requests >= 0", name="ck_virtual_keys_requests"),
        sa.CheckConstraint("tokens_in >= 0", name="ck_virtual_keys_tokens_in"),
        sa.CheckConstraint("tokens_out >= 0", name="ck_virtual_keys_tokens_out"),
    )
    op.create_table(
        "usage_events",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column(
            "virtual_key",
            sa.Text(),
            sa.ForeignKey("virtual_keys.key"),
            nullable=False,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        _timestamp("created_at"),
        sa.CheckConstraint("tokens_in >= 0", name="ck_usage_events_tokens_in"),
        sa.CheckConstraint("tokens_out >= 0", name="ck_usage_events_tokens_out"),
    )
    op.create_table(
        "task_executions",
        sa.Column("task_id", sa.Text(), primary_key=True),
        sa.Column("virtual_key_id", sa.Text(), nullable=False),
        sa.Column("skill", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("final_provider", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("error_category", sa.Text(), nullable=True),
        _timestamp("created_at"),
        _timestamp("completed_at", nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_task_executions_status",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_task_executions_attempts"),
        sa.CheckConstraint(
            "prompt_tokens >= 0",
            name="ck_task_executions_prompt_tokens",
        ),
        sa.CheckConstraint(
            "completion_tokens >= 0",
            name="ck_task_executions_completion_tokens",
        ),
    )
    op.create_index(
        "idx_task_executions_owner",
        "task_executions",
        ["virtual_key_id", "task_id"],
    )
    op.create_table(
        "task_attempts",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column(
            "task_id",
            sa.Text(),
            sa.ForeignKey("task_executions.task_id"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("attempt_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("validation_error_category", sa.Text(), nullable=True),
        sa.Column("provider_error_category", sa.Text(), nullable=True),
        _timestamp("created_at"),
        sa.CheckConstraint(
            "attempt_number > 0",
            name="ck_task_attempts_attempt_number",
        ),
        sa.CheckConstraint(
            """
            attempt_type IN (
                'initial', 'repair', 'fallback', 'fallback_repair',
                'post_tool', 'post_tool_repair',
                'post_tool_fallback', 'post_tool_fallback_repair'
            )
            """,
            name="ck_task_attempts_attempt_type",
        ),
        sa.CheckConstraint(
            """
            status IN (
                'completed', 'validation_error',
                'operational_error', 'configuration_error'
            )
            """,
            name="ck_task_attempts_status",
        ),
        sa.CheckConstraint(
            "prompt_tokens >= 0",
            name="ck_task_attempts_prompt_tokens",
        ),
        sa.CheckConstraint(
            "completion_tokens >= 0",
            name="ck_task_attempts_completion_tokens",
        ),
        sa.CheckConstraint(
            """
            validation_error_category IS NULL OR
            validation_error_category IN (
                'parsing', 'structure', 'semantic', 'tool_protocol'
            )
            """,
            name="ck_task_attempts_validation_category",
        ),
        sa.CheckConstraint(
            """
            provider_error_category IS NULL OR
            provider_error_category IN ('operational', 'configuration')
            """,
            name="ck_task_attempts_provider_category",
        ),
        sa.UniqueConstraint(
            "task_id",
            "attempt_number",
            name="uq_task_attempts_task_number",
        ),
    )
    op.create_index(
        "idx_task_attempts_task",
        "task_attempts",
        ["task_id", "attempt_number"],
    )
    op.create_table(
        "user_preferences",
        sa.Column("virtual_key_id", sa.Text(), nullable=False),
        sa.Column("preference_key", sa.Text(), nullable=False),
        sa.Column("preference_value_json", sa.Text(), nullable=False),
        _timestamp("updated_at"),
        sa.PrimaryKeyConstraint(
            "virtual_key_id",
            "preference_key",
            name="pk_user_preferences",
        ),
    )
    op.create_table(
        "task_tool_executions",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column(
            "task_id",
            sa.Text(),
            sa.ForeignKey("task_executions.task_id"),
            nullable=False,
        ),
        sa.Column("tool_number", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_category", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        _timestamp("created_at"),
        _timestamp("completed_at", nullable=True),
        sa.CheckConstraint("tool_number = 1", name="ck_task_tools_number"),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_task_tools_status",
        ),
        sa.CheckConstraint("duration_ms >= 0", name="ck_task_tools_duration"),
        sa.UniqueConstraint(
            "task_id",
            "tool_number",
            name="uq_task_tools_task_number",
        ),
    )
    op.create_index(
        "idx_task_tools_task",
        "task_tool_executions",
        ["task_id", "tool_number"],
    )
    op.create_table(
        "workflow_executions",
        sa.Column("workflow_id", sa.Text(), primary_key=True),
        sa.Column("virtual_key_id", sa.Text(), nullable=False),
        sa.Column("definition_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("completed_steps", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("error_category", sa.Text(), nullable=True),
        _timestamp("created_at"),
        _timestamp("completed_at", nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_workflow_executions_status",
        ),
        sa.CheckConstraint("step_count > 0", name="ck_workflow_step_count"),
        sa.CheckConstraint(
            "completed_steps >= 0 AND completed_steps <= step_count",
            name="ck_workflow_completed_steps",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_workflow_attempts"),
        sa.CheckConstraint("tool_count >= 0", name="ck_workflow_tool_count"),
        sa.CheckConstraint("prompt_tokens >= 0", name="ck_workflow_prompt_tokens"),
        sa.CheckConstraint(
            "completion_tokens >= 0",
            name="ck_workflow_completion_tokens",
        ),
    )
    op.create_index(
        "idx_workflow_executions_owner",
        "workflow_executions",
        ["virtual_key_id", "workflow_id"],
    )
    op.create_table(
        "workflow_steps",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column(
            "workflow_id",
            sa.Text(),
            sa.ForeignKey("workflow_executions.workflow_id"),
            nullable=False,
        ),
        sa.Column("step_order", sa.Integer(), nullable=False),
        sa.Column("step_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("skill", sa.Text(), nullable=False),
        sa.Column(
            "task_id",
            sa.Text(),
            sa.ForeignKey("task_executions.task_id"),
            nullable=True,
            unique=True,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("error_category", sa.Text(), nullable=True),
        _timestamp("created_at"),
        _timestamp("started_at", nullable=True),
        _timestamp("completed_at", nullable=True),
        sa.CheckConstraint("step_order > 0", name="ck_workflow_steps_order"),
        sa.CheckConstraint(
            """
            status IN (
                'pending', 'running', 'completed', 'failed', 'skipped'
            )
            """,
            name="ck_workflow_steps_status",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_workflow_steps_attempts"),
        sa.CheckConstraint("tool_count >= 0", name="ck_workflow_steps_tool_count"),
        sa.CheckConstraint(
            "prompt_tokens >= 0",
            name="ck_workflow_steps_prompt_tokens",
        ),
        sa.CheckConstraint(
            "completion_tokens >= 0",
            name="ck_workflow_steps_completion_tokens",
        ),
        sa.UniqueConstraint(
            "workflow_id",
            "step_order",
            name="uq_workflow_steps_order",
        ),
        sa.UniqueConstraint(
            "workflow_id",
            "step_id",
            name="uq_workflow_steps_step_id",
        ),
    )
    op.create_index(
        "idx_workflow_steps_execution",
        "workflow_steps",
        ["workflow_id", "step_order"],
    )


def downgrade() -> None:
    op.drop_index("idx_workflow_steps_execution", table_name="workflow_steps")
    op.drop_table("workflow_steps")
    op.drop_index(
        "idx_workflow_executions_owner",
        table_name="workflow_executions",
    )
    op.drop_table("workflow_executions")
    op.drop_index("idx_task_tools_task", table_name="task_tool_executions")
    op.drop_table("task_tool_executions")
    op.drop_table("user_preferences")
    op.drop_index("idx_task_attempts_task", table_name="task_attempts")
    op.drop_table("task_attempts")
    op.drop_index("idx_task_executions_owner", table_name="task_executions")
    op.drop_table("task_executions")
    op.drop_table("usage_events")
    op.drop_table("virtual_keys")
