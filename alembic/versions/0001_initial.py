"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-06
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telegram_user_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(length=255)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("telegram_user_id"),
    )
    op.create_index("ix_users_telegram_user_id", "users", ["telegram_user_id"])
    op.create_table(
        "recognition_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("telegram_user_id", sa.BigInteger(), sa.ForeignKey("users.telegram_user_id"), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=48), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_recognition_jobs_telegram_user_id", "recognition_jobs", ["telegram_user_id"])
    op.create_table(
        "recognition_fields",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Uuid(), sa.ForeignKey("recognition_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_name", sa.String(length=40), nullable=False),
        sa.Column("recognized_value", sa.Text(), nullable=False),
        sa.Column("corrected_value", sa.Text()),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("is_correct", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_recognition_fields_job_id", "recognition_fields", ["job_id"])
    op.create_table(
        "corrections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Uuid(), sa.ForeignKey("recognition_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_name", sa.String(length=40), nullable=False),
        sa.Column("old_value", sa.Text(), nullable=False),
        sa.Column("new_value", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_corrections_job_id", "corrections", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_corrections_job_id", table_name="corrections")
    op.drop_table("corrections")
    op.drop_index("ix_recognition_fields_job_id", table_name="recognition_fields")
    op.drop_table("recognition_fields")
    op.drop_index("ix_recognition_jobs_telegram_user_id", table_name="recognition_jobs")
    op.drop_table("recognition_jobs")
    op.drop_index("ix_users_telegram_user_id", table_name="users")
    op.drop_table("users")
