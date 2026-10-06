import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class JobStatus(StrEnum):
    RECOGNIZING = "recognizing"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_DUPLICATE_DECISION = "awaiting_duplicate_decision"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    jobs: Mapped[list["RecognitionJob"]] = relationship(back_populates="user")


class RecognitionJob(Base):
    __tablename__ = "recognition_jobs"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.telegram_user_id"), index=True)
    image_path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(48), default=JobStatus.RECOGNIZING)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user: Mapped[User] = relationship(back_populates="jobs")
    fields: Mapped[list["RecognitionField"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    corrections: Mapped[list["Correction"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class RecognitionField(Base):
    __tablename__ = "recognition_fields"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("recognition_jobs.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str] = mapped_column(String(40))
    recognized_value: Mapped[str] = mapped_column(Text, default="")
    corrected_value: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    is_correct: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    job: Mapped[RecognitionJob] = relationship(back_populates="fields")


class Correction(Base):
    __tablename__ = "corrections"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("recognition_jobs.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str] = mapped_column(String(40))
    old_value: Mapped[str] = mapped_column(Text)
    new_value: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    job: Mapped[RecognitionJob] = relationship(back_populates="corrections")
