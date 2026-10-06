from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    value: str = ""
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.0
    raw_text: str | None = None


class RecognitionResult(BaseModel):
    number: FieldResult
    date: FieldResult
    counterparty: FieldResult
    phone: FieldResult
    comment: FieldResult
    manager: FieldResult
    task_status: FieldResult

    def fields(self) -> dict[str, FieldResult]:
        return {name: getattr(self, name) for name in (
            "number", "date", "counterparty", "phone", "comment", "manager", "task_status"
        )}

    @property
    def has_low_confidence(self) -> bool:
        from app.config import get_settings
        return any(v.confidence < get_settings().low_confidence_threshold for v in self.fields().values())


class JobResponse(BaseModel):
    id: str
    status: str
    telegram_user_id: int
    created_at: datetime
    updated_at: datetime


class JobResultResponse(JobResponse):
    result: RecognitionResult
