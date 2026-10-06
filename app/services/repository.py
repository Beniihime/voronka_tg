import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Correction, RecognitionField, RecognitionJob, User
from app.schemas import FieldResult, RecognitionResult


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_job(self, telegram_user_id: int, username: str | None, image_path: str) -> RecognitionJob:
        user = await self.session.scalar(select(User).where(User.telegram_user_id == telegram_user_id))
        if user is None:
            user = User(telegram_user_id=telegram_user_id, username=username)
            self.session.add(user)
        elif username != user.username:
            user.username = username
        job = RecognitionJob(telegram_user_id=telegram_user_id, image_path=image_path)
        self.session.add(job)
        await self.session.flush()
        return job

    async def save_recognition(self, job: RecognitionJob, result: RecognitionResult) -> None:
        for name, field in result.fields().items():
            self.session.add(RecognitionField(
                job_id=job.id, field_name=name, recognized_value=field.value,
                confidence=field.confidence, is_correct=None,
            ))
        await self.session.flush()

    async def get_job(self, job_id: uuid.UUID, telegram_user_id: int | None = None) -> RecognitionJob | None:
        query = select(RecognitionJob).options(selectinload(RecognitionJob.fields)).where(RecognitionJob.id == job_id)
        if telegram_user_id is not None:
            query = query.where(RecognitionJob.telegram_user_id == telegram_user_id)
        return await self.session.scalar(query)

    @staticmethod
    def result_for(job: RecognitionJob) -> RecognitionResult:
        fields: dict[str, FieldResult] = {
            "manager": FieldResult(value="", confidence=0.0),
            "task_status": FieldResult(value="Не начато", confidence=1.0),
        }
        for item in job.fields:
            value = item.corrected_value if item.corrected_value is not None else item.recognized_value
            # Preserve results created before the source field was renamed.
            field_name = "comment" if item.field_name == "work" else item.field_name
            fields[field_name] = FieldResult(value=value, confidence=item.confidence, raw_text=item.recognized_value)
        return RecognitionResult(**fields)

    async def correct_field(self, job: RecognitionJob, field_name: str, new_value: str) -> None:
        field = next((item for item in job.fields if item.field_name == field_name), None)
        if field is None:
            raise KeyError(field_name)
        old_value = field.corrected_value if field.corrected_value is not None else field.recognized_value
        if old_value != new_value:
            self.session.add(Correction(job_id=job.id, field_name=field_name, old_value=old_value, new_value=new_value))
            field.corrected_value = new_value
            field.is_correct = new_value == field.recognized_value
        await self.session.flush()

    async def mark_confirmed(self, job: RecognitionJob) -> None:
        for field in job.fields:
            if field.is_correct is None:
                field.is_correct = True
        await self.session.flush()

    async def persist_effective_values(self, job: RecognitionJob, result: RecognitionResult) -> None:
        """Persist validation normalisation so Sheets and the training dataset agree."""
        for field in job.fields:
            normalized = result.fields()[field.field_name].value
            current = field.corrected_value if field.corrected_value is not None else field.recognized_value
            if current != normalized:
                self.session.add(Correction(
                    job_id=job.id, field_name=field.field_name,
                    old_value=current, new_value=normalized,
                ))
                field.corrected_value = normalized
                field.is_correct = normalized == field.recognized_value
        await self.session.flush()
