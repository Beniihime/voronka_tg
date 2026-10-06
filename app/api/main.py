import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_session
from app.logging import configure_logging
from app.schemas import JobResponse, JobResultResponse
from app.services.repository import JobRepository

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(settings.log_level)
    yield


app = FastAPI(title="1C Work Order Recognition API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> JobResponse:
    job = await JobRepository(session).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(id=str(job.id), status=job.status, telegram_user_id=job.telegram_user_id,
                       created_at=job.created_at, updated_at=job.updated_at)


@app.get("/api/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> JobResultResponse:
    repo = JobRepository(session)
    job = await repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if len(job.fields) != 5:
        raise HTTPException(status_code=409, detail="Recognition result is not available yet")
    return JobResultResponse(id=str(job.id), status=job.status, telegram_user_id=job.telegram_user_id,
                             created_at=job.created_at, updated_at=job.updated_at, result=repo.result_for(job))
