"""Export confirmed user-verified documents for OCR model improvement.

Usage: python -m app.training.export_dataset [--output dataset]
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import JobStatus, RecognitionJob
from app.db.session import SessionLocal
from app.services.repository import JobRepository


async def export(output: Path) -> int:
    image_dir = output / "images"
    annotation_dir = output / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    async with SessionLocal() as session:
        jobs = (await session.scalars(
            select(RecognitionJob).options(selectinload(RecognitionJob.fields)).where(
                RecognitionJob.status == JobStatus.CONFIRMED
            ).order_by(RecognitionJob.created_at)
        )).all()
        repository = JobRepository(session)
        exported = 0
        for job in jobs:
            source = Path(job.image_path)
            if not source.is_file():
                continue
            extension = source.suffix or ".jpg"
            name = f"{job.id}{extension}"
            shutil.copy2(source, image_dir / name)
            result = repository.result_for(job)
            document = {
                "image": name,
                "job_id": str(job.id),
                "fields": {key: value.value for key, value in result.fields().items()},
                "prediction": {field.field_name: field.recognized_value for field in job.fields},
            }
            (annotation_dir / f"{job.id}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            exported += 1
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export confirmed documents into an OCR dataset")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    count = asyncio.run(export(args.output))
    print(f"Exported {count} confirmed documents to {args.output}")


if __name__ == "__main__":
    main()
