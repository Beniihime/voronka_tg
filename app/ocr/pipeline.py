import logging
import tempfile
import time
from pathlib import Path

from app.config import Settings, get_settings
from app.ocr.extraction import extract_from_full_text, result_from_roi
from app.ocr.image_processing import crop_relative_roi, load_roi_config, preprocess_image
from app.ocr.paddle_provider import PaddleOCRProvider
from app.ocr.provider import OCRProvider
from app.schemas import FieldResult, RecognitionResult

logger = logging.getLogger(__name__)


class RecognitionPipeline:
    def __init__(self, provider: OCRProvider | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or PaddleOCRProvider(self.settings.ocr_language)
        self.roi = load_roi_config(self.settings.roi_config_path)

    def recognize(self, image_path: Path) -> RecognitionResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="one-c-ocr-") as temporary_directory:
            workdir = Path(temporary_directory)
            prepared = workdir / "prepared.png"
            preprocess_image(image_path, prepared)
            full_tokens = self.provider.recognize(prepared)
            fields = {}
            for field_name in ("number", "date", "counterparty", "phone", "comment", "manager"):
                raw_roi_path = workdir / f"{field_name}-raw.png"
                roi_path = workdir / f"{field_name}.png"
                try:
                    crop_relative_roi(image_path, raw_roi_path, self.roi[field_name])
                    preprocess_image(raw_roi_path, roi_path)
                    fields[field_name] = result_from_roi(field_name, self.provider.recognize(roi_path))
                except (KeyError, ValueError) as exc:
                    logger.warning("ROI processing failed for %s: %s", field_name, exc)
                    fields[field_name] = result_from_roi(field_name, [])
            fallback = extract_from_full_text(full_tokens)
            fallback["comment"] = fallback.pop("work")
            for name, value in fallback.items():
                if not fields[name].value or value.confidence > fields[name].confidence:
                    fields[name] = value
            fields["task_status"] = FieldResult(value="Не начато", confidence=1.0)
        logger.info("OCR finished in %.2fs; confidences=%s", time.monotonic() - started,
                    {name: value.confidence for name, value in fields.items()})
        return RecognitionResult(**fields)
