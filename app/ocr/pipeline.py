from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.ocr.extraction import extract_from_full_text
from app.ocr.paddle_provider import PaddleOCRProvider
from app.ocr.vl_extractor import PaddleOCRVLExtractor
from app.schemas import FieldResult, RecognitionResult

logger = logging.getLogger(__name__)


class RecognitionPipeline:
    def __init__(
        self,
        provider=None,
        settings=None,
    ):
        self.settings = settings or get_settings()

        self.provider = provider or PaddleOCRProvider(
            self.settings.ocr_language,
        )

        self.vl_extractor = None

        if self.settings.paddle_vl_enabled:
            self.vl_extractor = PaddleOCRVLExtractor(
                pipeline_version=self.settings.paddle_vl_pipeline_version,
                device=self.settings.paddle_vl_device,
            )

    def recognize(
        self,
        image_path: str | Path,
    ) -> RecognitionResult:
        image_path = Path(image_path)

        logger.info(
            "Starting recognition for image: %s",
            image_path,
        )

        fields = None

        if self.vl_extractor is not None:
            try:
                logger.info(
                    "========== START PADDLEOCR-VL ==========",
                )

                fields = self.vl_extractor.extract(
                    image_path,
                )

                logger.info(
                    "========== END PADDLEOCR-VL ==========",
                )

                for name, field in fields.items():
                    logger.info(
                        "VL FIELD %-15s value=%r confidence=%.3f",
                        name,
                        field.value,
                        field.confidence,
                    )

            except Exception:
                logger.exception(
                    "PaddleOCR-VL failed. Falling back to classic PaddleOCR.",
                )
                fields = None

        if fields is None:
            logger.info(
                "========== START CLASSIC PADDLEOCR ==========",
            )

            tokens = self.provider.recognize(
                image_path,
            )

            logger.info(
                "========== OCR RESULT: %s ==========",
                image_path,
            )

            for i, token in enumerate(tokens):
                logger.info(
                    "OCR[%03d] text=%r confidence=%.4f box=%s",
                    i,
                    token.text,
                    token.confidence,
                    token.box,
                )

            logger.info(
                "========== END OCR RESULT ==========",
            )

            fields = extract_from_full_text(tokens)

        fields["task_status"] = FieldResult(
            value="Не начато",
            confidence=1.0,
        )

        result = RecognitionResult(
            **fields,
        )

        logger.info(
            "EXTRACTED RESULT: "
            "number=%r "
            "date=%r "
            "counterparty=%r "
            "phone=%r "
            "comment=%r "
            "manager=%r "
            "task_status=%r",
            result.number.value,
            result.date.value,
            result.counterparty.value,
            result.phone.value,
            result.comment.value,
            result.manager.value,
            result.task_status.value,
        )

        return result
