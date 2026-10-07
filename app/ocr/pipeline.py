from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.ocr.extraction import extract_from_full_text
from app.ocr.paddle_provider import PaddleOCRProvider
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
            self.settings.ocr_language
        )

    def recognize(
        self,
        image_path: str | Path,
    ) -> RecognitionResult:
        image_path = Path(image_path)

        logger.info(
            "Starting OCR for image: %s",
            image_path,
        )

        # =====================================================
        # ЕДИНСТВЕННЫЙ OCR-ПРОХОД
        # =====================================================

        tokens = self.provider.recognize(
            image_path
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
            "========== END OCR RESULT =========="
        )

        # =====================================================
        # EXTRACTION
        # =====================================================

        fields = extract_from_full_text(
            tokens
        )

        fields["task_status"] = FieldResult(
            value="Не начато",
            confidence=1.0,
        )

        result = RecognitionResult(
            **fields
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