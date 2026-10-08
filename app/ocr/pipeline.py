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

    @staticmethod
    def _log_classic_tokens(tokens) -> None:
        logger.info(
            "========== CLASSIC PADDLEOCR RESULT ==========",
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
            "========== END CLASSIC PADDLEOCR RESULT ==========",
        )

    @staticmethod
    def _merge_field(
        fields: dict[str, FieldResult],
        classic_fields: dict[str, FieldResult],
        name: str,
    ) -> None:
        vl_field = fields.get(name, FieldResult())
        classic_field = classic_fields.get(name, FieldResult())

        # Контрагент у classic OCR сейчас определяется заметно точнее.
        # VL на этом скриншоте ошибочно объединяет соседние строки.
        if name == "counterparty":
            if classic_field.value:
                fields[name] = classic_field

                logger.info(
                    "FIELD CLASSIC OVERRIDE %-15s value=%r confidence=%.3f",
                    name,
                    classic_field.value,
                    classic_field.confidence,
                )

            return

        # Для остальных полей classic OCR используется только
        # если VL ничего не нашёл.
        if vl_field.value:
            return

        if classic_field.value:
            fields[name] = classic_field

            logger.info(
                "FIELD FALLBACK %-15s value=%r confidence=%.3f",
                name,
                classic_field.value,
                classic_field.confidence,
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

        fields: dict[str, FieldResult] | None = None

        # ================================================================
        # 1. PaddleOCR-VL
        # ================================================================

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
                    "PaddleOCR-VL failed. "
                    "Falling back to classic PaddleOCR.",
                )
                fields = None

        # ================================================================
        # 2. Если VL полностью упал
        # ================================================================

        if fields is None:
            logger.info(
                "========== START CLASSIC PADDLEOCR ==========",
            )

            tokens = self.provider.recognize(
                image_path,
            )

            self._log_classic_tokens(tokens)

            fields = extract_from_full_text(tokens)

        # ================================================================
        # 3. VL отработал, но не нашёл отдельные поля
        #
        # Сейчас это особенно важно для:
        #
        # phone
        # comment
        #
        # VL может хорошо понять структуру документа,
        # но пропустить мелкий текст.
        # ================================================================

        else:
            missing_fields = [
                name
                for name in (
                    "counterparty",
                    "phone",
                    "comment",
                )
                if not fields.get(
                    name,
                    FieldResult(),
                ).value
            ]

            if missing_fields:
                logger.info(
                    "VL fields missing: %s. "
                    "Running classic PaddleOCR fallback.",
                    ", ".join(missing_fields),
                )

                try:
                    logger.info(
                        "========== "
                        "START CLASSIC PADDLEOCR FALLBACK "
                        "==========",
                    )

                    tokens = self.provider.recognize(
                        image_path,
                    )

                    self._log_classic_tokens(tokens)

                    classic_fields = extract_from_full_text(
                        tokens,
                    )

                    logger.info(
                        "CLASSIC FALLBACK FIELDS: %s",
                        {
                            name: field.value
                            for name, field in classic_fields.items()
                        },
                    )

                    for name in missing_fields:
                        self._merge_field(
                            fields,
                            classic_fields,
                            name,
                        )

                    logger.info(
                        "========== "
                        "END CLASSIC PADDLEOCR FALLBACK "
                        "==========",
                    )

                except Exception:
                    logger.exception(
                        "Classic PaddleOCR fallback failed. "
                        "Keeping VL result.",
                    )

        # ================================================================
        # 4. Статус задачи
        # ================================================================

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
