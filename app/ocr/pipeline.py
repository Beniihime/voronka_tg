from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.ocr.extraction import extract_from_full_text
from app.ocr.paddle_provider import PaddleOCRProvider
from app.ocr.vl_extractor import PaddleOCRVLExtractor
from app.schemas import FieldResult, RecognitionResult

logger = logging.getLogger(__name__)

INVALID_COUNTERPARTIES = {
    "ип выезда",
    "тип выезда",
    "контрагент",
    "что сделано",
    "комментарий",
}


class RecognitionPipeline:
    def __init__(self, provider=None, settings=None):
        self.settings = settings or get_settings()
        self.provider = provider or PaddleOCRProvider(self.settings.ocr_language)
        self.vl_extractor = None

        if self.settings.paddle_vl_enabled:
            self.vl_extractor = PaddleOCRVLExtractor(
                pipeline_version=self.settings.paddle_vl_pipeline_version,
                device=self.settings.paddle_vl_device,
            )

    @staticmethod
    def _log_classic_tokens(tokens) -> None:
        logger.info("========== CLASSIC PADDLEOCR RESULT ==========")
        for i, token in enumerate(tokens):
            logger.info(
                "OCR[%03d] text=%r confidence=%.4f box=%s",
                i, token.text, token.confidence, token.box,
            )
        logger.info("========== END CLASSIC PADDLEOCR RESULT ==========")

    @staticmethod
    def _merge_field(
        fields: dict[str, FieldResult],
        classic_fields: dict[str, FieldResult],
        name: str,
    ) -> None:
        vl_field = fields.get(name, FieldResult())
        classic_field = classic_fields.get(name, FieldResult())
        vl_value = vl_field.value.strip()
        classic_value = classic_field.value.strip()

        if name == "counterparty":
            # Не заменяем корректное значение на UI-заголовок.
            classic_is_valid = (
                bool(classic_value)
                and classic_value.lower() not in INVALID_COUNTERPARTIES
            )
            vl_is_invalid = vl_value.lower() in INVALID_COUNTERPARTIES

            if classic_is_valid:
                fields[name] = classic_field
                logger.info(
                    "FIELD CLASSIC OVERRIDE %-15s value=%r confidence=%.3f",
                    name, classic_field.value, classic_field.confidence,
                )
            elif vl_is_invalid:
                fields[name] = FieldResult()
                logger.warning("Rejected UI label as counterparty: %r", vl_field.value)
            return

        # VL остаётся основным источником; classic OCR дополняет пустые поля.
        if vl_value:
            return

        if classic_value:
            fields[name] = classic_field
            logger.info(
                "FIELD FALLBACK %-15s value=%r confidence=%.3f",
                name, classic_field.value, classic_field.confidence,
            )

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        image_path = Path(image_path)
        logger.info("Starting recognition for image: %s", image_path)
        fields: dict[str, FieldResult] | None = None

        # 1. PaddleOCR-VL
        if self.vl_extractor is not None:
            try:
                logger.info("========== START PADDLEOCR-VL ==========")
                fields = self.vl_extractor.extract(image_path)
                logger.info("========== END PADDLEOCR-VL ==========")
                for name, field in fields.items():
                    logger.info(
                        "VL FIELD %-15s value=%r confidence=%.3f",
                        name, field.value, field.confidence,
                    )
            except Exception:
                logger.exception(
                    "PaddleOCR-VL failed. Falling back to classic PaddleOCR."
                )
                fields = None

        # 2. Если VL полностью упал, используем classic OCR.
        if fields is None:
            logger.info("========== START CLASSIC PADDLEOCR ==========")
            tokens = self.provider.recognize(image_path)
            self._log_classic_tokens(tokens)
            fields = extract_from_full_text(tokens)

        # 3. Classic OCR дополняет все пустые поля, а также исправляет
        # очевидную ошибку, когда UI-заголовок распознан как контрагент.
        else:
            names = ("number", "date", "counterparty", "phone", "comment", "manager")
            missing_fields = [
                name for name in names
                if (
                    not fields.get(name, FieldResult()).value.strip()
                    or (
                        name == "counterparty"
                        and fields.get(name, FieldResult()).value.strip().lower()
                        in INVALID_COUNTERPARTIES
                    )
                )
            ]

            if missing_fields:
                logger.info(
                    "VL fields missing/invalid: %s. Running classic PaddleOCR fallback.",
                    ", ".join(missing_fields),
                )
                try:
                    logger.info("========== START CLASSIC PADDLEOCR FALLBACK ==========")
                    tokens = self.provider.recognize(image_path)
                    self._log_classic_tokens(tokens)
                    classic_fields = extract_from_full_text(tokens)
                    logger.info(
                        "CLASSIC FALLBACK FIELDS: %s",
                        {name: field.value for name, field in classic_fields.items()},
                    )
                    for name in missing_fields:
                        self._merge_field(fields, classic_fields, name)
                    logger.info("========== END CLASSIC PADDLEOCR FALLBACK ==========")
                except Exception:
                    logger.exception(
                        "Classic PaddleOCR fallback failed. Keeping VL result."
                    )

        fields["task_status"] = FieldResult(value="Не начато", confidence=1.0)
        result = RecognitionResult(**fields)
        logger.info(
            "EXTRACTED RESULT: number=%r date=%r counterparty=%r phone=%r "
            "comment=%r manager=%r task_status=%r",
            result.number.value, result.date.value, result.counterparty.value,
            result.phone.value, result.comment.value, result.manager.value,
            result.task_status.value,
        )
        return result
