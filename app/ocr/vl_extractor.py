from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from app.schemas import FieldResult

logger = logging.getLogger(__name__)


NUMBER_RE = re.compile(
    r"(?<!\d)([\dОO]{3}\s*/\s*[\dОOСC]{6})(?!\d)",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"(?<!\d)"
    r"(\d{2}[.\-/]\d{2}[.\-/]\d{4})"
    r"(?:\s*(\d{1,2})[:\-](\d{2})(?:[:\-](\d{2}))?)?"
    r"(?!\d)",
)

PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?7|8)"
    r"[\s()\-]*(\d)[\s()\-]*"
    r"(\d)[\s()\-]*(\d)[\s()\-]*"
    r"(\d{3})[\s()\-]*(\d{2})[\s()\-]*(\d{2})"
    r"(?!\d)",
)

AUTHOR_RE = re.compile(
    r"(?:^|\s)автор\s*[:|\-]?\s*(.*?)"
    r"(?=\s*[:|\-]?\s*редактор\s*[:|\-]?|$)",
    re.IGNORECASE | re.DOTALL,
)

COUNTERPARTY_LABEL_RE = re.compile(
    r"(?:контрагент|покупатель|заказчик)"
    r"\s*[:|\-]\s*([^\n|]+)",
    re.IGNORECASE,
)


class PaddleOCRVLExtractor:
    """
    PaddleOCR-VL document extractor.

    VLM is responsible for understanding the document structure.
    The Python layer extracts and validates the fields used by the bot.
    """

    def __init__(
        self,
        pipeline_version: str = "v1.6",
        device: str = "cpu",
    ) -> None:
        self.pipeline_version = pipeline_version
        self.device = device
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCRVL is unavailable. "
                "Install paddleocr[doc-parser]==3.6.0."
            ) from exc

        logger.info(
            "Initializing PaddleOCR-VL: version=%s device=%s",
            self.pipeline_version,
            self.device,
        )

        self._pipeline = PaddleOCRVL(
            pipeline_version=self.pipeline_version,
            device=self.device,
        )

        return self._pipeline

    @staticmethod
    def _call_attr(obj: Any, name: str) -> Any:
        value = getattr(obj, name, None)

        if callable(value):
            try:
                return value()
            except TypeError:
                return None

        return value

    @classmethod
    def _result_to_text(cls, result: Any) -> str:
        for attr_name in (
            "markdown",
            "text",
            "content",
        ):
            value = cls._call_attr(result, attr_name)

            if isinstance(value, str) and value.strip():
                return value

            if value is not None and not isinstance(
                value,
                (str, bytes),
            ):
                try:
                    rendered = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                    if rendered.strip():
                        return rendered
                except (TypeError, ValueError):
                    pass

        value = cls._call_attr(result, "json")

        if value is not None:
            if isinstance(value, str) and value.strip():
                return value

            try:
                rendered = json.dumps(
                    value,
                    ensure_ascii=False,
                )

                if rendered.strip():
                    return rendered
            except (TypeError, ValueError):
                pass

        save_to_markdown = getattr(
            result,
            "save_to_markdown",
            None,
        )

        if callable(save_to_markdown):
            with tempfile.TemporaryDirectory(
                prefix="paddleocr_vl_",
            ) as temp_dir:
                try:
                    save_to_markdown(
                        save_path=temp_dir,
                        pretty=False,
                    )
                except TypeError:
                    save_to_markdown(temp_dir)

                files = sorted(
                    Path(temp_dir).rglob("*.md"),
                )

                contents: list[str] = []

                for file_path in files:
                    try:
                        contents.append(
                            file_path.read_text(
                                encoding="utf-8",
                            ),
                        )
                    except OSError:
                        continue

                if contents:
                    return "\n\n".join(contents)

        return str(result)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _clean(value: str) -> str:
        value = value.strip()
        value = re.sub(r"\s+", " ", value)
        return value.strip(" :;|,-–—")

    @staticmethod
    def _normalize_number(value: str) -> str:
        value = value.upper().replace("О", "0").replace("O", "0")
        value = re.sub(r"\s+", "", value)

        match = NUMBER_RE.search(value)
        if not match:
            return value.strip()

        return (
            re.sub(r"\s+", "", match.group(1))
            .replace("О", "0")
            .replace("O", "0")
        )

    @staticmethod
    def _normalize_date(value: str) -> str:
        match = DATE_RE.search(value.strip())
        if not match:
            return value.strip()

        date_part = match.group(1).replace("/", ".").replace("-", ".")
        hour = match.group(2)
        minute = match.group(3)
        second = match.group(4)

        if not hour or not minute:
            return date_part

        if second:
            return f"{date_part} {hour}:{minute}:{second}"

        return f"{date_part} {hour}:{minute}"

    @staticmethod
    def _normalize_phone(value: str) -> str:
        digits = re.sub(r"\D", "", value)

        if len(digits) != 11:
            return value.strip()

        if digits.startswith("8"):
            digits = "7" + digits[1:]

        if not digits.startswith("7"):
            return value.strip()

        return (
            f"+7 ({digits[1:4]}) "
            f"{digits[4:7]}-"
            f"{digits[7:9]}-"
            f"{digits[9:11]}"
        )

    @classmethod
    def _extract_number(cls, text: str) -> FieldResult:
        match = NUMBER_RE.search(text)
        if not match:
            return FieldResult()

        raw = match.group(1)
        return FieldResult(
            value=cls._normalize_number(raw),
            confidence=0.95,
            raw_text=raw,
        )

    @classmethod
    def _extract_date(cls, text: str) -> FieldResult:
        labelled_patterns = (
            r"дата\s*[:|\-]\s*([^\n|]+)",
            r"дaта\s*[:|\-]\s*([^\n|]+)",
        )

        for pattern in labelled_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            date_match = DATE_RE.search(match.group(1))
            if not date_match:
                continue

            raw = date_match.group(0)
            return FieldResult(
                value=cls._normalize_date(raw),
                confidence=0.95,
                raw_text=raw,
            )

        match = DATE_RE.search(text)
        if not match:
            return FieldResult()

        raw = match.group(0)
        return FieldResult(
            value=cls._normalize_date(raw),
            confidence=0.75,
            raw_text=raw,
        )

    @classmethod
    def _extract_phone(cls, text: str) -> FieldResult:
        match = PHONE_RE.search(text)
        if not match:
            return FieldResult()

        raw = match.group(0)
        return FieldResult(
            value=cls._normalize_phone(raw),
            confidence=0.95,
            raw_text=raw,
        )

    @staticmethod
    def _normalize_counterparty(value: str) -> str:
        value = re.sub(r"\s+", " ", value.strip())
        value = re.sub(
            r"\s*\.\s*(?:000|ОО0|0ОО|ООО)$",
            " ООО",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"(?:000|ОО0|0ОО|OOO)$",
            "ООО",
            value,
            flags=re.IGNORECASE,
        )
        return value.strip()

    @classmethod
    def _extract_counterparty(cls, text: str) -> FieldResult:
        match = COUNTERPARTY_LABEL_RE.search(text)

        if match:
            value = cls._clean(match.group(1))
            if value:
                return FieldResult(
                    value=cls._normalize_counterparty(value),
                    confidence=0.93,
                    raw_text=match.group(0),
                )

        patterns = (
            r"([^\n|]{1,80}?\s+(?:ООО|ОOО|000))\b",
            r"\b(ИП\s+[^\n|]{2,80})",
            r"\b((?:АО|ОАО|ЗАО)\s+[^\n|]{2,80})",
        )

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            value = cls._clean(match.group(1))
            if value:
                return FieldResult(
                    value=cls._normalize_counterparty(value),
                    confidence=0.82,
                    raw_text=match.group(0),
                )

        return FieldResult()

    @classmethod
    def _extract_author(cls, text: str) -> FieldResult:
        match = AUTHOR_RE.search(text)
        if not match:
            return FieldResult()

        value = cls._clean(match.group(1))
        if not value:
            return FieldResult()

        return FieldResult(
            value=value.rstrip(" :;-"),
            confidence=0.95,
            raw_text=match.group(0),
        )

    @classmethod
    def _extract_comment(cls, text: str) -> FieldResult:
        lines = [
            cls._clean(line)
            for line in text.splitlines()
            if cls._clean(line)
        ]

        if not lines:
            return FieldResult()

        start_index: int | None = None

        for index, line in enumerate(lines):
            normalized = line.lower()
            if "что сделано" in normalized or normalized == "комментарий":
                start_index = index
                break

        if start_index is None:
            return FieldResult()

        result_lines: list[str] = []
        first_line = lines[start_index]

        same_line_match = re.search(
            r"(?:что\s+сделано[^:|]*|комментарий)\s*[:|]\s*(.+)$",
            first_line,
            re.IGNORECASE,
        )

        if same_line_match:
            value = cls._clean(same_line_match.group(1))
            if value:
                result_lines.append(value)

        stop_markers = (
            "пп клиента",
            "конфигурация",
            "номер релиза",
            "статус отзыва",
            "дата сдачи отзыва",
            "статус протокола",
            "статус подтверждения",
        )

        for line in lines[start_index + 1:]:
            normalized = line.lower()
            if any(marker in normalized for marker in stop_markers):
                break
            result_lines.append(line)

        result_lines = [
            line
            for line in result_lines
            if line.lower() not in {
                "акты",
                "отчеты сотрудника",
                "ремонт оборудования",
                "дополнительно",
            }
        ]

        value = "\n".join(result_lines).strip()
        if not value:
            return FieldResult()

        return FieldResult(
            value=value,
            confidence=0.90,
            raw_text=value,
        )

    @classmethod
    def parse_document_text(
        cls,
        text: str,
    ) -> dict[str, FieldResult]:
        text = cls._normalize_text(text)

        return {
            "number": cls._extract_number(text),
            "date": cls._extract_date(text),
            "counterparty": cls._extract_counterparty(text),
            "phone": cls._extract_phone(text),
            "comment": cls._extract_comment(text),
            "manager": cls._extract_author(text),
        }

    def extract(
        self,
        image_path: str | Path,
    ) -> dict[str, FieldResult]:
        image_path = Path(image_path)
        pipeline = self._get_pipeline()

        logger.info(
            "Starting PaddleOCR-VL for image: %s",
            image_path,
        )

        logger.info("PaddleOCR-VL model loadeиd. Starting predict()")

        results = pipeline.predict(str(image_path))

        logger.info("PaddleOCR-VL predict() completed")

        if not results:
            raise RuntimeError("PaddleOCR-VL returned no results.")

        documents: list[str] = []

        for index, result in enumerate(results):
            text = self._result_to_text(result)
            if not text.strip():
                continue

            logger.info(
                "========== PADDLEOCR-VL RESULT %s ==========",
                index,
            )
            logger.info("%s", text[:12000])
            logger.info(
                "========== END PADDLEOCR-VL RESULT %s ==========",
                index,
            )
            documents.append(text)

        if not documents:
            raise RuntimeError("PaddleOCR-VL returned empty document text.")

        return self.parse_document_text("\n\n".join(documents))
