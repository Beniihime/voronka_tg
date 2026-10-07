import json
import logging
from pathlib import Path

from app.ocr.provider import OCRProvider, OCRToken

logger = logging.getLogger(__name__)


def _as_box(value) -> tuple[tuple[float, float], ...] | None:
    """
    Convert PaddleOCR polygon/box into a normalized tuple of points.

    Supports:
        [[x1, y1], [x2, y2], ...]
        [x1, y1, x2, y2]
    """
    if value is None:
        return None

    try:
        # Polygon: [[x, y], [x, y], ...]
        if (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], (list, tuple))
        ):
            points = tuple(
                (float(point[0]), float(point[1]))
                for point in value
            )
            return points if points else None

        # Rectangle: [x1, y1, x2, y2]
        if isinstance(value, (list, tuple)) and len(value) == 4:
            x1, y1, x2, y2 = map(float, value)
            return (
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            )
    except (TypeError, ValueError, IndexError):
        return None

    return None


def _page_dict(page) -> dict | None:
    """
    Extract PaddleOCR result data from different PaddleOCR 3.x result forms.
    """
    if isinstance(page, dict):
        return page

    page_json = getattr(page, "json", None)

    if isinstance(page_json, dict):
        return page_json

    if isinstance(page_json, str):
        try:
            parsed = json.loads(page_json)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    return None


class PaddleOCRProvider(OCRProvider):
    """
    Local Russian OCR.

    Important:
    One image -> exactly one PaddleOCR.predict() call.
    """

    def __init__(self, language: str = "ru") -> None:
        self.language = language
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed. "
                    "Install requirements.txt."
                ) from exc

            self._engine = PaddleOCR(lang=self.language)

        return self._engine

    def recognize(self, image_path: Path) -> list[OCRToken]:
        engine = self._get_engine()

        # IMPORTANT:
        # exactly one OCR call for the whole image.
        result = engine.predict(str(image_path))

        tokens: list[OCRToken] = []

        for page in result or []:
            data = _page_dict(page)

            if data is not None and "rec_texts" in data:
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])

                # PaddleOCR 3.x normally provides rec_boxes.
                # dt_polys is kept as a fallback.
                boxes = data.get(
                    "rec_boxes",
                    data.get("dt_polys", []),
                )

                for index, text in enumerate(texts):
                    score = (
                        float(scores[index])
                        if index < len(scores)
                        else 0.0
                    )

                    box = (
                        _as_box(boxes[index])
                        if index < len(boxes)
                        else None
                    )

                    tokens.append(
                        OCRToken(
                            text=str(text),
                            confidence=score,
                            box=box,
                        )
                    )

                continue

            # Compatibility with old PaddleOCR output:
            #
            # [
            #     [
            #         [box, (text, confidence)],
            #         ...
            #     ]
            # ]
            for line in page or []:
                if len(line) >= 2 and isinstance(line[1], tuple):
                    text, confidence = line[1]

                    box = (
                        _as_box(line[0])
                        if line
                        else None
                    )

                    tokens.append(
                        OCRToken(
                            text=str(text),
                            confidence=float(confidence),
                            box=box,
                        )
                    )

        logger.debug(
            "PaddleOCR returned %s text tokens",
            len(tokens),
        )

        return tokens