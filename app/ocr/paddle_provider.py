import logging
from pathlib import Path

from app.ocr.provider import OCRProvider, OCRToken

logger = logging.getLogger(__name__)


class PaddleOCRProvider(OCRProvider):
    """Local Russian OCR. Model files are fetched by Paddle on its first run."""

    def __init__(self, language: str = "ru") -> None:
        self.language = language
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("PaddleOCR is not installed. Install requirements.txt.") from exc
            # PaddleOCR 3.x removed the 2.x-only ``show_log`` and
            # ``use_angle_cls`` constructor arguments. The default pipeline
            # enables the appropriate document-orientation stages itself.
            self._engine = PaddleOCR(lang=self.language)
        return self._engine

    def recognize(self, image_path: Path) -> list[OCRToken]:
        engine = self._get_engine()
        # In PaddleOCR 3.x ``ocr`` is only a legacy wrapper that forwards
        # arguments to ``predict``. Calling it with the old ``cls`` argument
        # fails, so use the supported pipeline API directly.
        result = engine.predict(str(image_path))
        tokens: list[OCRToken] = []
        # PaddleOCR 2.x: [[ [box, (text, confidence)], ... ]]
        for page in result or []:
            if isinstance(page, dict) and "rec_texts" in page:
                tokens.extend(OCRToken(str(text), float(score)) for text, score in zip(
                    page["rec_texts"], page.get("rec_scores", [])
                ))
                continue
            page_json = getattr(page, "json", None)
            if isinstance(page_json, dict) and "rec_texts" in page_json:
                tokens.extend(OCRToken(str(text), float(score)) for text, score in zip(
                    page_json["rec_texts"], page_json.get("rec_scores", [])
                ))
                continue
            for line in page or []:
                if len(line) >= 2 and isinstance(line[1], tuple):
                    text, confidence = line[1]
                    tokens.append(OCRToken(text=str(text), confidence=float(confidence)))
        logger.debug("PaddleOCR returned %s text tokens", len(tokens))
        return tokens
