import sys
from types import SimpleNamespace
from pathlib import Path

from app.ocr.paddle_provider import PaddleOCRProvider


def test_creates_paddleocr_v3_without_removed_constructor_arguments(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    class PaddleOCRv3:
        def __init__(self, *, lang: str) -> None:
            calls.append({"lang": lang})

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=PaddleOCRv3))

    provider = PaddleOCRProvider(language="ru")
    assert isinstance(provider._get_engine(), PaddleOCRv3)
    assert calls == [{"lang": "ru"}]


def test_uses_predict_not_legacy_ocr_wrapper_for_paddleocr_v3() -> None:
    calls: list[str] = []

    class PaddleOCRv3:
        def ocr(self, *_args, **_kwargs):
            raise AssertionError("PaddleOCR 3 legacy ocr() must not be called")

        def predict(self, image_path: str):
            calls.append(image_path)
            return [{"rec_texts": ["043/008568"], "rec_scores": [0.98]}]

    provider = PaddleOCRProvider()
    provider._engine = PaddleOCRv3()

    tokens = provider.recognize(Path("/tmp/order.png"))

    assert calls == ["/tmp/order.png"]
    assert [(item.text, item.confidence) for item in tokens] == [("043/008568", 0.98)]
