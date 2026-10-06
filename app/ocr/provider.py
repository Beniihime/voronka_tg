from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float


class OCRProvider(ABC):
    """Swappable OCR boundary. Cloud providers can implement this contract later."""

    @abstractmethod
    def recognize(self, image_path: Path) -> list[OCRToken]:
        raise NotImplementedError
