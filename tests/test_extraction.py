from app.ocr.extraction import normalize_date, normalize_number, normalize_phone
from app.schemas import FieldResult, RecognitionResult
from app.services.validation import validate_result


def test_normalizes_recognized_values() -> None:
    assert normalize_number("043 / 008568") == "043/008568"
    assert normalize_date("06-10-2026 15:04:31") == "06.10.2026 15:04:31"
    assert normalize_phone("8 (904) 326-13-01") == "+7 (904) 326-13-01"


def test_validation_accepts_complete_application() -> None:
    result = RecognitionResult(
        number=FieldResult(value="043 / 008568", confidence=0.8),
        date=FieldResult(value="06.10.2026 15:04:31", confidence=0.8),
        counterparty=FieldResult(value="Безукладов Владислав", confidence=0.8),
        phone=FieldResult(value="89043261301", confidence=0.8),
        comment=FieldResult(value="Обновление 1С КФХ базовая", confidence=0.8),
        manager=FieldResult(value="Аксенова Виктория", confidence=0.8),
        task_status=FieldResult(value="Не начато", confidence=1.0),
    )
    validated = validate_result(result)
    assert validated.number.value == "043/008568"
    assert validated.phone.value == "+7 (904) 326-13-01"
