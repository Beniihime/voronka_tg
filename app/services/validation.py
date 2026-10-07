from datetime import datetime
import re

from app.ocr.extraction import normalize_number, normalize_phone
from app.schemas import RecognitionResult

class ValidationError(ValueError):
    pass


def validate_result(result: RecognitionResult) -> RecognitionResult:
    result.number.value = normalize_number(result.number.value)
    result.phone.value = normalize_phone(result.phone.value)
    try:
        datetime.strptime(result.date.value, "%d.%m.%Y %H:%M:%S")
    except ValueError as exc:
        raise ValidationError("Дата должна быть в формате ДД.ММ.ГГГГ ЧЧ:ММ:СС.") from exc
    if not re.fullmatch(r"\d{3}/\d{6}", result.number.value):
        raise ValidationError("Номер должен иметь формат 000/000000.")
    if not re.fullmatch(r"\+7 \(\d{3}\) \d{3}-\d{2}-\d{2}", result.phone.value):
        raise ValidationError("Телефон должен быть российским номером.")
    for name in ("counterparty", "comment", "manager"):
        if not getattr(result, name).value.strip():
            raise ValidationError(f"Поле «{name}» не должно быть пустым.")
    if not result.task_status.value.strip():
        raise ValidationError(
            "Статус задачи не должен быть пустым."
        )
    return result
