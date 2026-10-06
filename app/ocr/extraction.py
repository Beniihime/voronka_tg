import re
from datetime import datetime

from app.ocr.provider import OCRToken
from app.schemas import FieldResult

# Common OCR confusions are accepted here and normalised below; validation still
# requires the final strict 000/000000 format.
NUMBER_RE = re.compile(r"(?<!\d)([\dOО]{3}\s*/\s*[\dOОS]{6})(?!\d)")
DATE_RE = re.compile(r"(?<!\d)(\d{2}[.\-/]\d{2}[.\-/]\d{4}\s+\d{2}:\d{2}:\d{2})(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)[\s()\-]*([0-9][\s()\-]*[0-9][\s()\-]*[0-9])[\s()\-]*([0-9]{3})[\s()\-]*([0-9]{2})[\s()\-]*([0-9]{2})(?!\d)")


def clean_text(text: str) -> str:
    return " ".join(text.replace("|", " ").split()).strip()


def normalize_number(value: str) -> str:
    return re.sub(r"\s", "", value).replace("O", "0").replace("О", "0").replace("S", "5")


def normalize_date(value: str) -> str:
    candidate = value.replace("/", ".").replace("-", ".")
    try:
        return datetime.strptime(candidate, "%d.%m.%Y %H:%M:%S").strftime("%d.%m.%Y %H:%M:%S")
    except ValueError:
        return candidate


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits[0] in "78":
        digits = "7" + digits[1:]
    if len(digits) != 11 or not digits.startswith("7"):
        return clean_text(value)
    return f"+7 ({digits[1:4]}) {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"


def confidence_for(value: str, token_confidence: float, field_name: str) -> float:
    if not value:
        return 0.0
    score = min(1.0, max(0.0, token_confidence))
    if field_name == "number":
        score *= 1.0 if re.fullmatch(r"\d{3}/\d{6}", value) else 0.45
    elif field_name == "date":
        try:
            datetime.strptime(value, "%d.%m.%Y %H:%M:%S")
        except ValueError:
            score *= 0.45
    elif field_name == "phone":
        score *= 1.0 if re.fullmatch(r"\+7 \(\d{3}\) \d{3}-\d{2}-\d{2}", value) else 0.45
    elif len(value) < 2:
        score *= 0.3
    return round(score, 3)


def extract_pattern(tokens: list[OCRToken], field_name: str) -> FieldResult:
    raw = " ".join(token.text for token in tokens)
    patterns = {"number": NUMBER_RE, "date": DATE_RE, "phone": PHONE_RE}
    match = patterns[field_name].search(raw)
    if not match:
        return FieldResult(value="", confidence=0.0, raw_text=raw or None)
    value = match.group(0)
    value = {"number": normalize_number, "date": normalize_date, "phone": normalize_phone}[field_name](value)
    supporting = next((t.confidence for t in tokens if match.group(0) in t.text or t.text in match.group(0)), 0.75)
    return FieldResult(value=value, confidence=confidence_for(value, supporting, field_name), raw_text=raw)


def result_from_roi(field_name: str, tokens: list[OCRToken]) -> FieldResult:
    if field_name in {"number", "date", "phone"}:
        return extract_pattern(tokens, field_name)
    raw = clean_text(" ".join(t.text for t in tokens))
    if field_name == "manager":
        raw = re.sub(r"^(?:Автор|Менеджер)\s*:?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.split(r"\s+Редактор\s*:?", raw, flags=re.IGNORECASE)[0].strip()
    confidence = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.0
    return FieldResult(value=raw, confidence=confidence_for(raw, confidence, field_name), raw_text=raw or None)


def extract_labeled_value(
    tokens: list[OCRToken], label: str, stop_labels: tuple[str, ...], *, break_on_stop: bool = False
) -> FieldResult:
    """Fallback for plain-text fields when a configured ROI misses its target."""
    for index, token in enumerate(tokens):
        text = clean_text(token.text)
        if label.lower() not in text.lower():
            continue
        suffix = re.split(re.escape(label), text, flags=re.IGNORECASE, maxsplit=1)[-1].lstrip(": ")
        # The 1C tab title is often OCR'd as "Что сделано (заполняется исполнителем)".
        suffix = re.sub(r"\([^)]*\)", "", suffix).strip()
        if suffix:
            return FieldResult(value=suffix, confidence=confidence_for(suffix, token.confidence, "text"), raw_text=text)
        for next_token in tokens[index + 1:]:
            candidate = clean_text(next_token.text)
            if not candidate:
                continue
            if any(marker.lower() in candidate.lower() for marker in stop_labels):
                if break_on_stop:
                    return FieldResult()
                continue
            return FieldResult(value=candidate, confidence=confidence_for(candidate, next_token.confidence, "text"),
                               raw_text=candidate)
    return FieldResult()


def extract_from_full_text(tokens: list[OCRToken]) -> dict[str, FieldResult]:
    """Fallback extraction from a complete screenshot rather than an ROI."""
    extracted = {name: extract_pattern(tokens, name) for name in ("number", "date", "phone")}
    extracted["counterparty"] = extract_labeled_value(
        tokens, "Контрагент", ("Конт. лицо", "Конт лицо", "Телефон", "Территория", "Описание"), break_on_stop=True
    )
    extracted["work"] = extract_labeled_value(
        tokens, "Что сделано", ("Ремонт оборудования", "Комментарий", "Дополнительно")
    )
    extracted["manager"] = extract_labeled_value(
        tokens, "Менеджер", ("Редактор", "Номер", "Дата", "Контрагент"), break_on_stop=True
    )
    if not extracted["manager"].value:
        extracted["manager"] = extract_labeled_value(
            tokens, "Автор", ("Редактор", "Номер", "Дата", "Контрагент"), break_on_stop=True
        )
    return extracted
